"""LocalEnvironment subclass that routes Python invocations to a long-lived
IPython kernel so heavy imports (``import scanpy``, ``import omicverse``,
fitting a doublet model, …) happen *once* per task instead of once per
agent turn.

This is the cookbook "Subclassing the environment" pattern from
mini-swe-agent's docs, applied to scientific-Python workflows where the
fresh-subshell-per-command default loses ~5s of `import` overhead per
turn — over 20 turns that is the whole task budget.

The agent still emits plain bash. Detection is silent:

- ``python -c "<code>"``           → kernel.execute(<code>)
- ``python <<EOF\\n<code>\\nEOF``  → kernel.execute(<code>)
- ``python <script>.py``           → kernel.execute(<contents of script>)

Everything else (``ls``, ``cat``, ``mkdir``, ``cd && rm`` …) goes through
``subprocess.run`` via the parent class — preserving the full bash
interface the model already knows.

Kernel state persists across turns: ``adata`` set in one ``python -c``
remains available in the next. The agent is told this in the system
prompt so it can rely on the persistence.
"""

from __future__ import annotations

import atexit
import logging
import os
import queue
import re
import time
from pathlib import Path
from typing import Any

from minisweagent.environments.local import LocalEnvironment

logger = logging.getLogger("persistent_env")


# python[3]? -c "..." or python -c '...'   (DOTALL so multi-line code is OK)
_PY_DASH_C = re.compile(
    r"^\s*(?:cd\s+[^\s&|;]+\s*&&\s*)?"
    r"python3?\s+-c\s+"
    r"(['\"])(.*?)\1\s*$",
    re.DOTALL,
)

# python[3]? <<EOF / <<'EOF' / <<-EOF / <<"EOF" — any unquoted word as sentinel
_PY_HEREDOC = re.compile(
    r"^\s*(?:cd\s+[^\s&|;]+\s*&&\s*)?"
    r"python3?\s*<<-?\s*['\"]?(\w+)['\"]?\s*\n"
    r"(.*?)\n"
    r"\1\s*$",
    re.DOTALL,
)

# python[3]? script.py  — bare script invocation
_PY_SCRIPT = re.compile(
    r"^\s*(?:cd\s+[^\s&|;]+\s*&&\s*)?"
    r"python3?\s+([^\s|<>&;()`]+\.py)\s*$",
)

# strip ANSI color from kernel tracebacks
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


class PersistentKernelEnvironment(LocalEnvironment):
    """LocalEnvironment that runs Python via a persistent IPython kernel."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._km = None
        self._kc = None
        atexit.register(self.shutdown)

    # ------------------------------------------------------------------
    # kernel lifecycle
    # ------------------------------------------------------------------

    def _ensure_kernel(self) -> None:
        if self._km is not None and self._km.is_alive():
            return
        from jupyter_client.manager import start_new_kernel, KernelManager
        from jupyter_client.kernelspec import KernelSpec

        env = os.environ.copy()
        env.update(self.config.env)
        cwd = self.config.cwd or os.getcwd()

        # Resolve the python interpreter the kernel will run.  ``kernel_name=
        # "python3"`` looks up jupyter's registered kernelspec, which is
        # baked at install time and ignores the env's PATH.  Bench tasks
        # opt into specific conda envs via ``conda_env`` so we *must* honor
        # env['PATH'] / env['CONDA_PREFIX'] here.  Build an explicit kernel
        # command pointing at the desired python's ``ipykernel_launcher``.
        py_bin = None
        path_dirs = env.get("PATH", "").split(os.pathsep)
        for d in path_dirs:
            cand = os.path.join(d, "python")
            if os.access(cand, os.X_OK):
                py_bin = cand
                break
        if py_bin is None:
            conda_prefix = env.get("CONDA_PREFIX")
            if conda_prefix:
                cand = os.path.join(conda_prefix, "bin", "python")
                if os.access(cand, os.X_OK):
                    py_bin = cand
        # Note: ``KernelManager.format_kernel_cmd`` special-cases the bare
        # names "python" / "python3" / "python3.X" by replacing them with
        # ``sys.executable`` (the launcher's interpreter, which is omicdev
        # for the bench harness).  Use the absolute path so that fallback
        # is bypassed.
        custom_spec: KernelSpec | None = None
        if py_bin is not None:
            custom_spec = KernelSpec(
                argv=[
                    py_bin, "-m", "ipykernel_launcher",
                    "-f", "{connection_file}",
                ],
                display_name="bench-python (resolved)",
                language="python",
            )

        # Retry kernel startup with random backoff. Concurrent benchmark
        # sweeps occasionally collide on ZMQ ports — ``start_new_kernel``
        # picks 5 ports (shell/iopub/stdin/heartbeat/control) and races
        # with sibling kernels in other processes. The error surfaces as
        # ``zmq.error.ZMQError: Address already in use`` propagated to
        # ``RuntimeError: Kernel died before replying to kernel_info``.
        # A few retries with jitter virtually always recover.
        import random
        last_exc: Exception | None = None
        for attempt in range(5):
            try:
                logger.info("starting persistent IPython kernel cwd=%s "
                             "py=%s (attempt %d)",
                             cwd, py_bin or "<default>", attempt + 1)
                if custom_spec is not None:
                    km = KernelManager(kernel_name="python3")
                    km._kernel_spec = custom_spec  # type: ignore[attr-defined]
                    km.start_kernel(cwd=cwd, env=env)
                    kc = km.client()
                    kc.start_channels()
                    try:
                        kc.wait_for_ready(timeout=60)
                    except RuntimeError:
                        kc.stop_channels()
                        km.shutdown_kernel(now=True)
                        raise
                    self._km, self._kc = km, kc
                else:
                    self._km, self._kc = start_new_kernel(
                        kernel_name="python3",
                        cwd=cwd,
                        env=env,
                    )
                break
            except Exception as exc:  # noqa: BLE001 — match kernel-died too
                last_exc = exc
                # tear down half-started manager if any
                try:
                    if self._km is not None:
                        self._km.shutdown_kernel(now=True)
                except Exception:
                    pass
                self._km = None
                self._kc = None
                err_str = str(exc).lower()
                if "address already in use" in err_str or "kernel died" in err_str \
                        or "kernel_info" in err_str:
                    sleep_s = 0.5 + random.random() * 1.5 + attempt * 0.5
                    logger.warning(
                        "kernel start collided on attempt %d (%s); "
                        "retrying after %.1fs", attempt + 1,
                        type(exc).__name__, sleep_s)
                    time.sleep(sleep_s)
                    continue
                raise
        else:
            raise RuntimeError(
                f"persistent kernel failed to start after 5 attempts: "
                f"{type(last_exc).__name__}: {last_exc}"
            ) from last_exc

        # Drain any startup chatter on iopub before our first execute.
        deadline = time.time() + 2.0
        while time.time() < deadline:
            try:
                self._kc.get_iopub_msg(timeout=0.1)
            except queue.Empty:
                break

    def shutdown(self) -> None:
        if self._km is not None:
            try:
                self._km.shutdown_kernel(now=True)
            except Exception:
                pass
            self._km = None
            self._kc = None

    # ------------------------------------------------------------------
    # python detection + execution
    # ------------------------------------------------------------------

    def _extract_python_code(self, command: str) -> str | None:
        m = _PY_DASH_C.match(command)
        if m:
            return m.group(2)
        m = _PY_HEREDOC.match(command)
        if m:
            return m.group(2)
        m = _PY_SCRIPT.match(command)
        if m:
            path = m.group(1)
            full = Path(self.config.cwd or os.getcwd()) / path
            try:
                code = full.read_text()
            except OSError:
                return None
            return f"__file__ = {str(full)!r}\n{code}"
        return None

    def _run_in_kernel(self, code: str, timeout: int) -> dict[str, Any]:
        self._ensure_kernel()
        # Re-pin cwd in the kernel namespace (handles agent-issued ``cd /x &&
        # python ...`` prefixes, which we strip during extraction).
        cwd = self.config.cwd or os.getcwd()
        full_code = f"import os as _os; _os.chdir({cwd!r})\n{code}"
        msg_id = self._kc.execute(full_code, store_history=False)

        chunks: list[str] = []
        returncode = 0
        deadline = time.time() + max(1, int(timeout))

        try:
            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    self._km.interrupt_kernel()
                    chunks.append(f"\n[kernel interrupted after {timeout}s]\n")
                    returncode = -1
                    end = time.time() + 2
                    while time.time() < end:
                        try:
                            self._kc.get_iopub_msg(timeout=0.5)
                        except queue.Empty:
                            break
                    break

                try:
                    msg = self._kc.get_iopub_msg(timeout=min(remaining, 1.0))
                except queue.Empty:
                    continue

                # ignore messages from prior executions
                if msg.get("parent_header", {}).get("msg_id") != msg_id:
                    continue

                mtype = msg["msg_type"]
                content = msg["content"]
                if mtype == "stream":
                    chunks.append(content.get("text", ""))
                elif mtype == "execute_result":
                    text = content.get("data", {}).get("text/plain", "")
                    chunks.append(text)
                    if not text.endswith("\n"):
                        chunks.append("\n")
                elif mtype == "error":
                    tb = "\n".join(content.get("traceback", []))
                    chunks.append(_ANSI.sub("", tb) + "\n")
                    returncode = 1
                elif mtype == "status" and content.get("execution_state") == "idle":
                    break
        except Exception as exc:
            chunks.append(f"\n[kernel client error: {type(exc).__name__}: {exc}]\n")
            returncode = 2
            self.shutdown()

        output = "".join(chunks)
        # No truncation: LocalEnvironment passes ``subprocess.run(...).stdout``
        # through verbatim, and `ov.utils.registry_summary()` /
        # ``registry_lookup()`` produce multi-kB strings the agent legitimately
        # needs to read. If a runaway loop ever produces gigabytes, we let
        # mini-swe-agent's per-command timeout catch it instead.
        return {"output": output, "returncode": returncode,
                "exception_info": ""}

    # ------------------------------------------------------------------
    # entry point
    # ------------------------------------------------------------------

    def execute(self, action: dict, cwd: str = "",
                *, timeout: int | None = None) -> dict[str, Any]:
        command = action.get("command", "")
        py_code = self._extract_python_code(command)
        if py_code is not None:
            t = int(timeout or self.config.timeout)
            output = self._run_in_kernel(py_code, t)
            # Honor mini-swe-agent's COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
            # protocol — _check_finished raises Submitted if the agent
            # printed that sentinel as the first line of output.
            try:
                self._check_finished(output)
            except Exception:
                raise
            return output
        return super().execute(action, cwd, timeout=timeout)
