"""mini-swe-agent adapter — the only test-taker for the v1 benchmark.

Two arms share **the exact same loop, model, environment, and tool surface**.
The only difference is the system prompt:

- ``mini_swe_omicverse``: base mini-swe system prompt + ``prompts/omicverse_system.md``
  appended (omicverse domain knowledge: registry_summary / registry_lookup /
  skill_lookup discipline, coding conventions).
- ``mini_swe_baseline``: base mini-swe system prompt only.

Both arms can ``import omicverse as ov`` — the ``omicdev`` conda env contains
the full scverse stack. The baseline arm has to discover that on its own;
the treatment arm is told. This is the cleanest "domain-prompt ablation"
the benchmark can offer: same agent loop, same tools, same model, same
data, only the system prompt differs.

Tool surface (from mini-swe-agent default): one ``bash`` toolcall per turn,
executed in a per-trajectory workspace via ``LocalEnvironment``. ``PATH``
is locked to ``<CONDA_PREFIX>/bin`` so ``python``,
``pip``, ``jupyter`` all resolve to the omicdev conda env regardless of
who launched the runner.

Termination: agent emits ``echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT``
to signal completion. mini-swe's LocalEnvironment catches it and raises
``Submitted``, which the agent loop converts to ``exit_status="Submitted"``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
import traceback
from pathlib import Path

from bench._paths import OMICDEV_PY, REPO_ROOT, run_dir

# Path to the merged single-agent omicverse system prompt (treatment arm).
OMICVERSE_SYSTEM_PROMPT_PATH = REPO_ROOT / "prompts" / "omicverse_system.md"

# Ablation: same prompt with the registry/skill_lookup discovery section
# removed. Used to attribute the omicverse skill effect to the registry
# channel vs. the rest of the prompt (reasoning discipline + dispatch
# wrappers + coding conventions). Selected via prompt_variant='no_registry'.
OMICVERSE_NO_REGISTRY_PROMPT_PATH = REPO_ROOT / "prompts" / "omicverse_system_no_registry.md"

# Ablation: doc-RAG. Same prompt scaffold but the discovery channel is a
# vanilla embedding-RAG over raw docstrings (``doc_lookup`` script) rather
# than the Beacon registry. Tests whether Beacon's gain is purely a
# documentation-retrieval thing or whether the structured library-side
# contract (aliases, requires/produces, prerequisites, auto_fix, skill
# metadata) adds something on top. Selected via prompt_variant='doc_rag'.
OMICVERSE_DOC_RAG_PROMPT_PATH = REPO_ROOT / "prompts" / "omicverse_system_doc_rag.md"

# omicdev conda env — default for every bash subshell the agent runs.
OMICDEV_BIN = str(Path(OMICDEV_PY).parent)
OMICDEV_PREFIX = str(Path(OMICDEV_PY).parents[1])

# Per-task conda-env registry. Tasks may set ``conda_env: <name>`` to opt
# into a non-default environment when their backends require packages
# not present in ``omicdev``. The registry maps env-name → (bin_dir,
# prefix_dir, default-name). ``omicverse`` (miniforge) carries the SCENIC
# stack (pyscenic, arboreto, boltons, ctxcore, distributed, squidpy)
# missing from omicdev.
_ENV_REGISTRY: dict[str, tuple[str, str, str]] = {
    "omicdev":   (OMICDEV_BIN, OMICDEV_PREFIX, "omicdev"),
    "omicverse": (
        "<MINIFORGE_HOME>/envs/omicverse/bin",
        "<MINIFORGE_HOME>/envs/omicverse",
        "omicverse",
    ),
}


def _resolve_env(env_name: str | None) -> tuple[str, str, str]:
    """Look up ``(bin_dir, prefix, default_env_name)`` for the named env.

    Falls back to ``omicdev`` for ``None`` or unknown names so a typo
    doesn't silently corrupt the run.
    """
    if env_name and env_name in _ENV_REGISTRY:
        return _ENV_REGISTRY[env_name]
    return _ENV_REGISTRY["omicdev"]

# Base system prompt for both arms. Tells the model the protocol (one bash
# tool call per turn, COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT to terminate)
# without any omicverse domain knowledge. Adapted from the upstream
# ``mini.yaml`` config but stripped of SWE-specific phrasing.
BASE_SYSTEM_TEMPLATE = """\
You are a careful research assistant solving omics data-analysis tasks.

You have access to one tool: a bash shell. Each response must include
reasoning text plus exactly one bash tool call.

## Execution model

Each bash command starts in your assigned workspace directory — your
input files (and the AnnData you save) live there. **Use relative
paths only** (``./file.csv``, ``adata.h5ad``); never write absolute
paths into your commands.

Bash subshells are fresh per turn, so ``cd`` and ``export`` do not
persist between turns. Re-set environment in files if you need it
across turns. Do not prefix commands with ``cd <path> && …`` — the
shell already starts in the right place.

**However, ``python -c "..."`` and ``python <<EOF...EOF`` (and
``python script.py``) are routed to a long-lived IPython kernel scoped
to this task.** Variables and imports you set in one ``python -c`` call
**remain available** in the next:

  turn k:    python -c "import scanpy as sc; adata = sc.read_h5ad('adata.h5ad'); print(adata.shape)"
  turn k+1:  python -c "print(list(adata.obs.columns))"   # adata is still defined

Use this. Import once, load ``adata`` once, modify in place across
turns, and ``adata.write_h5ad('adata.h5ad')`` at the end. Do not
re-import or re-load on every turn.

To reset the kernel state (rare), call ``python -c "%reset -f"`` — but
you almost never need to.

## Environment

The Python interpreter on ``PATH`` is the ``omicdev`` conda environment
with the scientific Python stack (anndata, scanpy, scvelo, numpy, pandas,
scikit-learn, …) preinstalled. Do not ``pip install`` packages.

## Coding style

The omicdev environment is fixed: you cannot install or remove
packages. Write the analysis you need; if a package is missing the
import line itself will say so, and you switch to a different one.
Spend turns on real analysis, not on probing the environment.

## Termination

To finish the task, run:

    echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT

as a single command. Do not combine it with anything else. After this
command you cannot make further changes.
"""

# Per-task user message. ``{{task}}`` is substituted by mini-swe-agent.
INSTANCE_TEMPLATE = """\
Solve the following omics analysis task.

Your bash shell starts in the workspace directory. Use **relative
paths** for all inputs and outputs — do not write absolute paths.

{{io_block}}

Required output: save the final modified AnnData as ``adata.h5ad`` in the
working directory. The grader will load that file and check biological /
numerical properties of the result.

Budget: at most {{step_limit}} bash commands; each command has a wallclock
timeout of {{timeout_s}}s. Use them efficiently.

Task:

<task_description>
{{task}}
</task_description>

When you are done, run ``echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`` as a
single command.
"""


def _load_omicverse_prompt(prompt_variant: str = "full") -> str:
    """Load the omicverse domain system prompt.

    ``prompt_variant``:
      - ``'full'`` (default): the canonical treatment prompt with the
        registry_lookup / skill_lookup discovery section.
      - ``'no_registry'``: ablation prompt — same content minus the
        Discovery / Lookup-discipline sections. Used to test whether
        the gain comes from the registry channel or the rest of the
        prompt.
    """
    path = OMICVERSE_SYSTEM_PROMPT_PATH
    if prompt_variant == "no_registry":
        path = OMICVERSE_NO_REGISTRY_PROMPT_PATH
    elif prompt_variant == "doc_rag":
        path = OMICVERSE_DOC_RAG_PROMPT_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"omicverse system prompt not found at {path}; "
            "run `git status` and check the prompts/ directory"
        )
    return path.read_text()


def _seed_workspace(task: dict, workspace: Path) -> tuple[str, str]:
    """Copy the fixture into ``workspace`` and return the (relative-path,
    io_description) the prompt should reference."""
    fixture = task.get("fixture")
    if not fixture:
        return "", "(no input dataset is provided for this task)"
    src = Path(fixture)
    if src.is_file():
        dst = workspace / "adata.h5ad"
        if not dst.exists():
            shutil.copy2(src, dst)
        return ("adata.h5ad",
                f"Input dataset: ``./adata.h5ad`` "
                f"(copied from `{src.name}`).")
    if src.is_dir():
        # Visium / multiome / 16S / bulk_dg directory fixtures: copy contents
        # into the workspace verbatim. The agent loads them with the right
        # reader (``sc.read_visium`` / ``sc.read_h5ad`` / etc.) and writes
        # the final result to ``adata.h5ad`` next to them.
        for item in src.iterdir():
            target = workspace / item.name
            if target.exists():
                continue
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)
        listing = ", ".join(sorted(p.name for p in workspace.iterdir())[:12])
        return ("",
                f"Input directory contents (in the working dir): {listing}. "
                f"Use the appropriate reader (e.g. ``sc.read_visium`` for "
                f"Visium data, ``sc.read_h5ad`` for ``.h5ad`` files, "
                f"``pd.read_csv`` for tables) to load them, do the analysis, "
                f"and save the final AnnData as ``adata.h5ad``.")
    raise FileNotFoundError(f"fixture path does not exist: {src}")


def _system_template(*, with_teams: bool, prompt_variant: str = "full") -> str:
    if not with_teams:
        return BASE_SYSTEM_TEMPLATE
    return BASE_SYSTEM_TEMPLATE + "\n\n---\n\n" + _load_omicverse_prompt(prompt_variant)


def _do_run(task: dict, model_id: str, seed: int, *, with_teams: bool,
             system_name: str | None = None,
             model_factory: str = "litellm_ollama",
             prompt_variant: str = "full") -> dict:
    """Run mini-swe-agent for one (task, prompt-arm, model) triple.

    ``model_factory`` selects how the LLM is reached:
      - ``"litellm_ollama"``: ``LitellmModel`` against local ollama (qwen3.6 etc.)
      - ``"gpt_oauth"``:    ``GptOauthModel`` against ChatGPT subscription (gpt-5.5)

    ``system_name`` overrides the directory tag under ``results/runs/``;
    defaults are derived from ``with_teams`` + ``model_factory``.
    """
    # Heavy imports deferred — mini-swe-agent prints a banner on import.
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.exceptions import LimitsExceeded, Submitted
    from bench.adapters.persistent_env import PersistentKernelEnvironment

    if system_name is None:
        if model_factory == "gpt_oauth":
            system_name = "gpt_omicverse" if with_teams else "gpt_baseline"
        elif model_factory == "litellm_deepseek":
            system_name = "deepseek_omicverse" if with_teams else "deepseek_baseline"
        elif model_factory == "litellm_openrouter":
            system_name = "openrouter_omicverse" if with_teams else "openrouter_baseline"
        elif model_factory == "litellm_gemini":
            system_name = "gemini_omicverse" if with_teams else "gemini_baseline"
        elif model_factory == "litellm_glm":
            system_name = "glm_omicverse" if with_teams else "glm_baseline"
        elif model_factory == "litellm_minimax":
            system_name = "minimax_omicverse" if with_teams else "minimax_baseline"
        else:
            system_name = "mini_swe_omicverse" if with_teams else "mini_swe_baseline"
    out_dir = run_dir(system_name, task["id"], model_id, seed)
    workspace = out_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "log.txt"
    log_handler = logging.FileHandler(log_path, mode="w")
    log_handler.setLevel(logging.INFO)
    log_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"))
    for name in ("agent", "litellm_model"):
        lg = logging.getLogger(name)
        lg.addHandler(log_handler)
        lg.setLevel(logging.INFO)

    try:
        rel_input, io_block = _seed_workspace(task, workspace)
    except Exception as exc:
        traceback.print_exc(file=log_path.open("a"))
        return {
            "final_adata_path": None, "final_text": None, "n_turns": 0,
            "wallclock_s": 0.0,
            "error": f"workspace_setup_failed: {type(exc).__name__}: {exc}",
        }

    # Lock PATH so `python` / `pip` / `jupyter` resolve to the env this
    # task asks for (default ``omicdev``). Tasks needing extra backends
    # (e.g. B10 SCENIC needs pyscenic / arboreto / boltons) opt in via
    # ``conda_env: omicverse`` in their task dict.
    task_env_name = task.get("conda_env") or "omicdev"
    env_bin, env_prefix, env_default = _resolve_env(task_env_name)
    locked_env = {
        "PATH":               f"{env_bin}:{os.environ.get('PATH', '')}",
        "CONDA_PREFIX":       env_prefix,
        "CONDA_DEFAULT_ENV":  env_default,
        # Avoid surprises from user's HOME-level dotfiles.
        "PYTHONNOUSERSITE":   "1",
    }

    # mini-swe templates are Jinja2; render extra vars via agent.run kwargs.
    instance_template = INSTANCE_TEMPLATE  # leaves {{task}} for agent.run
    system_template = _system_template(with_teams=with_teams, prompt_variant=prompt_variant)

    # Per-command timeout: most steps finish in seconds; some single
    # commands (full preprocessing, scvi training) can run minutes. Give
    # each command the full task wallclock budget — the step_limit cap
    # bounds total work via the number of LLM calls.
    per_cmd_timeout = int(task.get("wallclock_s", 600))
    step_limit = int(task.get("max_turns", 30))

    trajectory_path = out_dir / "minisweagent_trajectory.json"

    if model_factory == "gpt_oauth":
        # GPT-5.x via the user's ChatGPT subscription (no API key cost).
        from bench.adapters.gpt_oauth_model import GptOauthModel
        model = GptOauthModel(
            model_name=model_id,
            cost_tracking="ignore_errors",
            reasoning_effort="medium",
        )
    elif model_factory == "litellm_deepseek":
        # DeepSeek API (OpenAI-compatible). API key from env DEEPSEEK_API_KEY.
        # No client-side ``max_tokens`` cap — the server applies the model's
        # own output budget. v4-pro is a reasoning model whose
        # reasoning_tokens share the completion budget; capping at 16k
        # would truncate deep plans mid-thought.
        from minisweagent.models.litellm_model import LitellmModel
        model = LitellmModel(
            model_name=f"openai/{model_id}",
            model_kwargs={
                "api_base": os.environ.get("DEEPSEEK_API_BASE",
                                             "https://api.deepseek.com/v1"),
                "api_key":  os.environ.get("DEEPSEEK_API_KEY", ""),
                "custom_llm_provider": "openai",
                "temperature": 0.0 + 0.1 * seed,
                "seed": seed,
            },
            cost_tracking="ignore_errors",
        )
    elif model_factory == "litellm_openrouter":
        # OpenRouter (OpenAI-compatible). API key from OPENROUTER_API_KEY.
        # Used to free local Ollama from FM-load contention when re-running
        # qwen on the same task in parallel with an active ollama-backed sweep.
        # No client-side ``max_tokens`` cap — let the server use the model's
        # own output budget.
        from minisweagent.models.litellm_model import LitellmModel
        model = LitellmModel(
            model_name=f"openai/{model_id}",
            model_kwargs={
                "api_base": os.environ.get("OPENROUTER_API_BASE",
                                             "https://openrouter.ai/api/v1"),
                "api_key":  os.environ.get("OPENROUTER_API_KEY", ""),
                "custom_llm_provider": "openai",
                "temperature": 0.0 + 0.1 * seed,
                "seed": seed,
            },
            cost_tracking="ignore_errors",
        )
    elif model_factory == "litellm_gemini":
        # Gemini via Google's OpenAI-compatible endpoint. Note: Google's
        # OpenAI-compat layer rejects ``seed`` (returns
        # ``Invalid JSON payload: Unknown name "seed"``), so we vary
        # behavior across seeds via ``temperature`` only. No client-side
        # ``max_tokens`` cap — let the server use the model's own output
        # budget.
        from minisweagent.models.litellm_model import LitellmModel
        model = LitellmModel(
            model_name=f"openai/{model_id}",
            model_kwargs={
                "api_base": os.environ.get("GEMINI_API_BASE",
                                             "https://generativelanguage.googleapis.com/v1beta/openai/"),
                "api_key":  os.environ.get("GEMINI_API_KEY", ""),
                "custom_llm_provider": "openai",
                "temperature": 0.0 + 0.1 * seed,
            },
            cost_tracking="ignore_errors",
        )
    elif model_factory == "litellm_glm":
        # Zhipu GLM coding-plan endpoint (OpenAI-compatible). 1M-context
        # reasoning model — emits reasoning_tokens that share the
        # completion budget with the actual answer. We don't impose a
        # client-side cap; the server applies the model's own max
        # output budget (well into the hundreds of thousands of tokens).
        # Endpoint accepts ``seed`` and ``temperature``.
        from minisweagent.models.litellm_model import LitellmModel
        model = LitellmModel(
            model_name=f"openai/{model_id}",
            model_kwargs={
                "api_base": os.environ.get("GLM_API_BASE",
                                             "https://open.bigmodel.cn/api/coding/paas/v4"),
                "api_key":  os.environ.get("GLM_API_KEY", ""),
                "custom_llm_provider": "openai",
                "temperature": 0.0 + 0.1 * seed,
                "seed": seed,
            },
            cost_tracking="ignore_errors",
        )
    elif model_factory == "litellm_minimax":
        # MiniMax M-series (OpenAI-compatible). Reasoning model — emits
        # ``<think>`` blocks inline in the message content, but tool_calls
        # are properly structured. Accepts ``seed`` and ``temperature``.
        # No client-side ``max_tokens`` cap (large output budget).
        from minisweagent.models.litellm_model import LitellmModel
        model = LitellmModel(
            model_name=f"openai/{model_id}",
            model_kwargs={
                "api_base": os.environ.get("MINIMAX_API_BASE",
                                             "https://api.minimaxi.com/v1"),
                "api_key":  os.environ.get("MINIMAX_API_KEY", ""),
                "custom_llm_provider": "openai",
                "temperature": 0.0 + 0.1 * seed,
                "seed": seed,
            },
            cost_tracking="ignore_errors",
        )
    else:
        from minisweagent.models.litellm_model import LitellmModel
        model = LitellmModel(
            model_name=f"openai/{model_id}",
            model_kwargs={
                "api_base": os.environ.get("OLLAMA_API_BASE",
                                            "http://127.0.0.1:11434/v1"),
                "api_key":  os.environ.get("OLLAMA_API_KEY",
                                            "sk-ollama-fake-12345"),
                "custom_llm_provider": "openai",
                "temperature": 0.0 + 0.1 * seed,
                "seed": seed,
                # Output cap: 128K. Qwen3.6 256k-modelfile has 256K total
                # context; reserve ~half for output so long python heredocs
                # + reasoning aren't truncated mid-response (smaller caps
                # caused qwen to emit tool_calls without reasoning_content
                # at long trajectories).
                "max_tokens": 131072,
                # 256K context for the qwen3.6 ``-256k`` modelfile.
                "extra_body": {"options": {"num_ctx": 262144}},
            },
            cost_tracking="ignore_errors",
        )
    env = PersistentKernelEnvironment(
        cwd=str(workspace),
        env=locked_env,
        timeout=per_cmd_timeout,
    )
    agent = DefaultAgent(
        model, env,
        system_template=system_template,
        instance_template=instance_template,
        step_limit=step_limit,
        cost_limit=0.0,        # disable cost cap — we use step_limit only
        output_path=trajectory_path,
    )

    final_text: str | None = None
    err: str | None = None
    n_turns = 0
    t0 = time.time()
    try:
        # Spin up the persistent kernel before the first agent turn so the
        # cost of importing scanpy / omicverse is paid here (~5s) rather
        # than appearing as a turn-1 cliff in trajectory timings.
        env._ensure_kernel()
        result = agent.run(
            task=task["prompt"],
            cwd_path=str(workspace),
            io_block=io_block,
            step_limit=step_limit,
            timeout_s=per_cmd_timeout,
        )
        # result is the `extra` dict from the exit message: {exit_status, submission, ...}
        final_text = (result or {}).get("submission") or (result or {}).get("content")
    except (Submitted, LimitsExceeded) as exc:
        # These are expected end-states routed through exception handlers.
        final_text = getattr(exc, "args", [""])[0] if exc.args else None
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        traceback.print_exc(file=log_path.open("a"))
    finally:
        n_turns = getattr(agent, "n_calls", 0)
        try:
            agent.save(trajectory_path)
        except Exception:
            pass
        try:
            env.shutdown()
        except Exception:
            pass
        for name in ("agent", "litellm_model"):
            logging.getLogger(name).removeHandler(log_handler)
        log_handler.close()

    final_adata = workspace / "adata.h5ad"
    if final_adata.exists():
        final_path = out_dir / "final.h5ad"
        if final_path.exists():
            final_path.unlink()
        shutil.copy2(final_adata, final_path)
        final_adata_path = str(final_path)
    else:
        final_adata_path = None

    return {
        "final_adata_path": final_adata_path,
        "final_text":       final_text,
        "n_turns":          int(n_turns),
        "wallclock_s":      time.time() - t0,
        "error":            err,
        "trajectory_path":  str(trajectory_path) if trajectory_path.exists() else None,
    }


# Named entry-points; runner/SYSTEMS dispatches by name. Two prompt arms ×
# two model backends (qwen3.6 via local ollama / gpt-5.5 via ChatGPT OAuth)
# = four mini-swe-agent variants, all sharing one persistent-kernel loop.

def run_omicverse(task: dict, model_id: str, seed: int) -> dict:
    """qwen3.6 + omicverse domain prompt."""
    return _do_run(task, model_id, seed, with_teams=True,
                    model_factory="litellm_ollama")


def run_baseline(task: dict, model_id: str, seed: int) -> dict:
    """qwen3.6 + bare system prompt."""
    return _do_run(task, model_id, seed, with_teams=False,
                    model_factory="litellm_ollama")


def run_deepseek_omicverse(task: dict, model_id: str, seed: int) -> dict:
    """DeepSeek (any deepseek-* model) + omicverse domain prompt."""
    return _do_run(task, model_id, seed, with_teams=True,
                    model_factory="litellm_deepseek")


def run_deepseek_baseline(task: dict, model_id: str, seed: int) -> dict:
    """DeepSeek (any deepseek-* model) + bare system prompt."""
    return _do_run(task, model_id, seed, with_teams=False,
                    model_factory="litellm_deepseek")


def run_openrouter_omicverse(task: dict, model_id: str, seed: int) -> dict:
    """OpenRouter-hosted model (e.g. ``qwen/qwen3.6-35b-a3b``) + omicverse prompt.

    Use when the same model needs to be benchmarked from a different
    backend so the local Ollama instance can host a *different* sweep
    without contention.
    """
    return _do_run(task, model_id, seed, with_teams=True,
                    model_factory="litellm_openrouter")


def run_openrouter_baseline(task: dict, model_id: str, seed: int) -> dict:
    """OpenRouter-hosted model + bare system prompt."""
    return _do_run(task, model_id, seed, with_teams=False,
                    model_factory="litellm_openrouter")


def run_gemini_omicverse(task: dict, model_id: str, seed: int) -> dict:
    """Gemini (via OpenAI-compatible endpoint) + omicverse domain prompt."""
    return _do_run(task, model_id, seed, with_teams=True,
                    model_factory="litellm_gemini")


def run_gemini_baseline(task: dict, model_id: str, seed: int) -> dict:
    """Gemini + bare system prompt."""
    return _do_run(task, model_id, seed, with_teams=False,
                    model_factory="litellm_gemini")


def run_gpt_omicverse(task: dict, model_id: str, seed: int) -> dict:
    """gpt-5.5 (via ChatGPT OAuth) + omicverse domain prompt."""
    return _do_run(task, model_id, seed, with_teams=True,
                    model_factory="gpt_oauth")


def run_gpt_baseline(task: dict, model_id: str, seed: int) -> dict:
    """gpt-5.5 (via ChatGPT OAuth) + bare system prompt."""
    return _do_run(task, model_id, seed, with_teams=False,
                    model_factory="gpt_oauth")


def run_glm_omicverse(task: dict, model_id: str, seed: int) -> dict:
    """Zhipu GLM (coding-plan endpoint, OpenAI-compatible) + omicverse domain prompt."""
    return _do_run(task, model_id, seed, with_teams=True,
                    model_factory="litellm_glm")


def run_glm_baseline(task: dict, model_id: str, seed: int) -> dict:
    """Zhipu GLM + bare system prompt."""
    return _do_run(task, model_id, seed, with_teams=False,
                    model_factory="litellm_glm")


def run_minimax_omicverse(task: dict, model_id: str, seed: int) -> dict:
    """MiniMax M-series (OpenAI-compatible) + omicverse domain prompt."""
    return _do_run(task, model_id, seed, with_teams=True,
                    model_factory="litellm_minimax")


def run_minimax_baseline(task: dict, model_id: str, seed: int) -> dict:
    """MiniMax M-series + bare system prompt."""
    return _do_run(task, model_id, seed, with_teams=False,
                    model_factory="litellm_minimax")


# -----------------------------------------------------------------------
# Ablation arms — same model + omicverse prompt, but with the
# registry_lookup / skill_lookup discovery section removed. Comparing
# these against ``gpt_omicverse`` / ``deepseek_omicverse`` (full prompt)
# attributes the omicverse skill effect to the registry channel vs. the
# rest of the prompt (reasoning + dispatch + coding conventions).
# -----------------------------------------------------------------------

def run_gpt_omicverse_no_registry(task: dict, model_id: str, seed: int) -> dict:
    """gpt-5.5 + omicverse prompt MINUS the registry/skill-lookup section."""
    return _do_run(task, model_id, seed, with_teams=True,
                    model_factory="gpt_oauth",
                    prompt_variant="no_registry",
                    system_name="gpt_omicverse_no_registry")


def run_deepseek_omicverse_no_registry(task: dict, model_id: str, seed: int) -> dict:
    """deepseek + omicverse prompt MINUS the registry/skill-lookup section."""
    return _do_run(task, model_id, seed, with_teams=True,
                    model_factory="litellm_deepseek",
                    prompt_variant="no_registry",
                    system_name="deepseek_omicverse_no_registry")


def run_gpt_omicverse_doc_rag(task: dict, model_id: str, seed: int) -> dict:
    """gpt-5.5 + omicverse prompt with doc_lookup (vanilla embedding-RAG over
    raw docstrings) replacing Beacon's registry/skill_lookup. Tests whether
    Beacon's structured library-side contract beats vanilla doc retrieval."""
    return _do_run(task, model_id, seed, with_teams=True,
                    model_factory="gpt_oauth",
                    prompt_variant="doc_rag",
                    system_name="gpt_omicverse_doc_rag")


def run_deepseek_omicverse_doc_rag(task: dict, model_id: str, seed: int) -> dict:
    """deepseek + omicverse prompt with doc_lookup replacing Beacon."""
    return _do_run(task, model_id, seed, with_teams=True,
                    model_factory="litellm_deepseek",
                    prompt_variant="doc_rag",
                    system_name="deepseek_omicverse_doc_rag")
