"""Run benchmark trajectories from a YAML config.

Usage:
    python -m bench.runner --config configs/smoke_a_layer.yaml

Each invocation cross-products ``scope.systems × scope.task_ids × scope.seeds``
(plus ``llm.models`` if multiple) and writes one trajectory JSON per run into
``trajectories/<run_name>/<task>__<system>__<model_short>__seed<n>.json``.

Trajectory JSON schema (BixBench-compatible top-level + end-state extensions)::

    {
      "problem_id":      task["id"],
      "run_name":        cfg["run_name"],
      "system":          system,
      "model":           model_id,
      "seed":            int,
      "problem":         task["prompt"],
      "agent_answer":    extracted-text answer (knowledge tasks),
      "agent_answer_raw":raw model output,
      "ideal_answer":    null (not used for end-state tasks),
      "question_format": "open",
      "eval_mode":       "adata_checks" | "knowledge_rubric",
      "num_actions":     int,
      "wallclock_s":     float,
      "error":           str | null,
      "nb":              dict (jupyter notebook if produced, else {}),
      "result_path":     "..."  # h5ad path for adata_checks (else null)
      "metadata":        { task_id, layer, difficulty, fixture, oracle,
                           checks, knowledge_check, max_turns,
                           wallclock_s_limit }
    }

Grading is a separate postprocessing step:

    python -m bench.grader --traj-dir trajectories/<run_name>
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

from bench.adapters import SYSTEMS
from bench.config import (
    load_config,
    resolve_models,
    resolve_tasks,
    run_paths,
    trajectory_path,
)


def _eval_mode_for(task: dict) -> str:
    """End-state tasks → ``adata_checks``; D-layer rubric tasks → ``knowledge_rubric``."""
    if "knowledge_check" in task:
        return "knowledge_rubric"
    return "adata_checks"


def _build_trajectory(
    *,
    task: dict,
    system: str,
    model_id: str,
    seed: int,
    cfg: dict,
    run_result: dict,
) -> dict:
    """Wrap an adapter's ``run()`` dict into the unified trajectory schema."""
    eval_mode = _eval_mode_for(task)
    return {
        "problem_id":       task["id"],
        "run_name":         cfg["run_name"],
        "system":           system,
        "model":            model_id,
        "seed":             seed,
        "problem":          task["prompt"],
        "agent_answer":     run_result.get("final_text") or "",
        "agent_answer_raw": run_result.get("final_text") or "",
        "ideal_answer":     None,
        "question_format":  "open",
        "eval_mode":        eval_mode,
        "num_actions":      run_result.get("n_turns", 0),
        "wallclock_s":      run_result.get("wallclock_s", 0.0),
        "error":            run_result.get("error"),
        "nb":               run_result.get("nb") or {},
        "result_path":      run_result.get("final_adata_path"),
        "started_at":       datetime.utcnow().isoformat() + "Z",
        "metadata": {
            "task_id":           task["id"],
            "layer":             task["layer"],
            "difficulty":        task["difficulty"],
            "fixture":           task.get("fixture"),
            "oracle":            task.get("oracle"),
            "checks":            task.get("checks", []),
            "knowledge_check":   task.get("knowledge_check"),
            "max_turns":         task["max_turns"],
            "wallclock_s_limit": task["wallclock_s"],
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="path to a run-config YAML in configs/")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rp = run_paths(cfg)

    tasks = resolve_tasks(cfg)
    models = resolve_models(cfg)
    systems = list(cfg["scope"]["systems"])
    seeds = list(cfg["scope"]["seeds"])
    skip_completed = bool(cfg["scope"]["skip_completed"])

    print(f"[runner] config: {cfg['_config_path']}", flush=True)
    print(f"[runner] run_name: {cfg['run_name']}", flush=True)
    print(f"[runner] {len(tasks)} task(s) × {len(systems)} system(s) × "
          f"{len(models)} model(s) × {len(seeds)} seed(s) = "
          f"{len(tasks) * len(systems) * len(models) * len(seeds)} trajectories",
          flush=True)
    print(f"[runner] tasks: {[t['id'] for t in tasks]}", flush=True)
    print(f"[runner] systems: {systems}", flush=True)

    for model_id in models:
        for seed in seeds:
            for task in tasks:
                for system in systems:
                    out = trajectory_path(rp, task["id"], system, model_id, seed)
                    if skip_completed and out.exists():
                        print(f"[skip] {out.name}", flush=True)
                        continue

                    print(f"[run]  {system:14s} {task['id']:35s} "
                          f"{model_id:18s} seed{seed}", flush=True)
                    invoker = SYSTEMS[system]

                    # Per-task wallclock guard. Each task spec carries
                    # ``wallclock_s`` (else default 600); we install a
                    # SIGALRM that interrupts the invoker if it overruns.
                    # Without this, a stuck agent (e.g. qwen3.6 looping
                    # on the same broken script) burns ollama indefinitely
                    # and blocks the rest of the sweep.
                    import signal
                    budget = int(task.get("wallclock_s") or 600)

                    class _TaskTimeout(Exception):
                        pass

                    def _alarm(signum, frame):
                        raise _TaskTimeout(
                            f"wallclock_timeout: exceeded {budget}s budget"
                        )

                    prev_handler = signal.signal(signal.SIGALRM, _alarm)
                    signal.alarm(budget)
                    try:
                        run_result = invoker(task, model_id, seed)
                    except _TaskTimeout as exc:
                        print(f"  [timeout] killed after {budget}s",
                              flush=True)
                        run_result = {
                            "final_adata_path": None,
                            "final_text": None,
                            "n_turns": -1,
                            "wallclock_s": float(budget),
                            "error": f"wallclock_timeout: {exc}",
                        }
                    except Exception as exc:
                        traceback.print_exc()
                        run_result = {
                            "final_adata_path": None,
                            "final_text": None,
                            "n_turns": 0,
                            "wallclock_s": 0.0,
                            "error": f"runner_exception: {type(exc).__name__}: {exc}",
                        }
                    finally:
                        signal.alarm(0)
                        signal.signal(signal.SIGALRM, prev_handler)

                    print(f"  → wallclock={run_result['wallclock_s']:.1f}s "
                          f"turns={run_result['n_turns']} "
                          f"err={(run_result['error'] or '')[:120]}", flush=True)

                    traj = _build_trajectory(
                        task=task, system=system, model_id=model_id,
                        seed=seed, cfg=cfg, run_result=run_result,
                    )
                    out.write_text(json.dumps(traj, default=str, indent=2))
                    print(f"  → {out}", flush=True)

    print(f"[runner] done. trajectories under {rp['trajectories']}", flush=True)
    print(f"[runner] grade with: python -m bench.grader "
          f"--traj-dir {rp['trajectories']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
