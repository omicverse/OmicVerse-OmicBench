"""Grade a directory of trajectories produced by ``bench.runner``.

Usage:
    python -m bench.grade_run --traj-dir trajectories/<run_name>

Outputs:
    results/<run_name>/grades.csv     one row per trajectory, with rubric_json.
    results/<run_name>/summary.md     headline Pass@1 + per-layer + failure histogram.

This is the postprocessing step for the new run-config / unified-trajectory
flow. It dispatches each trajectory to the right grader based on
``eval_mode`` (``adata_checks`` -> ``bench.grader.grade``,
``knowledge_rubric`` -> the rubric grader copied from the legacy runner).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from bench._paths import REPO_ROOT
from bench.failure_taxonomy import classify
from bench.grader import grade as grade_adata
from bench.tasks import ALL_TASKS
from bench.types import FailureMode, Grade

# Live task specs by task_id — prefer these over the trajectory-frozen
# metadata so grader-rubric fixes propagate to existing trajectories on
# re-grade. (Without this, a max-threshold bump in tasks.py wouldn't take
# effect until the trajectory was re-run, which is wasteful.)
LIVE_TASKS: dict[str, dict] = {t["id"]: t for t in ALL_TASKS}


# ---------------------------------------------------------------------------
# Knowledge-task rubric grader (moved from the legacy runner.py)
# ---------------------------------------------------------------------------

def grade_knowledge(knowledge_check: dict, response: str) -> dict:
    """Rubric grader for D-layer knowledge tasks.

    Each criterion is a dict with ``required_any`` (list of canonical strings)
    and an optional ``description``. The criterion passes iff the response
    contains AT LEAST ONE of the canonical strings as a substring (case-
    insensitive).
    """
    if not response:
        return {"passed": False, "score": 0.0, "rubric": {},
                "notes": "no text response"}

    crits = knowledge_check.get("judge_criteria", [])
    threshold = knowledge_check.get("pass_threshold", len(crits))
    rubric: dict[str, bool] = {}
    notes: list[str] = []
    text_lower = response.lower()

    for i, c in enumerate(crits):
        cid = f"C{i+1}"
        if isinstance(c, dict):
            tokens = [t.lower() for t in c.get("required_any", [])]
            hits = [t for t in tokens if t in text_lower]
            ok = bool(hits)
            rubric[cid] = ok
            label = c.get("description", "")
            notes.append(
                f"{cid}{(' ' + label) if label else ''}: "
                f"{'matched ' + repr(hits[0]) if ok else 'no match in ' + repr(tokens)}"
            )
        else:
            tokens = [
                w for w in c.lower().replace("(", " ").replace(")", " ")
                .replace(",", " ").split() if len(w) > 2
            ]
            n_hits = sum(1 for t in tokens if t in text_lower)
            rubric[cid] = bool(n_hits >= 2)
            notes.append(f"{cid} (legacy-rubric): {n_hits}/{len(tokens)} tokens")

    passed_count = sum(1 for v in rubric.values() if v)
    return {
        "passed": passed_count >= threshold,
        "score": passed_count / max(len(crits), 1),
        "rubric": rubric,
        "notes": " | ".join(notes)[:1500],
    }


# ---------------------------------------------------------------------------
# Single-trajectory dispatcher
# ---------------------------------------------------------------------------

def grade_trajectory(traj: dict[str, Any]) -> dict[str, Any]:
    """Grade one trajectory and return a flat row for the CSV."""
    meta = traj.get("metadata", {})
    task_id = meta.get("task_id") or traj.get("problem_id", "")
    system = traj.get("system", "")
    model_id = traj.get("model", "")
    seed = int(traj.get("seed", 0))

    eval_mode = traj.get("eval_mode", "adata_checks")

    # Live spec wins when the task still exists in bench.tasks; falls back
    # to trajectory-frozen metadata for tasks that have since been removed.
    live = LIVE_TASKS.get(task_id, {})

    if eval_mode == "knowledge_rubric":
        if system == "human_scanpy":
            grade = {"passed": False, "score": 0.0,
                     "failure_mode": FailureMode.NO_BASELINE.value,
                     "rubric": {},
                     "notes": "human_scanpy does not score knowledge tasks"}
        else:
            kc = live.get("knowledge_check") or meta.get("knowledge_check") or {}
            g = grade_knowledge(kc, traj.get("agent_answer") or "")
            grade = {"passed": g["passed"], "score": g["score"],
                     "failure_mode": (FailureMode.NONE.value if g["passed"]
                                       else FailureMode.JUDGE_REJECTED.value),
                     "rubric": g["rubric"],
                     "notes": g["notes"]}
    else:
        gr: Grade = grade_adata(
            final_adata_path=traj.get("result_path"),
            checks=live.get("checks") or meta.get("checks") or [],
            oracle_path=live.get("oracle") or meta.get("oracle"),
            task_id=task_id, system=system,
            model_id=model_id, seed=seed,
        )
        grade = {"passed": gr.passed, "score": gr.score,
                 "failure_mode": gr.failure_mode.value,
                 "rubric": gr.rubric, "notes": gr.notes}

    if not grade["passed"]:
        # Reuse the 7-class failure tagger; it expects an adapter-style dict.
        adapter_dict = {
            "final_adata_path": traj.get("result_path"),
            "final_text":       traj.get("agent_answer"),
            "n_turns":          traj.get("num_actions", 0),
            "wallclock_s":      traj.get("wallclock_s", 0.0),
            "error":            traj.get("error"),
        }
        task_dict = {
            "id":         task_id,
            "layer":      meta.get("layer", ""),
            "max_turns":  meta.get("max_turns", 0),
            "wallclock_s": meta.get("wallclock_s_limit", 0),
        }
        grade["failure_mode"] = classify(
            run_result=adapter_dict, grade_passed=False,
            task=task_dict, system=system,
        )

    return {
        "task_id":     task_id,
        "layer":       meta.get("layer", ""),
        "difficulty":  meta.get("difficulty", ""),
        "system":      system,
        "model_id":    model_id,
        "seed":        seed,
        "passed":      bool(grade["passed"]),
        "score":       float(grade["score"]),
        "failure_mode": grade["failure_mode"],
        "rubric_json": json.dumps(grade["rubric"]),
        "notes":       grade["notes"],
        "n_turns":     int(traj.get("num_actions", 0)),
        "wallclock_s": float(traj.get("wallclock_s", 0.0)),
        "adapter_error": traj.get("error"),
        "started_at":  traj.get("started_at"),
        "trajectory":  traj.get("_trajectory_path"),
    }


# ---------------------------------------------------------------------------
# Run-level driver
# ---------------------------------------------------------------------------

def _write_summary(df: pd.DataFrame, out_path: Path, run_name: str) -> None:
    lines: list[str] = []
    lines.append(f"# {run_name} — run summary")
    lines.append("")
    lines.append(f"Models: {', '.join(f'`{m}`' for m in sorted(df.model_id.unique()))} · "
                 f"Seeds: {sorted(df.seed.unique())} · "
                 f"Tasks: {df.task_id.nunique()} · "
                 f"Trajectories: {len(df)}")
    lines.append("")

    lines.append("## Pass@1 by system")
    lines.append("")
    lines.append("| system | passed | total | Pass@1 | mean wallclock |")
    lines.append("|---|---:|---:|---:|---:|")
    for s, sub in df.groupby("system"):
        sub_eff = sub if s != "human_scanpy" else sub[sub.failure_mode != "no_baseline"]
        if len(sub_eff) == 0:
            continue
        n_pass = int(sub_eff.passed.sum())
        n_tot = len(sub_eff)
        wc = sub_eff.wallclock_s.mean()
        lines.append(f"| {s} | {n_pass} | {n_tot} | {n_pass/n_tot:.2%} | {wc:.0f}s |")
    lines.append("")

    lines.append("## Pass@1 by layer × system")
    lines.append("")
    systems = sorted(df.system.unique())
    layers = sorted(df.layer.unique())
    header = "| layer | " + " | ".join(systems) + " |"
    sep    = "|---|" + "|".join(["---:"] * len(systems)) + "|"
    lines.append(header)
    lines.append(sep)
    for L in layers:
        row = [L]
        for s in systems:
            sub = df[(df.layer == L) & (df.system == s)]
            if s == "human_scanpy":
                sub = sub[sub.failure_mode != "no_baseline"]
            if len(sub) == 0:
                row.append("—")
            else:
                k = int(sub.passed.sum())
                row.append(f"{k}/{len(sub)} ({k/len(sub):.0%})")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Failure-mode breakdown (failed trajectories only)")
    lines.append("")
    failed = df[~df.passed]
    if len(failed):
        fm = failed.groupby(["system", "failure_mode"]).size().unstack(fill_value=0)
        lines.append(fm.to_markdown())
    else:
        lines.append("_(no failures)_")
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n")


def grade_run(traj_dir: Path, results_dir: Path | None = None,
              run_name: str | None = None) -> int:
    traj_dir = Path(traj_dir).resolve()
    if run_name is None:
        run_name = traj_dir.name
    if results_dir is None:
        results_dir = REPO_ROOT / "results" / run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(traj_dir.glob("*.json"))
    if not files:
        print(f"[grade_run] no trajectories under {traj_dir}", file=sys.stderr)
        return 1

    rows: list[dict] = []
    for fp in files:
        try:
            traj = json.loads(fp.read_text())
        except Exception as exc:
            print(f"[grade_run] WARN parse {fp.name}: {exc}", file=sys.stderr)
            continue
        traj["_trajectory_path"] = str(fp.relative_to(REPO_ROOT)) \
            if fp.is_absolute() and str(fp).startswith(str(REPO_ROOT)) else str(fp)
        try:
            row = grade_trajectory(traj)
        except Exception as exc:
            print(f"[grade_run] WARN grade {fp.name}: {exc}", file=sys.stderr)
            continue
        rows.append(row)
        print(f"  {row['system']:14s} {row['task_id']:35s} seed{row['seed']} "
              f"→ passed={row['passed']} score={row['score']:.2f} "
              f"mode={row['failure_mode']}", flush=True)

    df = pd.DataFrame(rows)
    csv_path = results_dir / "grades.csv"
    df.to_csv(csv_path, index=False)
    print(f"[grade_run] wrote {csv_path} ({len(df)} rows)", flush=True)

    summary_path = results_dir / "summary.md"
    _write_summary(df, summary_path, run_name)
    print(f"[grade_run] wrote {summary_path}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", required=True,
                    help="directory holding trajectory JSONs (one run)")
    ap.add_argument("--results-dir", default=None,
                    help="output dir; default: results/<traj_dir.name>/")
    ap.add_argument("--run-name", default=None,
                    help="display name; default: traj_dir basename")
    args = ap.parse_args()
    return grade_run(
        Path(args.traj_dir),
        Path(args.results_dir) if args.results_dir else None,
        args.run_name,
    )


if __name__ == "__main__":
    sys.exit(main())
