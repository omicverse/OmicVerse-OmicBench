"""Per-check pass/fail report across all sweeps.

Aggregates every grades.csv under ``data/results/<run_name>/`` into a
fine-grained table where each row is a single rubric check (not a whole
task) and columns are the systems being compared. Highlights:

- which exact checks fail for which arm (vs. just "task X failed at 0.67")
- which checks are universally failing — usually a grader threshold
  bug rather than a model capability gap
- which checks discriminate between gpt_omicverse and gpt_baseline
  vs. mini_swe_omicverse and mini_swe_baseline

Usage::

    python scripts/per_check_report.py             # markdown stdout
    python scripts/per_check_report.py --csv       # CSV stdout
    python scripts/per_check_report.py --filter gpt_omicverse
                                                    # only show checks where
                                                    # gpt_omicverse fails
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "data" / "results"

SYSTEMS_ORDER = [
    "gpt_baseline",
    "gpt_omicverse",
    "mini_swe_baseline",
    "mini_swe_omicverse",
    "human_scanpy",
]


def load_all_grades() -> dict[tuple[str, str], dict]:
    """Walk every results/<run>/grades.csv, keep latest row per (task, system).

    Later runs overwrite earlier ones — sweeps with newer timestamps
    sort last alphabetically when run names are timestamp-prefixed.
    """
    keyed: dict[tuple[str, str], dict] = {}
    for f in sorted(glob.glob(str(RESULTS_ROOT / "*" / "grades.csv"))):
        for row in csv.DictReader(open(f)):
            keyed[(row["task_id"], row["system"])] = row
    return keyed


def build_check_table(keyed) -> dict[tuple[str, str], dict[str, bool]]:
    """{(task_id, check_id) → {system → passed_bool}}"""
    table: dict[tuple[str, str], dict[str, bool]] = defaultdict(dict)
    for (task, system), row in keyed.items():
        try:
            rubric = json.loads(row["rubric_json"])
        except (json.JSONDecodeError, KeyError):
            continue
        for check_id, passed in rubric.items():
            table[(task, check_id)][system] = bool(passed)
    return table


def render_markdown(table, filter_system: str | None = None) -> str:
    out = []
    header = ["Task", "Check"] + [s.replace("_", "&shy;_") for s in SYSTEMS_ORDER]
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "|".join(["---", "---"] + [":-:"] * len(SYSTEMS_ORDER)) + "|")
    prev_task = ""
    n_rows = 0
    for (task, check), arms in sorted(table.items()):
        if filter_system is not None:
            if arms.get(filter_system) is not False:
                continue
        cells = [task if task != prev_task else "", check]
        prev_task = task if not (filter_system and prev_task == task) else prev_task
        for s in SYSTEMS_ORDER:
            v = arms.get(s)
            if v is None:
                cells.append("—")
            else:
                cells.append("✓" if v else "✗")
        out.append("| " + " | ".join(cells) + " |")
        n_rows += 1
    if filter_system:
        out.append("")
        out.append(f"_{n_rows} checks where {filter_system} failed._")
    return "\n".join(out)


def render_csv(table) -> str:
    out = ["task,check," + ",".join(SYSTEMS_ORDER)]
    for (task, check), arms in sorted(table.items()):
        cells = [task, check] + [
            ("PASS" if arms[s] else "FAIL") if s in arms else "NA"
            for s in SYSTEMS_ORDER
        ]
        out.append(",".join(cells))
    return "\n".join(out)


def render_universal_failures(table) -> str:
    """Checks that fail for every arm that produced output for them.

    These are usually grader / fixture bugs rather than capability gaps.
    """
    out = ["## Universal failures (likely grader / fixture issues)"]
    out.append("")
    out.append("| Task | Check | n_arms | rationale-to-investigate |")
    out.append("|---|---|:-:|---|")
    for (task, check), arms in sorted(table.items()):
        # ignore human_scanpy (no_baseline skips inflate the universal-fail count)
        cmpt = {s: v for s, v in arms.items() if s != "human_scanpy"}
        if not cmpt or any(cmpt.values()):
            continue
        out.append(f"| {task} | `{check}` | {len(cmpt)} | every non-human arm fails |")
    return "\n".join(out)


def render_gpt_ov_gaps(table) -> str:
    """Checks where gpt_omicverse fails but at least one other arm passes —
    these are gpt_omicverse-specific capability gaps OR grader rules that
    interact unfavourably with the omicverse system prompt.
    """
    out = ["## Checks where gpt_omicverse fails but a peer passes"]
    out.append("")
    out.append("| Task | Check | who passes |")
    out.append("|---|---|---|")
    rows = 0
    for (task, check), arms in sorted(table.items()):
        if arms.get("gpt_omicverse") is not False:
            continue
        passers = [s for s, v in arms.items() if v and s != "gpt_omicverse"]
        if not passers:
            continue
        out.append(f"| {task} | `{check}` | {', '.join(passers)} |")
        rows += 1
    if rows == 0:
        out.append("| _(none)_ | | |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="store_true",
                    help="emit CSV instead of markdown")
    ap.add_argument("--filter",
                    help="only include checks where this system failed")
    ap.add_argument("--summary", action="store_true",
                    help="emit universal-failure + gpt_ov-gap summaries")
    args = ap.parse_args()

    keyed = load_all_grades()
    table = build_check_table(keyed)

    if args.csv:
        print(render_csv(table))
        return 0

    if args.summary:
        print(render_universal_failures(table))
        print()
        print(render_gpt_ov_gaps(table))
        return 0

    print(render_markdown(table, filter_system=args.filter))
    return 0


if __name__ == "__main__":
    sys.exit(main())
