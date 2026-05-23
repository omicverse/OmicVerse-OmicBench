"""Detailed per-check report — extracts numerical metrics from grader notes.

Each rubric check stores a free-form ``notes`` line such as
``"ari_vs_celltype: ARI=0.347"`` or
``"velocity_flows_from_root: outward cosine=0.273 (velocity flows from 'Ductal')"``.
This script parses those notes per-(task, check) and tabulates the leading
numeric metric per arm so you can see *quantitative* differences even when
both arms pass.

Useful for finding cases where gpt_omicverse passes a check with a
better metric than gpt_baseline (or vice versa) — pass/fail-only
tables hide these.

Usage::

    python scripts/per_check_detail.py             # full markdown table
    python scripts/per_check_detail.py --quant     # only checks with extractable numbers
    python scripts/per_check_detail.py --diff      # only rows where arms diverge
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "data" / "results"

SYSTEMS = [
    "gpt_baseline", "gpt_omicverse",
    "mini_swe_baseline", "mini_swe_omicverse",
    "human_scanpy",
]

# Patterns that surface a single representative numeric metric from a notes
# fragment. Tried in order; first match wins.
NUM_PATTERNS = [
    re.compile(r"outward cosine\s*=\s*(-?\d+\.\d+)", re.I),
    re.compile(r"mean cosine\s*=\s*(-?\d+\.\d+)", re.I),
    re.compile(r"\bARI\s*=\s*(-?\d+\.\d+)", re.I),
    re.compile(r"silhouette[^=]*=\s*(-?\d+\.\d+)", re.I),
    re.compile(r"Jaccard[^=]*=\s*(-?\d+\.\d+)", re.I),
    re.compile(r"correlation[^=]*=\s*(-?\d+\.\d+)", re.I),
    re.compile(r"agreement[^=]*=\s*(-?\d+\.\d+)", re.I),
    re.compile(r"improvement[^=]*=\s*(-?\d+\.\d+|\+-?\d+\.\d+)", re.I),
    re.compile(r"score\s*=\s*(-?\d+\.\d+)", re.I),
    re.compile(r"=\s*(-?\d+\.\d+)"),
]


def load_grades() -> dict[tuple[str, str], dict]:
    keyed = {}
    for f in sorted(glob.glob(str(RESULTS_ROOT / "*" / "grades.csv"))):
        for row in csv.DictReader(open(f)):
            keyed[(row["task_id"], row["system"])] = row
    return keyed


def split_notes(notes: str) -> dict[str, str]:
    """`'check_a: ... | check_b: ...'` → ``{check_a: ..., check_b: ...}``."""
    out: dict[str, str] = {}
    for chunk in notes.split(" | "):
        if ":" in chunk:
            cid, _, msg = chunk.partition(":")
            out[cid.strip()] = msg.strip()
    return out


def extract_metric(msg: str) -> str | None:
    """Return the leading numeric metric (with sign) from a notes fragment."""
    for pat in NUM_PATTERNS:
        m = pat.search(msg)
        if m:
            return m.group(1)
    return None


def cell_for(passed: bool | None, metric: str | None) -> str:
    if passed is None:
        return "—"
    icon = "✓" if passed else "✗"
    if metric is not None:
        return f"{icon} {metric}"
    return icon


def build(keyed):
    """{(task, check) → {system → (passed, metric_str_or_None)}}"""
    table: dict[tuple[str, str], dict[str, tuple[bool, str | None]]] = defaultdict(dict)
    for (task, system), row in keyed.items():
        try:
            rubric = json.loads(row["rubric_json"])
        except (json.JSONDecodeError, KeyError):
            continue
        notes_by_check = split_notes(row.get("notes", ""))
        for check_id, passed in rubric.items():
            metric = extract_metric(notes_by_check.get(check_id, ""))
            table[(task, check_id)][system] = (bool(passed), metric)
    return table


def render_markdown(table, *, only_quant=False, only_diff=False) -> str:
    out = []
    out.append("| Task | Check | gpt_base | gpt_ov | qwen_base | qwen_ov | human |")
    out.append("|---|---|:-:|:-:|:-:|:-:|:-:|")
    prev_task = ""
    n = 0
    for (task, check), arms in sorted(table.items()):
        cells_quant = [arms.get(s, (None, None))[1] for s in SYSTEMS]
        cells_pass = [arms.get(s, (None, None))[0] for s in SYSTEMS]
        if only_quant and not any(q for q in cells_quant):
            continue
        if only_diff:
            seen = set(cells_quant) - {None}
            seen_p = set(cells_pass) - {None}
            if len(seen) <= 1 and len(seen_p) <= 1:
                continue
        cells = [task if task != prev_task else "", check]
        for s in SYSTEMS:
            p, m = arms.get(s, (None, None))
            cells.append(cell_for(p, m))
        out.append("| " + " | ".join(cells) + " |")
        prev_task = task
        n += 1
    out.append("")
    out.append(f"_{n} rows_")
    return "\n".join(out)


def render_gpt_metric_compare(table) -> str:
    """Only checks where both gpt arms passed AND extracted a numeric
    metric — show which is biologically better."""
    out = ["## gpt_omicverse vs gpt_baseline (both passed; better metric wins)"]
    out.append("")
    out.append("| Task | Check | gpt_base | gpt_ov | winner | margin |")
    out.append("|---|---|:-:|:-:|:-:|:-:|")
    rows = 0
    for (task, check), arms in sorted(table.items()):
        cb = arms.get("gpt_baseline")
        co = arms.get("gpt_omicverse")
        if cb is None or co is None:
            continue
        if not cb[0] or not co[0]:
            continue  # at least one fails — already shown elsewhere
        if cb[1] is None or co[1] is None:
            continue
        try:
            cb_v, co_v = float(cb[1]), float(co[1])
        except ValueError:
            continue
        # Heuristic: "better" = bigger for everything except thresholds
        # like p-values. None of our extracted metrics are p-values, so
        # bigger is better.
        if cb_v == co_v:
            continue
        winner = "gpt_ov" if co_v > cb_v else "gpt_base"
        margin = abs(co_v - cb_v)
        out.append(f"| {task} | `{check}` | {cb_v:.3f} | {co_v:.3f} | "
                    f"**{winner}** | {margin:.3f} |")
        rows += 1
    if rows == 0:
        out.append("| _(no quant divergence in passing checks)_ | | | | | |")
    return "\n".join(out)


def render_4arm_compare(table) -> str:
    """4-arm metric comparison: for each (task, check) where ≥2 arms
    produced an extractable numeric metric, show all 4 model arms side-
    by-side, highlight the per-row winner, and tally aggregate margins.
    """
    out = ["## 4-arm metric comparison (gpt/qwen × baseline/omicverse)"]
    out.append("")
    out.append("Bigger = better for every metric extracted here "
                "(ARI, Jaccard, silhouette, cosine, agreement, …).")
    out.append("")
    out.append("| Task | Check | gpt_base | gpt_ov | qwen_base | qwen_ov | winner |")
    out.append("|---|---|:-:|:-:|:-:|:-:|:-:|")

    arms = ["gpt_baseline", "gpt_omicverse",
            "mini_swe_baseline", "mini_swe_omicverse"]
    short = {"gpt_baseline": "gpt_base",
             "gpt_omicverse": "gpt_ov",
             "mini_swe_baseline": "qwen_base",
             "mini_swe_omicverse": "qwen_ov"}

    # Aggregate stats:
    #   omicverse_vs_baseline_within_model: per (model, check) which side wins
    n_per_arm_wins = {a: 0 for a in arms}
    margin_per_arm = {a: 0.0 for a in arms}
    rows = 0

    for (task, check), arm_data in sorted(table.items()):
        # Collect numeric metrics per arm. Only consider PASSING arms
        # so we don't compare a 0.0 (failing) vs a 0.5 (passing) cell.
        vals: dict[str, float] = {}
        for a in arms:
            entry = arm_data.get(a)
            if entry is None:
                continue
            passed, metric = entry
            if not passed or metric is None:
                continue
            try:
                vals[a] = float(metric)
            except ValueError:
                continue
        if len(vals) < 2:
            continue

        winner = max(vals, key=vals.get)
        # Skip rows where every arm tied
        if len(set(round(v, 4) for v in vals.values())) <= 1:
            continue

        cells = [task, f"`{check}`"]
        for a in arms:
            entry = arm_data.get(a)
            if entry is None:
                cells.append("—")
                continue
            passed, metric = entry
            if not passed:
                cells.append("✗")
                continue
            if metric is None:
                cells.append("✓")
                continue
            try:
                v = float(metric)
            except ValueError:
                cells.append(str(metric))
                continue
            tag = f"**{v:.3f}**" if a == winner else f"{v:.3f}"
            cells.append(tag)
        cells.append(short[winner])

        out.append("| " + " | ".join(cells) + " |")
        n_per_arm_wins[winner] += 1
        # Margin against the worst arm in the row
        worst = min(vals.values())
        margin_per_arm[winner] += vals[winner] - worst
        rows += 1

    out.append("")
    out.append(f"_{rows} quant rows where arms diverge._")
    out.append("")
    out.append("### Per-arm aggregate")
    out.append("")
    out.append("| Arm | check wins | sum of (winner − row-worst) |")
    out.append("|---|---:|---:|")
    for a in sorted(arms, key=lambda x: -n_per_arm_wins[x]):
        out.append(f"| {short[a]} | {n_per_arm_wins[a]} | "
                    f"{margin_per_arm[a]:.3f} |")

    # Within-model omicverse-vs-baseline (the key bench question)
    out.append("")
    out.append("### Within-model: omicverse vs baseline")
    out.append("")
    out.append("| Model | omicverse-wins | baseline-wins | omicverse-margin sum | baseline-margin sum |")
    out.append("|---|:-:|:-:|---:|---:|")
    for model_label, base_arm, ov_arm in [
        ("gpt-5.5 (gpt-5.5)", "gpt_baseline", "gpt_omicverse"),
        ("qwen3.6 (mini)", "mini_swe_baseline", "mini_swe_omicverse"),
    ]:
        n_ov = n_base = 0
        sum_ov = sum_base = 0.0
        for (task, check), arm_data in table.items():
            base = arm_data.get(base_arm)
            ov = arm_data.get(ov_arm)
            if base is None or ov is None:
                continue
            if not base[0] or not ov[0]:
                continue
            if base[1] is None or ov[1] is None:
                continue
            try:
                bv, ov_v = float(base[1]), float(ov[1])
            except ValueError:
                continue
            if bv == ov_v:
                continue
            if ov_v > bv:
                n_ov += 1
                sum_ov += ov_v - bv
            else:
                n_base += 1
                sum_base += bv - ov_v
        out.append(f"| {model_label} | {n_ov} | {n_base} | {sum_ov:+.3f} | {sum_base:+.3f} |")

    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", action="store_true",
                    help="only checks with extractable numeric metrics")
    ap.add_argument("--diff", action="store_true",
                    help="only rows where arms diverge in pass/fail or metric")
    ap.add_argument("--gpt-compare", action="store_true",
                    help="emit gpt_omicverse-vs-gpt_baseline metric comparison")
    ap.add_argument("--four-arm", action="store_true",
                    help="emit 4-arm metric comparison + per-arm aggregate "
                         "+ within-model omicverse-vs-baseline")
    args = ap.parse_args()

    keyed = load_grades()
    table = build(keyed)

    if args.gpt_compare:
        print(render_gpt_metric_compare(table))
        return 0

    if args.four_arm:
        print(render_4arm_compare(table))
        return 0

    print(render_markdown(table, only_quant=args.quant, only_diff=args.diff))
    return 0


if __name__ == "__main__":
    sys.exit(main())
