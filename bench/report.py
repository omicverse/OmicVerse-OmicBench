"""Generate results/report.md from results/runs.parquet.

Sections:
- Pass@1 ± 95% bootstrap CI per (model, system)
- Per-layer Pass@1
- Pairwise paired McNemar with BH-FDR(q=0.10)
- Per-task per-system marks
- 7-class failure-mode histogram
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bench.stats import benjamini_hochberg, bootstrap_ci, mcnemar_paired

REPO_ROOT = Path(__file__).resolve().parent.parent
PARQUET = REPO_ROOT / "results" / "runs.parquet"
OUT = REPO_ROOT / "results" / "report.md"

SYSTEMS_DEFAULT = ["ov_agent", "raw_llm", "biomni", "human_scanpy"]


def main() -> None:
    df = pd.read_parquet(PARQUET)
    df = df.sort_values("started_at").drop_duplicates(
        ["model_id", "system", "task_id", "seed"], keep="last"
    )

    models = sorted(df["model_id"].unique())
    seeds = sorted(df["seed"].unique())
    tasks_present = sorted(df["task_id"].unique())
    systems = [s for s in SYSTEMS_DEFAULT if s in df["system"].unique()]

    lines: list[str] = []
    lines.append("# bench report")
    lines.append("")
    lines.append(f"Models: {', '.join(f'`{m}`' for m in models)}  ·  "
                 f"Seeds: {seeds}  ·  Tasks: {len(tasks_present)}  ·  "
                 f"Systems: {', '.join(systems)}")
    lines.append("")
    for img in ("heatmap.png", "per_layer_bar.png", "scaling_curve.png"):
        if (REPO_ROOT / "results" / img).exists():
            lines.append(f"![{img}]({img})")
            lines.append("")

    # Pass@1 with bootstrap CI -----------------------------------------
    lines.append("## Pass@1 with 95% bootstrap CI")
    lines.append("")
    lines.append("Per (model, system), aggregated across all task-seed pairs. "
                 "10 000-resample percentile bootstrap. n = task-seed pairs.")
    lines.append("")
    for m in models:
        lines.append(f"### Model: `{m}`")
        lines.append("")
        lines.append("| System | Pass@1 | 95% CI | n | Mean wallclock | Median turns |")
        lines.append("|---|---:|---|---:|---:|---:|")
        sub_m = df[df.model_id == m]
        for s in systems:
            sub = sub_m[sub_m.system == s]
            if s == "human_scanpy":
                sub = sub[sub.failure_mode != "no_baseline"]
            if len(sub) == 0:
                continue
            mean, lo, hi = bootstrap_ci(sub["passed"].astype(bool).values.astype(float))
            wc = sub["wallclock_s"].mean()
            mt = sub["n_turns"].median()
            lines.append(f"| {s} | {mean:.2%} | [{lo:.2%}, {hi:.2%}] | {len(sub)} | "
                         f"{wc:.0f}s | {mt:.0f} |")
        lines.append("")

    # Per-layer ---------------------------------------------------------
    lines.append("## Per-layer Pass@1")
    lines.append("")
    for m in models:
        lines.append(f"### Model: `{m}`")
        lines.append("")
        layers = sorted(df["layer"].unique())
        header = "| Layer | n_tasks | " + " | ".join(systems) + " |"
        sep    = "|---|---:|" + "|".join(["---:"] * len(systems)) + "|"
        lines.append(header)
        lines.append(sep)
        sub_m = df[df.model_id == m]
        for L in layers:
            sub_L = sub_m[sub_m.layer == L]
            n_tasks = sub_L["task_id"].nunique()
            row = [L, str(n_tasks)]
            for s in systems:
                sub_LS = sub_L[sub_L.system == s]
                if s == "human_scanpy":
                    sub_LS = sub_LS[sub_LS.failure_mode != "no_baseline"]
                if len(sub_LS) == 0:
                    row.append("—")
                else:
                    p = sub_LS["passed"].astype(bool).mean()
                    row.append(f"{p:.0%} ({int(sub_LS['passed'].astype(bool).sum())}/{len(sub_LS)})")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # Pairwise McNemar with BH-FDR -------------------------------------
    lines.append("## Pairwise paired McNemar tests")
    lines.append("")
    lines.append("Per model. Each (task_id, seed) is one paired observation — "
                 "**no OR-collapse across seeds**. **b01** = system-A pass, B "
                 "fail. **b10** = A fail, B pass. **p** = exact two-sided "
                 "binomial. Significant after BH-FDR(q=0.10) shown in **bold**.")
    lines.append("")
    for m in models:
        lines.append(f"### Model: `{m}`")
        lines.append("")
        lines.append("| Comparison | b01 | b10 | n_disc | p (exact) | BH-sig |")
        lines.append("|---|---:|---:|---:|---:|:-:|")
        sub_m = df[df.model_id == m]
        pass_by_sys: dict[str, pd.Series] = {}
        for s in systems:
            sub = sub_m[sub_m.system == s]
            if s == "human_scanpy":
                sub = sub[sub.failure_mode != "no_baseline"]
            if len(sub) == 0:
                continue
            # Index by (task, seed) for paired-observation McNemar.
            # Each seed is an independent draw against system stochasticity;
            # OR-collapsing across seeds inflates the apparent pass rate
            # and violates McNemar's paired assumption (Codex F3).
            pass_by_sys[s] = (
                sub.set_index(["task_id", "seed"])["passed"].astype(bool)
            )
        comps = []
        pvals: list[float | None] = []
        for i, sa in enumerate(systems):
            for sb in systems[i+1:]:
                if sa not in pass_by_sys or sb not in pass_by_sys:
                    continue
                b01, b10, p = mcnemar_paired(pass_by_sys[sa], pass_by_sys[sb])
                n_disc = b01 + b10
                comps.append((sa, sb, b01, b10, n_disc, p))
                pvals.append(p)
        rejects = benjamini_hochberg(pvals, q=0.10)
        for (sa, sb, b01, b10, n_disc, p), rej in zip(comps, rejects):
            p_str = "—" if p is None else f"{p:.3f}"
            sig = "**yes**" if rej else "no"
            lines.append(f"| {sa} vs {sb} | {b01} | {b10} | {n_disc} | {p_str} | {sig} |")
        lines.append("")

    # Per-task marks ---------------------------------------------------
    lines.append("## Per-task per-system seed-pass-rate")
    lines.append("")
    lines.append("Cell value = `k/n` (k seeds passed of n run). One lucky seed "
                 "out of three is **not** a task win (Codex F3).")
    lines.append("")
    for m in models:
        lines.append(f"### Model: `{m}`")
        lines.append("")
        sub_m = df[df.model_id == m]
        header = "| Task | L | " + " | ".join(systems) + " |"
        sep    = "|---|---|" + "|".join([":-:"] * len(systems)) + "|"
        lines.append(header)
        lines.append(sep)
        for t in tasks_present:
            sub_t = sub_m[sub_m.task_id == t]
            if len(sub_t) == 0:
                continue
            row = [t, sub_t["layer"].iloc[0]]
            for s in systems:
                sub_ts = sub_t[sub_t.system == s]
                if len(sub_ts) == 0:
                    row.append("·")
                else:
                    if s == "human_scanpy" and (sub_ts["failure_mode"] == "no_baseline").any():
                        row.append("—")
                    else:
                        n = len(sub_ts)
                        k = int(sub_ts["passed"].astype(bool).sum())
                        if n == 1:
                            row.append("✓" if k == 1 else "✗")
                        else:
                            row.append(f"{k}/{n}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # Failure-mode breakdown -------------------------------------------
    lines.append("## Failure-mode breakdown (failed runs only, 7-class taxonomy)")
    lines.append("")
    fm = df[df.passed == False].groupby(["system", "failure_mode"]).size().unstack(fill_value=0)
    if len(fm) > 0:
        fm = fm.reindex(index=[s for s in systems if s in fm.index])
        lines.append(fm.to_markdown())
        lines.append("")

    OUT.write_text("\n".join(lines) + "\n")
    print(f"[done] {OUT}")


if __name__ == "__main__":
    main()
