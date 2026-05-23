"""Generate PNG figures from results/runs.parquet.

Outputs:
  results/heatmap.png        — task × system pass marks faceted by model
  results/per_layer_bar.png  — per-layer Pass@1 bars per system, faceted by model
  results/scaling_curve.png  — Pass@1 vs active params, line per system
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
PARQUET = REPO_ROOT / "results" / "runs.parquet"
OUT_DIR = REPO_ROOT / "results"
SYSTEMS_ORDER = ["ov_agent", "raw_llm", "biomni", "human_scanpy"]


def _model_size_b(model_id: str) -> float:
    m = model_id.lower()
    if "35b-a3b" in m or "a3b" in m: return 3.0   # MoE active count
    if "32b" in m: return 32.0
    if "14b" in m: return 14.0
    if "8b"  in m: return 8.0
    if "4b"  in m: return 4.0
    n = re.search(r"(\d+(?:\.\d+)?)b", m)
    return float(n.group(1)) if n else 0.0


def main() -> None:
    df = pd.read_parquet(PARQUET)
    df = df.sort_values("started_at").drop_duplicates(
        ["model_id", "system", "task_id", "seed"], keep="last"
    )
    df = df[~((df.system == "human_scanpy") & (df.failure_mode == "no_baseline"))]

    models = sorted(df["model_id"].unique(), key=_model_size_b)
    systems = [s for s in SYSTEMS_ORDER if s in df["system"].unique()]
    tasks = sorted(df["task_id"].unique())

    n_models, n_tasks, n_sys = len(models), len(tasks), len(systems)

    # heatmap ---------------------------------------------------------
    fig, axes = plt.subplots(1, max(n_models, 1),
                             figsize=(max(4 * n_models, 5), max(0.3 * n_tasks + 1, 4)),
                             sharey=True, squeeze=False)
    for ax, m in zip(axes[0], models):
        sub = df[df.model_id == m]
        mat = np.full((n_tasks, n_sys), np.nan)
        for i, t in enumerate(tasks):
            for j, s in enumerate(systems):
                cell = sub[(sub.task_id == t) & (sub.system == s)]
                if len(cell):
                    mat[i, j] = float(cell["passed"].astype(bool).any())
        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(n_sys))
        ax.set_xticklabels(systems, rotation=30, ha="right", fontsize=8)
        ax.set_title(m, fontsize=9)
    axes[0, 0].set_yticks(range(n_tasks))
    axes[0, 0].set_yticklabels(tasks, fontsize=7)
    fig.colorbar(im, ax=axes[0, -1], shrink=0.7, label="pass (any seed)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "heatmap.png", dpi=130)
    plt.close(fig)
    print("wrote heatmap.png")

    # per-layer bar ---------------------------------------------------
    layers = sorted(df["layer"].unique())
    fig, axes = plt.subplots(1, max(n_models, 1), figsize=(max(5 * n_models, 5), 4),
                             sharey=True, squeeze=False)
    for ax, m in zip(axes[0], models):
        sub = df[df.model_id == m]
        x = np.arange(len(layers))
        width = 0.8 / max(n_sys, 1)
        for j, s in enumerate(systems):
            sub_s = sub[sub.system == s]
            heights = []
            for L in layers:
                cell = sub_s[sub_s.layer == L]
                heights.append(float(cell["passed"].astype(bool).mean()) if len(cell) else 0.0)
            ax.bar(x + j * width - 0.4, heights, width, label=s)
        ax.set_xticks(x)
        ax.set_xticklabels(layers)
        ax.set_ylim(0, 1)
        ax.set_title(m, fontsize=9)
        ax.set_ylabel("Pass@1")
    axes[0, 0].legend(fontsize=7, loc="upper left", ncol=1)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "per_layer_bar.png", dpi=130)
    plt.close(fig)
    print("wrote per_layer_bar.png")

    # scaling curve ---------------------------------------------------
    if len(models) >= 2:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        x = [_model_size_b(m) for m in models]
        for s in systems:
            ys, errs_lo, errs_hi = [], [], []
            for m in models:
                sub = df[(df.model_id == m) & (df.system == s)]
                if len(sub) == 0:
                    ys.append(np.nan); errs_lo.append(np.nan); errs_hi.append(np.nan)
                    continue
                vals = sub["passed"].astype(bool).values.astype(float)
                mean = vals.mean()
                rng = np.random.default_rng(0)
                samples = vals[rng.integers(0, len(vals), (5000, len(vals)))].mean(axis=1)
                ys.append(mean)
                errs_lo.append(mean - np.percentile(samples, 2.5))
                errs_hi.append(np.percentile(samples, 97.5) - mean)
            errs = np.array([errs_lo, errs_hi])
            errs = np.nan_to_num(errs)
            ax.errorbar(x, ys, yerr=errs, marker="o", label=s, capsize=3)
        ax.set_xscale("log")
        ax.set_xlabel("active params (B)")
        ax.set_ylabel("Pass@1 (across all tasks × seeds)")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title("Pass@1 scaling — error bars = bootstrap 95% CI")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "scaling_curve.png", dpi=130)
        plt.close(fig)
        print("wrote scaling_curve.png")


if __name__ == "__main__":
    main()
