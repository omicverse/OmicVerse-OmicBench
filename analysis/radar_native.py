#!/usr/bin/env python3
"""Native-rubric 6-dimension radar for ovagent_bench.

Unlike scripts/radar_grade.py (which asks ds4-pro to score trajectories on
the BiomniBench-DA 6-dim framework — a poor fit here), this script groups
ovagent_bench's own 143 rubric check items into six dimensions that
actually exist in the benchmark and computes pass-rate per (model, mode,
dimension). Everything is derived from `results/*/grades.csv` — no LLM
judge, same scale as the headline pass% table.

The six native dimensions (audit `analysis/native_dim_map.csv` for the
exact key→dim assignment):

  1. object_plumbing            - artifact placed in the right AnnData slot
  2. data_integrity             - stored values in correct scale/layer/dtype
  3. tool_grounding             - real algorithm invoked, not a stub
  4. methodological_robustness  - multi-method comparison & agreement
  5. quantitative_quality       - metric/count meets a threshold
  6. biological_plausibility    - result respects known biology

Usage:
  python3 scripts/radar_native.py           # writes audit csv + radar png
  python3 scripts/radar_native.py --seeds 0 # restrict to seed=0
"""
import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
ANALYSIS = Path(__file__).resolve().parent

DIMS = [
    "object_plumbing",
    "data_integrity",
    "tool_grounding",
    "methodological_robustness",
    "quantitative_quality",
    "biological_plausibility",
]

# Manual key->dim mapping. Edit & re-run to refine. Anything missing falls
# through to "object_plumbing" with a warning.
DIM_KEYS = {
    "data_integrity": {
        "counts_int", "counts_layer", "x_is_normalized", "per_cell_sum",
        "mt_values_sane", "umap_shape", "velocity_layer", "perturb_layer",
    },
    "tool_grounding": {
        "ccc_real_call", "cnmf_real_call", "deconv_real_call", "fm_real_call",
        "ppi_real_call", "wgcna_real_call", "perturbation_real_call",
        "stt_real_call", "fine_tune_evidence", "min_cells_generated",
    },
    "methodological_robustness": {
        "multi_doublet_methods", "multi_method_agreement", "multi_annotation",
        "multi_method_de_consistency", "multi_method_da_results",
        "pseudotime_methods_agree", "modes_directionally_agree",
        "two_distinct_methods", "two_distinct_groups",
        "multi_resolution_stability", "two_pseudotime_obs",
    },
    "quantitative_quality": {
        # metrics / statistical tests
        "ari_vs_celltype", "ari_vs_ground_truth", "celltype_silhouette_fm",
        "spatial_silhouette", "kbet_lisi_quantitative",
        "batch_separation_reduced", "group_separation", "supervised_separation",
        "beta_permanova", "alpha_group_test", "unifrac_test", "unifrac_distance",
        "asv_shannon", "genus_shannon", "shannon_ordering",
        "rank_present", "rank_substructure",
        # count / shape thresholds
        "cluster_count", "n_domains", "n_modules", "n_programs", "n_regulons",
        "domain_unique_count", "label_unique_count", "subcluster_count",
        "subset_size", "hvg_count", "embedding_dim_at_least",
        "predicted_coverage", "pcoa_variance", "genes_filtered",
        "cells_after_qc", "cells_after_full_qc", "svg_count",
    },
    "biological_plausibility": {
        "umap_preserves_celltype", "celltype_structure_preserved",
        "hvg_includes_canonical_markers", "top_markers_canonical_per_cluster",
        "svg_marker", "synthetic_recovers_celltypes",
        "aucell_celltype_specificity", "tb_celltype_specificity",
        "ccc_reference_lr_hit", "dominant_program",
        "velocity_flows_from_root", "pseudotime_anchored_at_root",
        "branch_detection", "composition_matches_reference",
        "cd4_cd8_split", "only_t_cells",
    },
    # everything else -> object_plumbing (default)
}


def classify(key):
    for dim, keys in DIM_KEYS.items():
        if key in keys:
            return dim
    return "object_plumbing"


def load_grades():
    rows = [pd.read_csv(f, low_memory=False) for f in
            glob.glob(str(PROJECT / "results/*/grades.csv"))]
    df = pd.concat(rows, ignore_index=True).drop_duplicates(
        subset=["system", "model_id", "task_id", "seed"])
    df["mode"] = df.system.apply(
        lambda s: "omicverse" if s.endswith("_omicverse") else
                  ("baseline" if s.endswith("_baseline") else None))
    return df[df["mode"].notna()].copy()


def write_audit(all_keys):
    out = ANALYSIS / "native_dim_map.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["check_key", "dimension"])
        for k in sorted(all_keys):
            w.writerow([k, classify(k)])
    counts = defaultdict(int)
    for k in all_keys:
        counts[classify(k)] += 1
    print(f"wrote {out}  ({len(all_keys)} keys)")
    for d in DIMS:
        print(f"  {d:30s}  {counts[d]:3d} keys")
    return out


def aggregate(df):
    """-> {(model, mode): {dim: (passed, total)}}"""
    bucket = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for _, row in df.iterrows():
        rj = row.get("rubric_json")
        if not isinstance(rj, str):
            continue
        try:
            d = json.loads(rj)
        except Exception:
            continue
        key = (row.model_id, row["mode"])
        for ck, v in d.items():
            dim = classify(ck)
            bucket[key][dim][1] += 1
            if v:
                bucket[key][dim][0] += 1
    return bucket


def plot(bucket, seeds_tag):
    models = sorted({k[0] for k in bucket})
    n_models = len(models)
    cols = 4
    rows = (n_models + cols - 1) // cols
    fig = plt.figure(figsize=(cols * 4.0, rows * 4.4 + 1.2))
    angles = np.linspace(0, 2 * np.pi, len(DIMS), endpoint=False).tolist() + [0]

    # baseline = gray (under), +omicverse = green (over) — consistent per panel
    MODE_STYLE = {
        "baseline":  {"color": "#8A8A8A", "fill": 0.10, "lw": 1.7, "z": 3},
        "omicverse": {"color": "#2BA66B", "fill": 0.22, "lw": 2.2, "z": 5},
    }

    overall = {k: float(np.mean([100*p/t if t else 0
                                  for p, t in bucket[k].values()]))
               for k in bucket}
    models.sort(key=lambda m: -max(overall.get((m, "baseline"), 0),
                                   overall.get((m, "omicverse"), 0)))

    summary_rows = []
    for i, m in enumerate(models):
        ax = fig.add_subplot(rows, cols, i + 1, polar=True)
        # plot baseline first so omicverse sits on top
        for mode in ("baseline", "omicverse"):
            k = (m, mode)
            if k not in bucket:
                continue
            st = MODE_STYLE[mode]
            vals = []
            for d in DIMS:
                p, t = bucket[k][d]
                vals.append(100 * p / t if t else 0)
            vc = vals + [vals[0]]
            ax.fill(angles, vc, color=st["color"], alpha=st["fill"], zorder=st["z"]-1)
            ax.plot(angles, vc, "-", lw=st["lw"], color=st["color"],
                    marker="o", ms=4,
                    label=f"{mode} ({overall[k]:.0f})", zorder=st["z"])
            summary_rows.append((m, mode, overall[k], vals))
        # uplift annotation
        b = overall.get((m, "baseline")); o = overall.get((m, "omicverse"))
        if b is not None and o is not None:
            delta = o - b
            sign = "+" if delta >= 0 else ""
            ax.text(0.5, -0.30, f"Δ {sign}{delta:.1f}", transform=ax.transAxes,
                    fontsize=9, color="#2BA66B" if delta >= 0 else "#c0392b",
                    ha="center", va="top", fontweight="bold")
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([])
        for ang, lab in zip(angles[:-1], DIMS):
            ax.text(ang, 122, lab.replace("_", "\n"), ha="center",
                    va="center", fontsize=7.0, color="#333")
        ax.set_yticks([25, 50, 75, 100])
        ax.set_yticklabels(["25", "50", "75", "100"], fontsize=6.5, color="#777")
        ax.set_ylim(0, 100)
        ax.set_rlabel_position(90)
        ax.grid(True, ls=":", alpha=0.45)
        ax.spines["polar"].set_color("#cccccc")
        ax.set_title(m, fontsize=10, color="#222", pad=24, fontweight="bold")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22),
                  fontsize=7.5, frameon=False, ncol=2)

    fig.suptitle(
        f"ovagent_bench native 6-dim capability profile  |  baseline (dashed) "
        f"vs +omicverse (solid)  |  {seeds_tag}\n"
        "Each dimension = pass-rate over its rubric check items "
        "(see analysis/native_dim_map.csv for key→dim audit)",
        fontsize=11.5, y=0.995, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    out = ANALYSIS / "ovagent_radar_native.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"saved {out}")
    return summary_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="restrict to listed seeds (default: all)")
    args = ap.parse_args()

    df = load_grades()
    if args.seeds:
        df = df[df.seed.isin(args.seeds)]
        seeds_tag = f"seeds {','.join(map(str, args.seeds))}"
    else:
        seeds_tag = "all seeds pooled"
    print(f"loaded {len(df)} cells ({seeds_tag})")

    # collect all keys for audit
    all_keys = set()
    for rj in df.rubric_json.dropna():
        try:
            all_keys.update(json.loads(rj).keys())
        except Exception:
            pass
    write_audit(all_keys)

    bucket = aggregate(df)
    summary = plot(bucket, seeds_tag)

    print()
    print(f"{'model':32s} {'mode':10s} overall  " +
          "  ".join(f"{d[:9]:>9s}" for d in DIMS))
    for m, mode, ov, vals in summary:
        print(f"{m:32s} {mode:10s} {ov:5.1f}    " +
              "  ".join(f"{v:7.1f}  " for v in vals))


if __name__ == "__main__":
    main()
