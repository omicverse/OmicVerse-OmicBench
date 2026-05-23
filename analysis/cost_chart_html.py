#!/usr/bin/env python3
"""Interactive HTML version of cost_chart.py.

Same data as cost_chart.py (analysis/cost_summary.csv + results/*/grades.csv)
but renders to a standalone Plotly HTML file that can be embedded in
docs sites, omicverse.org, GitHub Pages, etc.

  python3 analysis/cost_chart_html.py
  # -> analysis/cost_vs_score.html  (standalone, self-contained ~3 MB)

The HTML loads Plotly from a CDN; pass `--inline-plotly` to inline the
library and produce a single-file no-network bundle.
"""
import argparse
import csv
import glob
import sys
from collections import defaultdict
from pathlib import Path

import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_cost import PRICING, summarise   # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
ANALYSIS = Path(__file__).resolve().parent

# Provider palette — same as the PNG version
PROVIDER = {
    "gpt-5.5":                       ("OpenAI",          "#222222"),
    "deepseek-v4-pro":               ("DeepSeek",        "#3B6FE0"),
    "deepseek-v4-flash":             ("DeepSeek",        "#3B6FE0"),
    "gemini-3.1-flash-lite-preview": ("Google · Gemini", "#4FB99F"),
    "glm-5.1":                       ("Zhipu",           "#A04EBF"),
    "MiniMax-M2.7":                  ("MiniMax",         "#E8894C"),
    "qwen3.6:35b-a3b-256k":          ("Alibaba (open)",  "#6E8B3D"),
}
OMICVERSE_RING = "#C0392B"

SHORT_LABEL = {
    "gpt-5.5":                       "gpt-5.5",
    "deepseek-v4-pro":               "DeepSeek v4-pro",
    "deepseek-v4-flash":             "DeepSeek v4-flash",
    "gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash Lite",
    "glm-5.1":                       "GLM-5.1",
    "MiniMax-M2.7":                  "MiniMax-M2.7",
    "qwen3.6:35b-a3b-256k":          "Qwen3.6 35B-A3B",
}


def load_cost_rows():
    with open(ANALYSIS / "cost_summary.csv") as fh:
        rows = []
        for r in csv.DictReader(fh):
            for k in ("seed", "n_calls", "in_tok", "fresh_tok", "out_tok"):
                r[k] = int(r[k])
            rows.append(r)
        return rows


def load_grades():
    bucket = defaultdict(list)
    for f in glob.glob(str(PROJECT / "results" / "*" / "grades.csv")):
        with open(f) as fh:
            for row in csv.DictReader(fh):
                s = row["system"]
                if s.endswith("_omicverse"):   arm = "omicverse"
                elif s.endswith("_baseline"):  arm = "baseline"
                elif "no_registry" in s:       arm = "omicverse_no_registry"
                elif "doc_rag" in s:           arm = "omicverse_doc_rag"
                else:                          arm = s
                p = row.get("passed", "").strip().lower()
                if p in ("true", "1", "1.0"):
                    bucket[(row["model_id"], arm)].append(1)
                elif p in ("false", "0", "0.0"):
                    bucket[(row["model_id"], arm)].append(0)
    return {k: (sum(v) / len(v), len(v)) for k, v in bucket.items() if v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inline-plotly", action="store_true",
                    help="Inline plotly.js (~3 MB) so the file works without network.")
    ap.add_argument("--out", default=str(ANALYSIS / "cost_vs_score.html"))
    args = ap.parse_args()

    cost_agg = summarise(load_cost_rows())
    grades = load_grades()

    fig = go.Figure()
    provider_legend_seen = set()
    pareto_pts = []

    # Pass 1 — draw uplift connector LINES first so they sit underneath markers.
    # (Plotly z-order = trace order. NB: avoid add_annotation arrows on log
    # axes — the `ax`/`axref` tail is not log-transformed, arrows misalign.)
    for model_id in PRICING:
        if PRICING[model_id] == (0.0, 0.0, 0.0):
            continue
        b = cost_agg.get((model_id, "baseline"))
        o = cost_agg.get((model_id, "omicverse"))
        bg = grades.get((model_id, "baseline"))
        og = grades.get((model_id, "omicverse"))
        if not (b and o and bg and og):
            continue
        fig.add_trace(go.Scatter(
            x=[b["cached_usd"], o["cached_usd"]],
            y=[bg[0] * 100, og[0] * 100],
            mode="lines",
            line=dict(color=OMICVERSE_RING, width=1.4),
            opacity=0.45,
            hoverinfo="skip",
            showlegend=False,
        ))

    # Pass 2 — draw the (baseline, OmicVerse) markers on top.
    for model_id in PRICING:
        if PRICING[model_id] == (0.0, 0.0, 0.0):
            continue
        provider_name, color = PROVIDER.get(model_id, ("other", "#666"))
        short = SHORT_LABEL.get(model_id, model_id)

        for arm, marker_dict, label_fmt in (
            ("baseline",
             dict(size=14, color="white", line=dict(color=color, width=2.5),
                  symbol="circle"),
             "{short}"),
            ("omicverse",
             dict(size=20, color=color, line=dict(color=OMICVERSE_RING, width=2.8),
                  symbol="circle"),
             "<b>OmicVerse[{short}]</b>"),
        ):
            agg = cost_agg.get((model_id, arm))
            grade_v = grades.get((model_id, arm))
            if not agg or not grade_v:
                continue
            cost = agg["cached_usd"]
            score = grade_v[0] * 100
            n = grade_v[1]

            hover = (f"<b>{short}</b> ({provider_name})<br>"
                     f"arm: {arm}<br>"
                     f"cost/task: ${cost:.3f}<br>"
                     f"Pass@1: {score:.1f}%<br>"
                     f"cells: {n}<br>"
                     f"in_tok mean: {agg['in_tok_mean']:,.0f}<br>"
                     f"cache hit (est): {100*agg['cache_hit']:.1f}%<extra></extra>")

            show_legend = (arm == "baseline"
                           and provider_name not in provider_legend_seen)
            if show_legend:
                provider_legend_seen.add(provider_name)

            fig.add_trace(go.Scatter(
                x=[cost], y=[score],
                mode="markers+text",
                marker=marker_dict,
                text=[label_fmt.format(short=short)],
                textposition="top right" if arm == "omicverse" else "bottom right",
                textfont=dict(size=11, color="#222"),
                hovertemplate=hover,
                name=provider_name,
                legendgroup=provider_name,
                showlegend=show_legend,
            ))
            pareto_pts.append((cost, score))

    # Pareto front
    pareto_pts.sort()
    front, best = [], -1e9
    for c, s in pareto_pts:
        if s > best:
            front.append((c, s)); best = s
    if front:
        fig.add_trace(go.Scatter(
            x=[c for c, _ in front], y=[s for _, s in front],
            mode="lines", line=dict(color="#999", width=1.2, dash="dash"),
            name="Pareto front", hoverinfo="skip",
        ))

    fig.update_xaxes(
        type="log",
        title="Cost per task (USD, log scale, cache-adjusted)",
        gridcolor="#e0e0e0",
        showline=True, linecolor="#888", mirror=True,
    )
    fig.update_yaxes(
        title="Pass@1 (%)",
        gridcolor="#e0e0e0",
        showline=True, linecolor="#888", mirror=True,
    )
    fig.update_layout(
        title=dict(
            text=("<b>OmicVerse-OmicBench — Pass@1 vs cost per task</b><br>"
                  "<span style='font-size:13px;color:#777'>"
                  "hollow = baseline · solid with red ring = +OmicVerse · "
                  "arrow = uplift</span>"),
            x=0.5, xanchor="center", y=0.97,
        ),
        template="simple_white",
        font=dict(family="IBM Plex Sans, Inter, Helvetica, Arial, sans-serif",
                  size=13, color="#222"),
        width=1100, height=720,
        margin=dict(l=70, r=40, t=110, b=70),
        legend=dict(
            x=0.99, y=0.02, xanchor="right", yanchor="bottom",
            bgcolor="rgba(255,255,255,0.92)", bordercolor="#ccc", borderwidth=1,
            font=dict(size=11),
        ),
        hoverlabel=dict(bgcolor="white", font=dict(size=12),
                        bordercolor="#888"),
    )

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    fig.write_html(
        str(out),
        include_plotlyjs=("inline" if args.inline_plotly else "cdn"),
        full_html=True,
        config={"displaylogo": False, "responsive": True,
                "toImageButtonOptions": {"format": "png", "filename": "omicbench_cost_vs_score",
                                         "width": 1400, "height": 900, "scale": 2}},
    )
    size_kb = out.stat().st_size / 1024
    print(f"saved {out}  ({size_kb:.0f} KB, "
          f"plotly={'inlined' if args.inline_plotly else 'CDN'})")


if __name__ == "__main__":
    main()
