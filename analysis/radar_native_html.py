#!/usr/bin/env python3
"""Interactive HTML version of the native 6-dim capability radar.

Same data + dimensions as radar_native.py, but renders to a standalone
Plotly HTML file. 7 mini-radars (one per LLM) in a 4×2 subplot grid;
each panel overlays baseline (gray) and +OmicVerse (green); hover shows
the exact per-dimension pass-rate.

  python3 analysis/radar_native_html.py
  # -> analysis/ovagent_radar_native.html

  python3 analysis/radar_native_html.py --inline-plotly
  # -> single-file no-network bundle (~3 MB)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent))
from radar_native import DIMS, load_grades, aggregate   # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
ANALYSIS = Path(__file__).resolve().parent

MODE_STYLE = {
    "baseline":  {"line": "#8A8A8A", "fill": "rgba(138,138,138,0.18)", "width": 1.8},
    "omicverse": {"line": "#2BA66B", "fill": "rgba(43,166,107,0.30)",  "width": 2.4},
}


def model_means(bucket):
    """-> {(model, mode): mean of dim percentages 0..100, mean_dim_vals list}"""
    out = {}
    for k, dim_counts in bucket.items():
        vals = []
        for d in DIMS:
            p, t = dim_counts[d]
            vals.append(100 * p / t if t else 0)
        out[k] = (float(np.mean(vals)), vals)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--inline-plotly", action="store_true")
    ap.add_argument("--out", default=str(ANALYSIS / "ovagent_radar_native.html"))
    args = ap.parse_args()

    df = load_grades()
    if args.seeds:
        df = df[df.seed.isin(args.seeds)]
        seeds_tag = f"seeds {','.join(map(str, args.seeds))}"
    else:
        seeds_tag = "all seeds pooled"

    bucket = aggregate(df)
    means = model_means(bucket)
    models = sorted({k[0] for k in bucket},
                    key=lambda m: -max(means.get((m, "baseline"),  (0, []))[0],
                                       means.get((m, "omicverse"), (0, []))[0]))

    cols = 4
    rows = (len(models) + cols - 1) // cols

    panel_titles = []
    for m in models:
        b = means.get((m, "baseline"),  (None, []))[0]
        o = means.get((m, "omicverse"), (None, []))[0]
        delta = (o - b) if (b is not None and o is not None) else None
        if delta is not None:
            sign = "+" if delta >= 0 else ""
            color = "#2BA66B" if delta >= 0 else "#c0392b"
            panel_titles.append(
                f"<b>{m}</b><br>"
                f"<span style='font-size:11px;color:{color}'>Δ {sign}{delta:.1f}</span>")
        else:
            panel_titles.append(f"<b>{m}</b>")

    specs = [[{"type": "polar"}] * cols for _ in range(rows)]
    fig = make_subplots(rows=rows, cols=cols, specs=specs,
                        subplot_titles=panel_titles,
                        horizontal_spacing=0.06, vertical_spacing=0.13)

    # The 6 axis labels close the loop by repeating dim[0] at the end.
    theta_labels = DIMS + [DIMS[0]]

    for i, m in enumerate(models):
        r, c = (i // cols) + 1, (i % cols) + 1
        for mode in ("baseline", "omicverse"):
            k = (m, mode)
            if k not in means:
                continue
            overall, vals = means[k]
            vals_closed = vals + [vals[0]]
            hover = "<br>".join(
                f"{d}: {v:.1f}%" for d, v in zip(DIMS, vals)
            )
            st = MODE_STYLE[mode]
            fig.add_trace(go.Scatterpolar(
                r=vals_closed,
                theta=theta_labels,
                mode="lines+markers",
                line=dict(color=st["line"], width=st["width"]),
                marker=dict(size=5, color=st["line"]),
                fill="toself", fillcolor=st["fill"],
                name=f"{mode} ({overall:.0f}%)",
                legendgroup=mode,
                showlegend=(i == 0),
                hovertemplate=(
                    f"<b>{m}</b> · {mode}<br>"
                    "%{theta}: %{r:.1f}%<extra></extra>"
                ),
            ), row=r, col=c)

    polar_kwargs = dict(
        radialaxis=dict(range=[0, 100], showticklabels=True,
                        tickvals=[25, 50, 75, 100], tickfont=dict(size=8),
                        gridcolor="#d8d8d8", color="#777", angle=90),
        angularaxis=dict(tickfont=dict(size=10, color="#222"),
                         direction="clockwise",
                         gridcolor="#dddddd",
                         linecolor="#bbb"),
        bgcolor="white",
    )
    for i in range(1, rows * cols + 1):
        # Only configure polar axes that actually exist (1 .. n_models)
        if i > len(models):
            continue
        suffix = "" if i == 1 else str(i)
        fig.update_layout(**{f"polar{suffix}": polar_kwargs})

    fig.update_layout(
        title=dict(
            text=("<b>OmicVerse-OmicBench — 6-dim capability radar</b>"
                  "<br><span style='font-size:13px;color:#777'>"
                  f"baseline (gray) vs +OmicVerse (green), {seeds_tag}; "
                  "Δ under each panel = overall pass-rate uplift"
                  "</span>"),
            x=0.5, xanchor="center", y=0.985,
        ),
        template="simple_white",
        font=dict(family="IBM Plex Sans, Inter, Helvetica, Arial, sans-serif",
                  size=13, color="#222"),
        width=cols * 380, height=rows * 460 + 100,
        margin=dict(l=20, r=20, t=130, b=40),
        legend=dict(
            x=0.5, y=-0.04, xanchor="center", yanchor="top",
            orientation="h", bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#ccc", borderwidth=1, font=dict(size=12),
        ),
        hoverlabel=dict(bgcolor="white", font=dict(size=12), bordercolor="#888"),
    )

    out = Path(args.out)
    fig.write_html(
        str(out),
        include_plotlyjs=("inline" if args.inline_plotly else "cdn"),
        full_html=True,
        config={"displaylogo": False, "responsive": True,
                "toImageButtonOptions": {"format": "png",
                                         "filename": "ovagent_radar_native",
                                         "width": cols * 380,
                                         "height": rows * 460 + 100, "scale": 2}},
    )
    size_kb = out.stat().st_size / 1024
    print(f"saved {out}  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
