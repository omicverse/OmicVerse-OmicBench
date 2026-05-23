#!/usr/bin/env python3
"""Per-task cost for OmicVerse-OmicBench runs.

  python3 analysis/bench_cost.py                       # all known runs
  python3 analysis/bench_cost.py <run_name> [...]      # subset

For each run (a dir under trajectories/) reports per-task tokens, NAIVE
cost (no prompt caching) and CACHE-ADJUSTED cost.

Why two numbers — an OmicBench task is a long agent loop; every LLM
call resends the growing conversation, so cumulative input tokens
reach 1-2M per task. But ~95% of that is an unchanged prefix that
prompt caching bills at a steep discount.

Per-call token usage is read from each assistant message's
`extra.response.usage.{prompt_tokens, completion_tokens}` in the
mini-swe-agent trajectory. Timestamps come from `extra.timestamp`.

Cache estimation: per cell, sort calls by timestamp and treat only
the positive per-call input-token increments as fresh (uncached);
the rest is a cache hit. This assumes the cache stays warm between
calls, so the cache-adjusted number is a best-case (lower-bound)
cost; the truth sits between naive and cache-adjusted.

Writes a CSV summary to `analysis/cost_summary.csv` so the chart
script can run without re-reading every trajectory.
"""
import argparse
import csv
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
ANALYSIS = Path(__file__).resolve().parent

# (input, cached_input, output) USD per 1M tokens. Public list prices,
# approximate as of 2026-05; cached_input set to ~10% of input where the
# provider documents prompt-caching with the standard discount, lower
# for DeepSeek (token-cache near zero), and equal to input for
# providers without a documented cache discount.
# Edit and rerun to refresh.
PRICING = {
    "gpt-5.5":                       (5.00, 0.500,  30.00),
    "deepseek-v4-pro":               (1.74, 0.0174,  3.48),
    "deepseek-v4-flash":             (0.14, 0.0028,  0.28),
    "gemini-3.1-flash-lite-preview": (0.10, 0.025,   0.40),
    "glm-5.1":                       (0.60, 0.06,    2.40),
    "MiniMax-M2.7":                  (0.30, 0.030,   1.20),
    # Qwen3.6 35B-A3B priced from OpenRouter (Alibaba's open-weights model
    # served by 3rd-party endpoints; the v1.0 sweep actually ran on local
    # ollama at $0, so this row is an "if-on-OpenRouter" reference for
    # apples-to-apples cost comparison with the API providers above.
    # OpenRouter does not document a prompt-cache discount for this
    # model — cached price equals input price, so the cache-adjusted
    # number reduces to naive for qwen.
    "qwen3.6:35b-a3b-256k":          (0.15, 0.15,    1.00),
}

# Map "model_short" in trajectory filenames -> model_id key in PRICING.
# Filenames flatten the model id, replacing all non-alphanumerics with `_`.
def _model_safe(model_id):
    return re.sub(r"[^A-Za-z0-9]+", "_", model_id).strip("_")

MODEL_SAFE_TO_ID = {_model_safe(m): m for m in PRICING}


def parse_traj_path(traj_path):
    """results/runs/<system>/<task>__<model_safe>_seed<n>/minisweagent_trajectory.json -> dict."""
    p = Path(traj_path)
    cell_dir = p.parent.name
    system = p.parent.parent.name
    m = re.match(r"(?P<task>.+?)__(?P<model_safe>.+?)_seed(?P<seed>\d+)$", cell_dir)
    if not m:
        return None
    d = m.groupdict()
    d["system"] = system
    d["seed"] = int(d["seed"])
    d["model_id"] = MODEL_SAFE_TO_ID.get(d["model_safe"], d["model_safe"])
    return d


def cell_tokens(traj_path):
    """Return (calls, total_in, fresh_in, total_out) for one trajectory.

    Reads per-call `prompt_tokens / completion_tokens / timestamp` out of
    each assistant message's `extra.response.usage`. Skips calls that
    don't carry usage data (some providers / some failure paths).
    """
    try:
        d = json.loads(Path(traj_path).read_text())
    except Exception:
        return None
    calls = []
    for m in d.get("messages", []):
        if m.get("role") != "assistant":
            continue
        ex = m.get("extra")
        if not isinstance(ex, dict):
            continue
        ts = ex.get("timestamp") or 0
        usage = (ex.get("response") or {}).get("usage") or {}
        pin = usage.get("prompt_tokens") or 0
        pout = usage.get("completion_tokens") or 0
        if pin <= 0 and pout <= 0:
            continue
        calls.append((ts, pin, pout))
    if not calls:
        return None
    calls.sort()
    seq_in = [c[1] for c in calls]
    tot_out = sum(c[2] for c in calls)
    tot_in = sum(seq_in)
    fresh = seq_in[0] + sum(max(0, seq_in[i] - seq_in[i - 1])
                            for i in range(1, len(seq_in)))
    return len(calls), tot_in, fresh, tot_out


def per_cell_rows(runs_root):
    """Yield one dict per (system, task, model, seed) cell.

    runs_root layout: <runs_root>/<system>/<task>__<model>_seed<n>/minisweagent_trajectory.json
    """
    for path in glob.glob(str(runs_root / "*/*/minisweagent_trajectory.json")):
        meta = parse_traj_path(path)
        if not meta:
            continue
        tk = cell_tokens(path)
        if not tk:
            continue
        n_calls, tot_in, fresh, tot_out = tk
        yield {
            "task_id": meta["task"],
            "system": meta["system"],
            "model_id": meta["model_id"],
            "seed": meta["seed"],
            "n_calls": n_calls,
            "in_tok": tot_in,
            "fresh_tok": fresh,
            "out_tok": tot_out,
        }


def summarise(rows):
    """-> {(model_id, arm): {n, in_tok_mean, fresh_tok_mean, out_tok_mean,
                              cache_hit, naive_usd, cached_usd}}

    arm = 'baseline' or 'omicverse' inferred from `system` suffix; cells
    that don't fit are aggregated under arm='other'.
    """
    bucket = defaultdict(list)
    for r in rows:
        s = r["system"]
        if s.endswith("_omicverse"):
            arm = "omicverse"
        elif s.endswith("_baseline"):
            arm = "baseline"
        elif s.endswith("_omicverse_no_registry"):
            arm = "omicverse_no_registry"
        elif s.endswith("_omicverse_doc_rag"):
            arm = "omicverse_doc_rag"
        else:
            arm = s
        bucket[(r["model_id"], arm)].append(r)

    out = {}
    for k, cells in bucket.items():
        model_id = k[0]
        p = PRICING.get(model_id, (0.0, 0.0, 0.0))
        pin, pcache, pout = p
        tot_in = sum(c["in_tok"] for c in cells)
        tot_fresh = sum(c["fresh_tok"] for c in cells)
        tot_out = sum(c["out_tok"] for c in cells)
        n = len(cells)
        cached = tot_in - tot_fresh
        naive_usd = (tot_in * pin + tot_out * pout) / 1e6 / n
        cached_usd = (tot_fresh * pin + cached * pcache + tot_out * pout) / 1e6 / n
        out[k] = {
            "n": n,
            "in_tok_mean": tot_in / n,
            "fresh_tok_mean": tot_fresh / n,
            "out_tok_mean": tot_out / n,
            "cache_hit": (cached / tot_in) if tot_in else 0.0,
            "naive_usd": naive_usd,
            "cached_usd": cached_usd,
        }
    return out


def write_summary_csv(rows, out_path):
    fields = ["task_id", "system", "model_id", "seed",
              "n_calls", "in_tok", "fresh_tok", "out_tok"]
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default=str(PROJECT / "results" / "runs"),
                    help="Root with <system>/<task>__<model>_seed<n>/minisweagent_trajectory.json cells")
    ap.add_argument("--summary-csv", default=str(ANALYSIS / "cost_summary.csv"))
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    if not runs_root.exists():
        print(f"runs dir missing: {runs_root}", file=sys.stderr)
        sys.exit(1)

    rows = list(per_cell_rows(runs_root))
    print(f"read {len(rows)} cells from {runs_root}", file=sys.stderr)
    write_summary_csv(rows, args.summary_csv)
    print(f"wrote {args.summary_csv}", file=sys.stderr)

    agg = summarise(rows)
    print(f"\n{'model_id':32s} {'arm':22s} {'n':>4s}  "
          f"{'in/task':>10s}  {'fresh':>10s}  {'cache%':>6s}  "
          f"{'naive$':>7s}  {'cached$':>8s}")
    print("-" * 110)
    for (m, arm), r in sorted(agg.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        print(f"{m:32s} {arm:22s} {r['n']:4d}  "
              f"{r['in_tok_mean']:10,.0f}  {r['fresh_tok_mean']:10,.0f}  "
              f"{100*r['cache_hit']:5.1f}%  "
              f"{r['naive_usd']:7.2f}  {r['cached_usd']:8.3f}")


if __name__ == "__main__":
    main()
