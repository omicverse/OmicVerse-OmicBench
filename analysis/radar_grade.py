#!/usr/bin/env python3
"""LLM-as-judge variant: score trajectories on the BiomniBench-DA 6 dims.

This is included for completeness; the headline analysis uses
`radar_native.py`, which derives six dimensions directly from OmicBench's
own rubric pass-rates (no LLM judge needed). Use this script only if you
want to reproduce the cross-framework comparison with BiomniBench-DA's
six dims, which we found a poor fit for OmicBench (two of the six are
floor- or ceiling-effect dominated; see analysis/README.md).

Requires trajectories on disk (not shipped with this repo — re-run
`scripts/run.sh` to regenerate). The cached `radar_grades.jsonl` lets
`plot` reproduce the figure without re-grading.

  source bench-env.sh   # sets DEEPSEEK_API_KEY etc.
  analysis/radar_grade.py grade  [--systems ...] [--seeds 0]   [-j 6]
  analysis/radar_grade.py plot   [--systems ...]
"""
import argparse
import concurrent.futures as cf
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

import openai

PROJECT = Path(__file__).resolve().parents[1]
ANALYSIS = Path(__file__).resolve().parent
GRADES = ANALYSIS / "radar_grades.jsonl"

DIMS = ["data_handling", "method_selection", "statistical_rigor",
        "biological_interpretation", "scientific_reasoning", "source_reliability"]
DIM_LABELS = {d: d.replace("_", " ") for d in DIMS}

OMICVERSE_SYSTEMS = ["gpt_omicverse", "deepseek_omicverse",
                     "gemini_omicverse", "glm_omicverse",
                     "minimax_omicverse", "mini_swe_omicverse"]

PROMPT = """You are grading an AI agent's trajectory on a single-cell / omics analysis task. Score the agent's work on the six capability dimensions below, each on a 0-100 continuous scale (NOT pass/fail). Be calibrated: 100 means an expert-level answer; 70 means competent with notable gaps; 40 means substantial deficiencies; 0 means the dimension was unaddressed or fundamentally wrong.

Dimensions:
1. data_handling — Did the agent correctly load, inspect, validate, and structure the input AnnData / files? Were the right layers / obs columns / var fields used at the right scale (raw vs normalized vs log)? Were data integrity checks (shape, NaN, dtype, gene/cell counts) performed?
2. method_selection — Were the chosen tools/methods appropriate for the task? When the task asked for multi-method comparison, was that done? Did the agent justify its choice rather than default-apply?
3. statistical_rigor — Quality of metrics, statistical tests, thresholds, parameter scans, validation. Are quantitative claims backed by real computations (silhouette / ARI / permanova / p-values), not asserted?
4. biological_interpretation — Did the agent connect results to known biology (markers, cell types, signaling, pathways)? Do outputs respect expected biological structure?
5. scientific_reasoning — Quality of the agent's reasoning narrative: comparing alternatives, justifying decisions, acknowledging limitations / uncertainty, recognizing edge cases.
6. source_reliability — Did the agent ground its code in real APIs (correct omicverse / scanpy / sklearn calls, not hallucinated functions)? Did it consult docs / registry / authoritative references when uncertain? Was its citation discipline appropriate?

Task instruction:
<<<
{task_md}
>>>

Agent trajectory (assistant turns + bash invocations; truncated if needed):
<<<
{trajectory}
>>>

Machine-checked rubric outcomes from the harness (pass/fail per check, context only — your dimension scores need not align if the trajectory reveals more nuance):
{rubric_pass}

Return ONLY a JSON object on a single line, no prose, no markdown fence:
{{"data_handling": int, "method_selection": int, "statistical_rigor": int, "biological_interpretation": int, "scientific_reasoning": int, "source_reliability": int, "notes": "<one short sentence>"}}"""


def model_safe(model_id):
    return re.sub(r"[^A-Za-z0-9]+", "_", model_id).strip("_")


def find_traj(system, model_id, task_id, seed):
    safe = model_safe(model_id)
    candidates = [
        PROJECT / f"results/runs/{system}/{task_id}__{safe}_seed{seed}/minisweagent_trajectory.json",
        PROJECT / f"results/runs/{system}/{task_id}__{safe}_seed{seed}/log.txt",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def render_trajectory(path, max_chars=80000):
    """Compact, judge-friendly text of a mini-swe-agent trajectory.

    Keep system, user (truncated), and ALL assistant turns. Bash outputs
    are kept but tail-truncated. Total capped at ~80kB (~20k tokens)."""
    if path.suffix == ".txt":
        text = path.read_text(errors="ignore")
        return text[-max_chars:] if len(text) > max_chars else text
    d = json.loads(path.read_text())
    msgs = d.get("messages", [])
    parts = []
    user_seen = False
    for m in msgs:
        role = m.get("role", "?")
        content = m.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content)
        if not content.strip():
            continue
        if role == "system":
            parts.append(f"[system prompt (trimmed)]\n{content[:400]}...")
        elif role == "user" and not user_seen:
            parts.append(f"[user task]\n{content[:4000]}")
            user_seen = True
        elif role in ("user", "tool"):
            # bash / tool output - head+tail trim to keep useful signal
            if len(content) > 2400:
                content = content[:1400] + "\n...[trimmed]...\n" + content[-900:]
            parts.append(f"[bash/tool output]\n{content}")
        elif role == "assistant":
            parts.append(f"[assistant reasoning + command]\n{content[:6000]}")
        elif role == "exit":
            continue
        else:
            parts.append(f"[{role}]\n{content[:1500]}")
    text = "\n\n".join(parts)
    if len(text) > max_chars:
        head = text[: max_chars // 2]
        tail = text[-max_chars // 2 :]
        text = head + "\n\n[... trajectory truncated for length ...]\n\n" + tail
    return text


def load_task_md(task_id):
    p = PROJECT / "tasks_md" / f"{task_id}.md"
    return p.read_text() if p.exists() else f"(task md missing for {task_id})"


def load_rubric_pass(rubric_json):
    if not rubric_json or rubric_json != rubric_json:  # NaN guard
        return "(no rubric outcomes)"
    try:
        d = json.loads(rubric_json)
        return ", ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in d.items())
    except Exception:
        return str(rubric_json)[:500]


def already_graded():
    seen = {}
    if GRADES.exists():
        for line in GRADES.read_text().splitlines():
            try:
                o = json.loads(line)
                seen[(o["system"], o["model_id"], o["task_id"], o["seed"])] = o
            except Exception:
                pass
    return seen


def call_judge(client, model, task_md, traj_text, rubric_pass, retries=3):
    prompt = PROMPT.format(task_md=task_md[:6000], trajectory=traj_text,
                           rubric_pass=rubric_pass[:3000])
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=4096,  # reasoning tokens eat budget on v4-pro
                timeout=300,
            )
            txt = resp.choices[0].message.content.strip()
            # strip code fences if present
            txt = re.sub(r"^```(?:json)?\s*", "", txt)
            txt = re.sub(r"\s*```\s*$", "", txt)
            obj = json.loads(txt)
            for d in DIMS:
                if d not in obj:
                    raise ValueError(f"missing {d}")
                obj[d] = max(0, min(100, int(obj[d])))
            usage = getattr(resp, "usage", None)
            return obj, {"in": getattr(usage, "prompt_tokens", 0),
                         "out": getattr(usage, "completion_tokens", 0)}
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"judge failed after {retries}: {last_err}")


def cmd_grade(args):
    import pandas as pd
    rows = []
    for f in glob.glob(str(PROJECT / "results/*/grades.csv")):
        try:
            rows.append(pd.read_csv(f, low_memory=False))
        except Exception:
            pass
    df = pd.concat(rows, ignore_index=True).drop_duplicates(
        subset=["system", "model_id", "task_id", "seed"])
    df = df[df["system"].isin(args.systems)]
    if args.seeds:
        df = df[df["seed"].isin(args.seeds)]
    df = df.sort_values(["system", "model_id", "task_id", "seed"]).reset_index(drop=True)
    print(f"cells to grade: {len(df)}", file=sys.stderr)

    seen = already_graded()
    pending = [r for _, r in df.iterrows()
               if (r.system, r.model_id, r.task_id, int(r.seed)) not in seen]
    print(f"already cached: {len(seen)}; pending: {len(pending)}", file=sys.stderr)
    if not pending:
        return

    client = openai.OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1"),
    )

    GRADES.parent.mkdir(exist_ok=True)
    out_fh = open(GRADES, "a")
    lock_done = {"n": 0, "tin": 0, "tout": 0}

    def grade_one(r):
        traj = find_traj(r.system, r.model_id, r.task_id, int(r.seed))
        if traj is None:
            return None, f"no trajectory for {r.system}/{r.task_id}/{r.model_id}/seed{r.seed}"
        try:
            text = render_trajectory(traj)
            obj, usage = call_judge(client, args.model,
                                    load_task_md(r.task_id), text,
                                    load_rubric_pass(r.rubric_json))
            obj.update({"system": r.system, "model_id": r.model_id,
                        "task_id": r.task_id, "seed": int(r.seed),
                        "harness_score": float(r.score) if r.score == r.score else None,
                        "judge_model": args.model,
                        "in_tok": usage["in"], "out_tok": usage["out"]})
            return obj, None
        except Exception as e:
            return None, f"{r.system}/{r.task_id}/{r.model_id}/seed{r.seed}: {e}"

    with cf.ThreadPoolExecutor(max_workers=args.j) as ex:
        futs = {ex.submit(grade_one, r): r for r in pending}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            obj, err = fut.result()
            if err:
                print(f"  ERR {err}", file=sys.stderr)
                continue
            out_fh.write(json.dumps(obj) + "\n")
            out_fh.flush()
            lock_done["n"] += 1
            lock_done["tin"] += obj["in_tok"]
            lock_done["tout"] += obj["out_tok"]
            if i % 5 == 0 or i == len(pending):
                cost = (lock_done["tin"] * 1.74 + lock_done["tout"] * 3.48) / 1e6
                print(f"  [{i}/{len(pending)}] done={lock_done['n']} "
                      f"in={lock_done['tin']:,} out={lock_done['tout']:,} "
                      f"approx_cost=${cost:.2f}", file=sys.stderr)
    out_fh.close()


def cmd_plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import defaultdict

    if not GRADES.exists():
        print(f"no grades file at {GRADES}", file=sys.stderr); sys.exit(1)

    # group by (model_id, mode) where mode = 'omicverse' or 'baseline'
    bucket = defaultdict(lambda: defaultdict(list))  # (model, mode) -> dim -> list
    for line in GRADES.read_text().splitlines():
        try: o = json.loads(line)
        except Exception: continue
        sys_ = o["system"]
        if sys_.endswith("_omicverse"):
            mode = "omicverse"
        elif sys_.endswith("_baseline"):
            mode = "baseline"
        else:
            continue
        for d in DIMS:
            bucket[(o["model_id"], mode)][d].append(o[d])

    if not bucket:
        print("no graded data", file=sys.stderr); sys.exit(1)

    means = {k: {d: float(np.mean(bucket[k][d])) if bucket[k][d] else None
                 for d in DIMS} for k in bucket}
    overall = {k: float(np.mean([v for v in means[k].values() if v is not None]))
               for k in bucket}

    models = sorted({k[0] for k in bucket},
                    key=lambda m: -max(overall.get((m, "baseline"), 0),
                                       overall.get((m, "omicverse"), 0)))
    n_models = len(models)

    palette = {  # one color per model
        "gpt-5.5":                       "#1F4FBF",
        "deepseek-v4-pro":               "#E8894C",
        "deepseek-v4-flash":             "#F2BB94",
        "glm-5.1":                       "#A04EBF",
        "MiniMax-M2.7":                  "#F0C040",
        "gemini-3.1-flash-lite-preview": "#4FB99F",
        "qwen3.6:35b-a3b-256k":          "#777777",
    }

    cols = 4
    rows = (n_models + cols - 1) // cols
    fig = plt.figure(figsize=(cols * 4.0, rows * 4.4 + 1.2))
    angles = np.linspace(0, 2 * np.pi, len(DIMS), endpoint=False).tolist() + [0]

    for i, m in enumerate(models):
        ax = fig.add_subplot(rows, cols, i + 1, polar=True)
        color = palette.get(m, "#555555")
        for mode, ls, alpha_fill, mk in (("baseline", "--", 0.05, "o"),
                                          ("omicverse", "-", 0.18, "o")):
            key = (m, mode)
            if key not in bucket:
                continue
            vals = [means[key][d] or 0 for d in DIMS]
            vc = vals + [vals[0]]
            ax.fill(angles, vc, color=color, alpha=alpha_fill, zorder=2)
            n = len(bucket[key][DIMS[0]])
            ax.plot(angles, vc, ls, lw=1.8, color=color, marker=mk, ms=3.5,
                    label=f"{mode} ({overall[key]:.0f}, n={n})", zorder=4)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([])
        for ang, lab in zip(angles[:-1], DIMS):
            ax.text(ang, 118, lab.replace("_", "\n"), ha="center", va="center",
                    fontsize=7.5, color="#333")
        ax.set_yticks([25, 50, 75])
        ax.set_yticklabels(["25", "50", "75"], fontsize=6.5, color="#777")
        ax.set_ylim(0, 100)
        ax.set_rlabel_position(90)
        ax.grid(True, ls=":", alpha=0.45)
        ax.spines["polar"].set_color("#cccccc")
        ax.set_title(m, fontsize=10, color="#222", pad=22, fontweight="bold")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.21),
                  fontsize=7.5, frameon=False, ncol=2)

    fig.suptitle("ovagent_bench capability profile  |  baseline (dashed) vs +omicverse (solid)\n"
                 "ds4-pro LLM-judge scoring of agent trajectories, 0-100 per dimension",
                 fontsize=12, y=0.995, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    out = ANALYSIS / "ovagent_radar.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"saved {out}")

    # Console summary
    print("\n%-32s %-10s overall " % ("model", "mode") +
          "  ".join(f"{d[:8]:>8s}" for d in DIMS) + "    uplift")
    for m in models:
        for mode in ("baseline", "omicverse"):
            k = (m, mode)
            if k not in bucket:
                continue
            up = ""
            if mode == "omicverse" and (m, "baseline") in bucket:
                up = f"+{overall[k] - overall[(m,'baseline')]:5.1f}"
            print("%-32s %-10s %5.1f   " % (m, mode, overall[k]) +
                  "  ".join(f"{means[k][d] or 0:7.1f} " for d in DIMS) +
                  f"    {up}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("grade")
    g.add_argument("--systems", nargs="+", default=OMICVERSE_SYSTEMS)
    g.add_argument("--seeds", nargs="+", type=int, default=[0])
    g.add_argument("--model", default="deepseek-v4-pro")
    g.add_argument("-j", type=int, default=6)
    g.set_defaults(func=cmd_grade)

    pl = sub.add_parser("plot")
    pl.add_argument("--systems", nargs="+", default=OMICVERSE_SYSTEMS)
    pl.add_argument("--label-map", type=json.loads, default={})
    pl.set_defaults(func=cmd_plot)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
