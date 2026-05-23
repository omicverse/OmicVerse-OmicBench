# analysis/ — extended capability profiling

The headline OmicBench numbers (Pass@1 per model × arm) come from
`bench/grader.py` and live in `results/<run>/grades.csv`. This directory
adds an extended layer: **per-dimension capability profiling** that
attributes the omicverse-arm uplift to specific skill axes rather than a
single overall score.

Two scripts live here, with different methodologies. The native-dim
variant is the one we recommend; the LLM-judge variant is kept for
cross-framework comparison.

## `radar_native.py` — native 6-dim radar  (recommended)

Groups OmicBench's 140+ rubric check items into six dimensions derived
**from the rubric itself**, then plots per-(model, mode, dimension)
pass-rate as a small-multiples radar.

The six dimensions, with the rationale for each:

| # | Dimension | What it tests |
|---|---|---|
| 1 | object_plumbing | Artifact placed in the correct AnnData slot (~62 checks) |
| 2 | data_integrity | Stored values in correct scale / layer / dtype (~8 checks) |
| 3 | tool_grounding | Real algorithm invoked, not a stub or fabricated output (~10 checks) |
| 4 | methodological_robustness | Multi-method comparison and inter-method agreement (~11 checks) |
| 5 | quantitative_quality | Computed metric or count meets a threshold (~34 checks) |
| 6 | biological_plausibility | Result respects known biology — markers, lineage direction, structure (~15 checks) |

The full key→dimension mapping is in `native_dim_map.csv`; edit it and
re-run to refine.

**Why these six.** OmicBench's rubric is procedural (boolean check
functions), unlike BiomniBench-DA's narrative rubric. Each dimension
above corresponds to a coherent cluster of rubric checks that share a
testing mechanism. Three of them (`object_plumbing`, `data_integrity`,
`methodological_robustness`) saturate near 90%+ for most models and
arms — they index baseline competence, not the omicverse-specific
contribution. The remaining three (`tool_grounding`,
`quantitative_quality`, `biological_plausibility`) are where the
ablation manifests.

**Headline finding.** Across all 7 models tested, the omicverse arm's
uplift is concentrated almost entirely in `tool_grounding`:

| Dimension | Baseline (7-model mean) | +omicverse | Δ |
|---|---:|---:|---:|
| **tool_grounding** | **32.3%** | **80.9%** | **+48.6 pp** |
| quantitative_quality | 89.7% | 96.2% | +6.5 pp |
| biological_plausibility | 86.0% | 91.4% | +5.4 pp |
| methodological_robustness | 92.9% | 97.3% | +4.4 pp |
| object_plumbing | 94.0% | 96.1% | +2.1 pp |
| data_integrity | 96.0% | 97.3% | +1.3 pp |

Reading: baseline models know that algorithms like `cnmf`,
`wgcna`, `ccc`, foundation-model inference, etc. exist, but
mini-swe-agent struggles to assemble the correct call from a bare
bash shell (`*_real_call` checks pass 20-50% of the time). The
omicverse arm provides typed wrappers and skill registry entries
that close most of that gap. The other five dimensions start near
ceiling, so the radar mostly looks like a unit hex with one corner
extended.

### Reproduce

```bash
python analysis/radar_native.py
# -> analysis/ovagent_radar_native.png
# -> analysis/native_dim_map.csv (audit; edit to refine the mapping)
```

Optional restriction to seed=0 for parity with single-seed runs:

```bash
python analysis/radar_native.py --seeds 0
```

## `radar_grade.py` — LLM-judge variant  (legacy, for cross-framework comparison)

Sends each agent trajectory to DeepSeek-v4-pro as a judge, asking for
0-100 scores on the **BiomniBench-DA** six dimensions:
`data_handling`, `method_selection`, `statistical_rigor`,
`biological_interpretation`, `scientific_reasoning`,
`source_reliability`.

Two reasons this is a poor primary framework for OmicBench (and a
cautionary tale about cross-benchmark methodology transfer):

1. `biological_interpretation` and `scientific_reasoning` are
   floor-effect dominated. OmicBench tasks don't ask the agent to
   write narrative interpretation — the harness inspects AnnData
   artifacts. mini-swe-agent's "reasoning + one bash command" format
   amplifies the floor effect. Most models score 30-50 here regardless
   of arm.
2. `source_reliability` is ceiling-effect dominated. Bash commands
   execute against real APIs by construction, so the judge gives
   nearly everyone 80-95.

That means 3 of 6 dimensions are weak measurements in this context,
and the overall uplifts (+0 to +12) understate the real signal that
shows up in pass% (+17pp average).

Kept here so you can reproduce the cross-framework comparison and see
the failure mode yourself. The cached `radar_grades.jsonl` (548 cells)
lets `plot` work without re-grading.

### Reproduce

```bash
# Re-grade (requires DEEPSEEK_API_KEY + the trajectories on disk):
source bench-env.sh
python analysis/radar_grade.py grade --seeds 0 -j 8
# -> appends to analysis/radar_grades.jsonl  (~$10 in API spend)

# Plot from cache only (works without trajectories):
python analysis/radar_grade.py plot
# -> analysis/ovagent_radar.png
```

## Files

| File | What |
|---|---|
| `radar_native.py` | Native-dim radar generator |
| `radar_grade.py` | LLM-judge variant generator |
| `native_dim_map.csv` | The 140 rubric keys with their dimension assignments — audit & edit |
| `radar_grades.jsonl` | Cached LLM-judge scores for 548 (system, model, task, seed=0) cells |
| `ovagent_radar_native.png` | Headline radar (per-model mini-radars, gray=baseline / green=+omicverse) |
| `ovagent_radar.png` | LLM-judge radar (kept for reference) |
