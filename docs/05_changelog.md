# Changelog

## v1.0 (2026-05-22) — public release

First public release of the OmicVerse-OmicBench harness + results. Companion to the [omicverse/OmicBench](https://huggingface.co/datasets/omicverse/OmicBench) dataset on Hugging Face.

### Scope

- **38 of 44 OmicBench tasks** evaluated (Layer D — D01_multiome_joint_embedding, D02_peak_gene_linking — excluded; see "Known gaps").
- 7 LLMs × 2 system-prompt arms (`*_baseline` vs `*_omicverse`, the Beacon system prompt).
- Seeds {0,1,2} for 6 LLMs; qwen3.6:35b-a3b at seed 0 only (local ollama, 1-seed budget).
- Two ablation arms: `no_registry` (drop discovery tools) and `doc_rag` (replace structured registry with vanilla embedding RAG).

### Headline

| metric | result |
|---|---|
| Pass@1 baseline (7-LLM mean) | 66.9% |
| Pass@1 +Beacon (7-LLM mean) | 84.7% |
| Δ uplift                     | +17.8 pp |
| Largest single uplift        | qwen3.6:35b-a3b, +34.2 pp |

Per-dimension breakdown (`analysis/radar_native.py`): the entire uplift concentrates on **tool_grounding** (+48.6 pp average). See `analysis/README.md` for the six-dimension methodology.

### Grader

40 check types in `bench/grader.py`, all biology-grounded:
- Marker-list overlap (PanglaoDB-derived PBMC + Bastidas-Ponce 2019 pancreas)
- ARI vs ground-truth cell types
- Cell-type silhouette / spatial silhouette
- scIB-style biology preservation
- Multi-method dispatch + agreement (cosine for velocity, Jaccard for DA)
- Root anchoring (Ductal pseudotime / Ductal-out velocity)
- LR-pair reference (CellChatDB v2 + cellphonedb v5)
- 16S diversity (alpha + beta)

### Statistics

10 000-sample bootstrap CI for Pass@1; paired McNemar at (task, seed) granularity; BH-FDR(q=0.10) across pairwise contrasts; 7-class failure taxonomy from trace + grader output.

### Known gaps  (to fix in v1.1)

| Gap | Why it exists in v1.0 |
|---|---|
| D01_multiome_joint_embedding, D02_peak_gene_linking | Layer D was added to OmicBench after this sweep; needs an extra environment (`snapATAC2`, `muon`) that wasn't part of `bench-env.template.sh`. Tasks exist in HF; harness supports them; the sweep just didn't include them. |
| qwen3.6:35b-a3b on seeds 1 & 2 | Local ollama time budget. The other 6 LLMs ran 3 seeds. |
| Some layer-shard gpt_* runs not merged into `gpt_full_canonical` | For gpt-5.5, B07/B12/C03/E02 had data in layer shards but weren't included in the canonical merge. Will be re-merged in v1.1 alongside the D-layer fill-in. |

## v0 — pre-release iterations (not public)

Internal pilots (v1 pilot, v2, v3, v3.1, v3.2) — superseded; task definitions, fixtures, and graders were rewritten between iterations. Not part of any public release.
