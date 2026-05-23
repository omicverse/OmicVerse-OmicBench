# Run configurations

Each YAML file here defines **one reproducible benchmark run**.
`bash scripts/run.sh configs/<name>.yaml` materialises the cross-product
of `scope.systems × scope.task_ids × scope.seeds` and writes one
trajectory JSON per cell into `trajectories/<run_name>/`.

Grading is a separate postprocessing step:

```bash
bash scripts/grade.sh <run_name>
# -> results/<run_name>/grades.csv  +  summary.md
```

## Available configs

All 17 configs ship pre-tuned for the v1.0 run set. Pick the one matching
the LLM you have an API key for, then `bash scripts/run.sh configs/<...>.yaml`.

### Headline single-LLM sweeps (full 38-task suite × seeds {0,1,2})

| Config | LLM | Endpoint | Wallclock |
|---|---|---|---|
| `gpt_full.yaml`             | gpt-5.5            | ChatGPT OAuth (Codex CLI tokens) | ~6 hr |
| `deepseek_v4_pro_full.yaml`   | deepseek-v4-pro    | https://api.deepseek.com/v1 | ~3 hr |
| `deepseek_full.yaml`          | deepseek-v4-flash  | https://api.deepseek.com/v1 | ~2 hr |
| `gemini_full.yaml`            | gemini-3.1-flash-lite-preview | Gemini OpenAI-compatible | ~3 hr |
| `glm_full.yaml`               | GLM-5.1            | BigModel coding-plan endpoint | ~3 hr |
| `minimax_full.yaml`           | MiniMax-M2.7       | OpenAI-compatible | ~3 hr |
| `full_v1.yaml`                | qwen3.6:35b-a3b    | local ollama                | ~6 hr (1 seed only) |
| `multiseed_full_v1.yaml`      | qwen3.6:35b-a3b    | local ollama                | ~18 hr (3 seeds; not in v1.0 results) |

> The v1.0 gpt-5.5 sweep was actually executed as six layer-sharded sub-runs
> (`gpt_abc / gpt_c / gpt_e / gpt_f / gpt_g / gpt_l`) to fit
> OAuth concurrency limits and to allow a mid-sweep C01 prompt patch.
> `gpt_full.yaml` is the equivalent single-config merge — if you hit
> "401 refresh" errors on long sweeps, copy it to per-layer files with
> `scope.layers` narrowed to one or two layers and run sequentially.

### Ablation arms (1 seed each)

| Config | LLM | What changes vs full OmicVerse |
|---|---|---|
| `ablation_gpt_no_registry.yaml`   | gpt-5.5 | omicverse system prompt with the registry-lookup section stripped — tests the contribution of structured discovery |
| `ablation_gpt_doc_rag.yaml`       | gpt-5.5 | replaces structured registry with vanilla embedding RAG over docstrings — tests whether "more docs in context" is enough |
| `ablation_v4flash_no_registry.yaml` | deepseek-v4-flash | same ablation on v4-flash for cross-model robustness |
| `ablation_v4flash_doc_rag.yaml`     | deepseek-v4-flash | same |

## Schema

| Field | Description |
|---|---|
| `run_name` | Output namespace. All artefacts shard under this name. |
| `description` | Free-text — shown in the run summary, useful as a record of intent. |
| `llm.model` | Model id passed verbatim to every adapter. |
| `llm.endpoint` | OpenAI-compatible base URL. |
| `llm.api_key_env` | Name of the environment variable holding the API key. |
| `llm.temperature` | Sampling temperature (the runner adds a small per-seed jitter). |
| `agent.max_agent_turns` | Cap on agent loop iterations (per-task `max_turns` overrides this if smaller). |
| `scope.systems` | LLM-specific arm names from `bench/adapters/__init__.py` (e.g. `gpt_omicverse` / `gpt_baseline`, `deepseek_omicverse` / `deepseek_baseline`, …). Each pair shares the same loop, tools, and model — only the system prompt differs. |
| `scope.task_ids` | Explicit task ids; if empty, falls back to `layers`. |
| `scope.layers` | Layer filter (`A`, `B`, `C`, `E`, `F`, `G`, `L`); used only if `task_ids` is empty. |
| `scope.seeds` | One or more integer seeds. |
| `scope.skip_completed` | If `true`, skip trajectories that already exist on disk. |
| `paths.{trajectories,results,workspace,logs}_dir` | Per-namespace output roots. |

## Layer D is not currently in any config

The Layer D tasks (D01_multiome_joint_embedding, D02_peak_gene_linking)
exist in [OmicBench on HF](https://huggingface.co/datasets/omicverse/OmicBench)
but require an extra environment (`snapATAC2`, `muon`) that
`bench-env.template.sh` does not yet provision. They are scheduled for
v1.1. To run them ahead of the v1.1 release, install those deps, then
add `D` to `scope.layers` in any sweep config.
