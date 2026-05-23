# Source this before any bench command:  source bench-env.template.sh
#
# This is a TEMPLATE. Fill in the API keys and paths for your environment
# before sourcing. All paths assume the supplementary tree is unpacked at
# OVBENCH_ROOT.

# === Required: project root ===
export OVBENCH_ROOT="${OVBENCH_ROOT:-$(pwd)}"
export OVBENCH_DATA="${OVBENCH_ROOT}/data"

# === Conda environment running omicverse ===
# Point CONDA_PREFIX at your omicverse-enabled environment so the
# notebook executor picks the right Python.
export CONDA_DEFAULT_ENV="${CONDA_DEFAULT_ENV:-omicdev}"
export CONDA_PREFIX="${CONDA_PREFIX:-/path/to/your/conda/env}"

# === Cache redirects (keeps everything under OVBENCH_DATA) ===
export HF_HOME="${OVBENCH_DATA}/hf_cache"
export HUGGINGFACE_HUB_CACHE="${OVBENCH_DATA}/hf_cache"
export TRANSFORMERS_CACHE="${OVBENCH_DATA}/hf_cache"
export HF_DATASETS_CACHE="${OVBENCH_DATA}/hf_cache/datasets"
export PIP_CACHE_DIR="${OVBENCH_DATA}/pip_cache"
export XDG_CACHE_HOME="${OVBENCH_DATA}/xdg_cache"
export TRITON_CACHE_DIR="${OVBENCH_DATA}/xdg_cache/triton"
export TORCH_HOME="${OVBENCH_DATA}/xdg_cache/torch"
export TORCHINDUCTOR_CACHE_DIR="${OVBENCH_DATA}/xdg_cache/torchinductor"

# === ov.Agent harness store ===
export OVAGENT_HOME="${OVBENCH_DATA}/ovagent_home"

# === LLM API keys — fill in before running ===

# DeepSeek (used by deepseek_full and deepseek_v4_pro_full)
export DEEPSEEK_API_KEY="<YOUR_DEEPSEEK_KEY>"
export DEEPSEEK_API_BASE="https://api.deepseek.com/v1"

# Gemini (Google's OpenAI-compatible endpoint)
export GEMINI_API_KEY="<YOUR_GEMINI_KEY>"
export GEMINI_API_BASE="https://generativelanguage.googleapis.com/v1beta/openai/"

# Zhipu GLM-5.1 coding-plan endpoint
export GLM_API_KEY="<YOUR_GLM_KEY>"
export GLM_API_BASE="https://open.bigmodel.cn/api/coding/paas/v4"

# MiniMax M-series
export MINIMAX_API_KEY="<YOUR_MINIMAX_KEY>"
export MINIMAX_API_BASE="https://api.minimaxi.com/v1"

# OpenRouter (only needed for the qwen-via-openrouter side experiment)
export OPENROUTER_API_KEY="<YOUR_OPENROUTER_KEY>"
export OPENROUTER_API_BASE="https://openrouter.ai/api/v1"

# gpt-5.5 (ChatGPT OAuth) uses ChatGPT OAuth — log in once with `codex login`,
# auth lands at ~/.codex/auth.json. No env var needed.

# === Foundation-model checkpoints (for L02/L03/L04 tasks) ===
# Place scGPT / Geneformer / SCimilarity / scFoundation weights here.
export OV_FM_CHECKPOINT_DIR="<PATH_TO_FM_CHECKPOINTS>"
# Optional per-model override when layout differs from <base>/<spec>/:
# export OV_FM_CHECKPOINT_DIR_GENEFORMER="<PATH_TO_GENEFORMER>"

# === Bench-internal paths ===
export OVBENCH_RESULTS="${OVBENCH_DATA}/results"
export OVBENCH_FIXTURES="${OVBENCH_ROOT}/fixtures"

# === mini-swe-agent: redirect global config out of $HOME ===
export MSWEA_GLOBAL_CONFIG_DIR="${OVBENCH_DATA}/mini_swe_config"
export MSWEA_COST_TRACKING="ignore_errors"
mkdir -p "${MSWEA_GLOBAL_CONFIG_DIR}"

# === CellxGene census useragent (used by some L-layer tasks) ===
export CELLXGENE_CENSUS_USERAGENT="ovagent-bench/0.1"

# Confirm
if [[ "${1:-}" == "--show" ]]; then
    echo "OVBENCH_ROOT       = ${OVBENCH_ROOT}"
    echo "OVBENCH_DATA       = ${OVBENCH_DATA}"
    echo "CONDA_PREFIX       = ${CONDA_PREFIX}"
    echo "OVAGENT_HOME       = ${OVAGENT_HOME}"
fi
