#!/usr/bin/env bash
# Run a benchmark sweep from a YAML config.
#
# Usage:
#   bash scripts/run.sh configs/smoke_a_layer.yaml
#   bash scripts/run.sh configs/full_v1.yaml
#   bash scripts/run.sh configs/multiseed_full_v1.yaml
#
# After the sweep finishes, grade with:
#   bash scripts/grade.sh <run_name>

set -uo pipefail
cd "$(dirname "$0")/.."
source ./bench-env.sh

OVPY=<CONDA_PREFIX>/bin/python
CFG="${1:?usage: $0 <path/to/config.yaml>}"

if [[ ! -f "${CFG}" ]]; then
    echo "ERROR: config not found: ${CFG}" >&2
    exit 1
fi

# Require ollama if endpoint is local.
if grep -q "127.0.0.1:11434" "${CFG}"; then
    if ! curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
        echo "ERROR: ollama not at :11434 — run scripts/ollama_up.sh first" >&2
        exit 1
    fi
fi

RUN_NAME=$(grep -E '^run_name:' "${CFG}" | head -1 | awk '{print $2}' | tr -d '"')
TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/${RUN_NAME}"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/runner_${TS}.stdout"

echo "[$(date '+%H:%M:%S')] start config=${CFG} run=${RUN_NAME}" | tee -a "${LOG}"
"${OVPY}" -m bench.runner --config "${CFG}" 2>&1 | tee -a "${LOG}"
echo "[$(date '+%H:%M:%S')] done; grade with: bash scripts/grade.sh ${RUN_NAME}" | tee -a "${LOG}"
