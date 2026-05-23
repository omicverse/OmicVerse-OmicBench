#!/usr/bin/env bash
# Grade a finished sweep + write results/<run_name>/{grades.csv, summary.md}.
#
# Usage:
#   bash scripts/grade.sh <run_name>           # grades trajectories/<run_name>/
#   bash scripts/grade.sh trajectories/<run>   # explicit dir also accepted

set -euo pipefail
cd "$(dirname "$0")/.."
source ./bench-env.sh

OVPY=<CONDA_PREFIX>/bin/python
ARG="${1:?usage: $0 <run_name | trajectories/run_name>}"

if [[ -d "trajectories/${ARG}" ]]; then
    TRAJ="trajectories/${ARG}"
    RUN_NAME="${ARG}"
elif [[ -d "${ARG}" ]]; then
    TRAJ="${ARG}"
    RUN_NAME="$(basename "${ARG}")"
else
    echo "ERROR: no such trajectory dir: trajectories/${ARG} or ${ARG}" >&2
    exit 1
fi

echo "[grade] traj-dir=${TRAJ} run=${RUN_NAME}"
"${OVPY}" -m bench.grade_run --traj-dir "${TRAJ}" --run-name "${RUN_NAME}"

SUMMARY="results/${RUN_NAME}/summary.md"
if [[ -f "${SUMMARY}" ]]; then
    echo
    echo "================  ${RUN_NAME}  SUMMARY  ================"
    head -40 "${SUMMARY}"
fi
