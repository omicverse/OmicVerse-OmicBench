#!/bin/bash
# Regenerate report.md + 3 PNG figures from results/runs.parquet.
set -uo pipefail
cd "$(dirname "$0")/.."
source ./bench-env.sh
OVPY=<CONDA_PREFIX>/bin/python

"$OVPY" -m bench.report
"$OVPY" -m bench.figures
echo "wrote results/report.md + figures"
