#!/usr/bin/env bash
# Fetch the OmicBench task fixtures + rubrics from Hugging Face.
#
# Pulls into ${OMICBENCH_DATA:-data/OmicBench}. Idempotent — re-running
# resumes / verifies. Requires `huggingface_hub` (installed via pyproject)
# and an HF token with `read` access (set HUGGING_FACE_HUB_TOKEN before
# sourcing bench-env.sh, or run `huggingface-cli login` once).
#
# OmicBench is gated on HF; you may need to accept the dataset terms on
# https://huggingface.co/datasets/omicverse/OmicBench first.

set -euo pipefail

DEST="${OMICBENCH_DATA:-${OMICVERSE_OMICBENCH_ROOT:-data/OmicBench}}"
REPO="omicverse/OmicBench"

mkdir -p "$(dirname "$DEST")"

echo "[fetch_omicbench] downloading $REPO -> $DEST"

if command -v hf >/dev/null 2>&1; then
    hf download --repo-type dataset "$REPO" --local-dir "$DEST"
elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download --repo-type dataset "$REPO" --local-dir "$DEST"
else
    python -m huggingface_hub.commands.huggingface_cli \
        download --repo-type dataset "$REPO" --local-dir "$DEST"
fi

echo "[fetch_omicbench] done. $(du -sh "$DEST" | awk '{print $1}') under $DEST"
echo "[fetch_omicbench] tasks:"
ls "$DEST" | grep -E '^omicbench-' | wc -l
