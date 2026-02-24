#!/usr/bin/env bash
set -euo pipefail

# UI semantic boxer run template
# 1) Edit defaults below as needed
# 2) Run: bash collect/auto/run-ui-semantic-boxer.sh

DEVICE="Harmony"                         # Android | Harmony
ADB_ENDPOINT=""                           # optional, Android only
OUTPUT_DIR=""                             # empty -> auto timestamped folder
USE_VLM="on"                              # on | off
VLM_MODEL="qwen/qwen3-vl-30b-a3b-instruct"            # default: qwen3-vl 30A3
BASE_URL="https://openrouter.ai/api/v1"   # OpenRouter by default
MAX_VLM_CALLS=12
MAX_ITEMS=20
MIN_AREA=16

# Prefer API key from env; required when USE_VLM=on
: "${OPENROUTER_API_KEY:?Please export OPENROUTER_API_KEY first}"

CMD=(
  python -m collect.auto.ui_semantic_boxer
  --device "$DEVICE"
  --use_vlm "$USE_VLM"
  --vlm_model "$VLM_MODEL"
  --base_url "$BASE_URL"
  --api_key "$OPENROUTER_API_KEY"
  --max_vlm_calls "$MAX_VLM_CALLS"
  --max_items "$MAX_ITEMS"
  --min_area "$MIN_AREA"
)

if [[ -n "$ADB_ENDPOINT" ]]; then
  CMD+=(--adb_endpoint "$ADB_ENDPOINT")
fi

if [[ -n "$OUTPUT_DIR" ]]; then
  CMD+=(--output_dir "$OUTPUT_DIR")
fi

echo "Running UI semantic boxer on $DEVICE ..."
"${CMD[@]}"
