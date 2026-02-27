#!/usr/bin/env bash
set -euo pipefail

# UI semantic boxer run template
# 1) Edit defaults below as needed
# 2) Run: bash collect/auto/run-ui-semantic-boxer.sh

DEVICE="Harmony"                         # Android | Harmony
ADB_ENDPOINT=""                           # optional, Android only
APP_NAME="小红书"                          # App name for actions.json
OUTPUT_DIR=""                             # empty -> auto timestamped folder
USE_VLM="on"                              # on | off
VLM_MODEL="qwen/qwen3-vl-30b-a3b-instruct"            # default: qwen3-vl 30A3
BASE_URL="https://openrouter.ai/api/v1"   # OpenRouter by default
MAX_VLM_CALLS=12
MAX_ITEMS=20
MIN_AREA=16
ENABLE_KIND_VLM="on"                      # on | off, check ui_kind with page-level VLM
KIND_VLM_MODE="page_once"                 # page_once
KIND_VLM_MAX_RETRY=2
TASK_DESC_WITH_KIND="on"                  # on | off

OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}"
if [[ "$USE_VLM" == "on" && -z "$OPENROUTER_API_KEY" ]]; then
  echo "Please export OPENROUTER_API_KEY first"
  exit 1
fi

CMD=(
  python -m collect.auto.ui_semantic_boxer
  --device "$DEVICE"
  --app_name "$APP_NAME"
  --use_vlm "$USE_VLM"
  --vlm_model "$VLM_MODEL"
  --base_url "$BASE_URL"
  --api_key "$OPENROUTER_API_KEY"
  --max_vlm_calls "$MAX_VLM_CALLS"
  --max_items "$MAX_ITEMS"
  --min_area "$MIN_AREA"
  --enable_kind_vlm "$ENABLE_KIND_VLM"
  --kind_vlm_mode "$KIND_VLM_MODE"
  --kind_vlm_max_retry "$KIND_VLM_MAX_RETRY"
  --task_desc_with_kind "$TASK_DESC_WITH_KIND"
)

if [[ -n "$ADB_ENDPOINT" ]]; then
  CMD+=(--adb_endpoint "$ADB_ENDPOINT")
fi

if [[ -n "$OUTPUT_DIR" ]]; then
  CMD+=(--output_dir "$OUTPUT_DIR")
fi

echo "Running UI semantic boxer on $DEVICE ..."
"${CMD[@]}"
