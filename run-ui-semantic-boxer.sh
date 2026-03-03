#!/usr/bin/env bash
set -euo pipefail

# UI semantic boxer run template
# 1) Edit defaults below as needed
# 2) Run: bash collect/auto/run-ui-semantic-boxer.sh

# Device and app
DEVICE="Harmony"                          # Android | Harmony
ADB_ENDPOINT=""                           # optional, Android only
APP_NAME="微博"                           # App name written to actions.json
OUTPUT_DIR=""                             # empty => auto timestamp folder

# VLM and request budget
USE_VLM="on"                              # on | off
VLM_TEXT_ONLY="off"                        # on => hierarchy only provides boxes, text always from VLM，off，
                                           #当页面中存在歧义的元素或者需要更准确的对象描述时，开启；只使用VLM提供的文本，层级信息仍然来自页面结构
VLM_MODEL="qwen/qwen3-vl-30b-a3b-instruct"
BASE_URL="https://openrouter.ai/api/v1"
MAX_VLM_CALLS=12
MAX_ITEMS=32
MIN_AREA=16

# Page-level kind review
ENABLE_KIND_VLM="on"                      # on | off
KIND_VLM_MODE="page_once"                 # currently only page_once is supported
KIND_VLM_MAX_RETRY=2
TASK_DESC_WITH_KIND="on"                  # on | off

# Duplicate label disambiguation (group-level VLM)
ENABLE_DUPLICATE_DESC_VLM="on"            # on | off
MAX_DUPLICATE_DESC_VLM_CALLS=8

# Final page-level ambiguity review for task_description
ENABLE_TASK_DESC_VLM_REVIEW="on"          # on | off
TASK_DESC_VLM_REVIEW_MAX_RETRY=2

# Refine generic VLM text labels for better in-page localization
ENABLE_VLM_TEXT_REFINE="on"               # on | off
MAX_VLM_TEXT_REFINE_CALLS=12

OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}"
if [[ "$USE_VLM" == "on" && -z "$OPENROUTER_API_KEY" ]]; then
  echo "Please export OPENROUTER_API_KEY first"
  exit 1
fi

CMD=(
  # Pipeline: capture screen/hierarchy -> rule classify -> VLM verify -> export actions/react
  python -m collect.auto.ui_semantic_boxer
  --device "$DEVICE"
  --app_name "$APP_NAME"
  --use_vlm "$USE_VLM"
  --vlm_text_only "$VLM_TEXT_ONLY"
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
  --enable_duplicate_desc_vlm "$ENABLE_DUPLICATE_DESC_VLM"
  --max_duplicate_desc_vlm_calls "$MAX_DUPLICATE_DESC_VLM_CALLS"
  --enable_task_desc_vlm_review "$ENABLE_TASK_DESC_VLM_REVIEW"
  --task_desc_vlm_review_max_retry "$TASK_DESC_VLM_REVIEW_MAX_RETRY"
  --enable_vlm_text_refine "$ENABLE_VLM_TEXT_REFINE"
  --max_vlm_text_refine_calls "$MAX_VLM_TEXT_REFINE_CALLS"
)

if [[ -n "$ADB_ENDPOINT" ]]; then
  CMD+=(--adb_endpoint "$ADB_ENDPOINT")
fi

if [[ -n "$OUTPUT_DIR" ]]; then
  CMD+=(--output_dir "$OUTPUT_DIR")
fi

echo "Running UI semantic boxer on $DEVICE ..."
"${CMD[@]}"
