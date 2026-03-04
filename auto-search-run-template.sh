#!/usr/bin/env bash
set -euo pipefail

# Auto Search 运行参数模板
# 使用方式：
# 1) 修改下面变量
# 2) 执行: bash runner/mobiagent/auto-search-run-template.sh

APP_NAME="微博"
DEPTH=2
BREADTH=2
DEVICE="Harmony"                 # Android | Harmony
SERVICE_IP="166.111.53.96"
DECIDER_PORT=7003
EXPLORER_MODEL="qwen/qwen3-vl-235b-a22b-instruct" # qwen/qwen3.5-plus-02-15
OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
USE_QWEN3="on"                   # on | off
DATA_DIR=""                      # 为空时使用默认输出目录
ALLOW_HIERARCHY_TEXT_DECIDER="off" 
ENABLE_UI_SEMANTIC_COLLECT="on"   # on | off, 是否在auto-search过程中启用UI语义信息收集模块
                                  #提供给VLM更丰富的页面信息以辅助决策；开启后会增加一定的API调用和整体运行时间，请根据需要选择是否开启

UI_COLLECT_ASYNC="on"             # on | off
UI_COLLECT_QUEUE_SIZE=8
UI_COLLECT_DRAIN_ON_EXIT="on"     # on | off
UI_COLLECT_DRAIN_TIMEOUT_SEC=180
UI_COLLECT_USE_VLM="on"           # on | off
UI_COLLECT_VLM_TEXT_ONLY="off"    # on | off，当页面中存在歧义的元素或者需要更准确的对象描述时，开启；只使用VLM提供的文本，层级信息仍然来自页面结构
UI_COLLECT_VLM_MODEL="qwen/qwen3-vl-30b-a3b-instruct" # qwen/qwen3.5-35b-a3b
UI_COLLECT_BASE_URL="${OPENROUTER_BASE_URL}"  # 可单独指定UI采集模型提供商，例如本地vLLM: http://127.0.0.1:8001/v1
UI_COLLECT_API_KEY="${OPENROUTER_API_KEY}"    # 可单独指定UI采集key；本地vLLM通常可留空
UI_COLLECT_MAX_ITEMS=32
UI_COLLECT_MAX_VLM_CALLS=12
UI_COLLECT_MIN_AREA=16

# 建议通过环境变量注入，不要把密钥写死在仓库文件里
: "${OPENROUTER_API_KEY:?Please export OPENROUTER_API_KEY first}"

CMD=(
  python -m runner.mobiagent.auto-search
  --app_name "$APP_NAME"
  --depth "$DEPTH"
  --breadth "$BREADTH"
  --device "$DEVICE"
  --service_ip "$SERVICE_IP"
  --decider_port "$DECIDER_PORT"
  --openrouter_base_url "$OPENROUTER_BASE_URL"
  --openrouter_api_key "$OPENROUTER_API_KEY"
  --explorer_model "$EXPLORER_MODEL"
  --use_qwen3 "$USE_QWEN3"
  --allow_hierarchy_text_decider "$ALLOW_HIERARCHY_TEXT_DECIDER"
  --enable_ui_semantic_collect "$ENABLE_UI_SEMANTIC_COLLECT"
  --ui_collect_async "$UI_COLLECT_ASYNC"
  --ui_collect_queue_size "$UI_COLLECT_QUEUE_SIZE"
  --ui_collect_drain_on_exit "$UI_COLLECT_DRAIN_ON_EXIT"
  --ui_collect_drain_timeout_sec "$UI_COLLECT_DRAIN_TIMEOUT_SEC"
  --ui_collect_use_vlm "$UI_COLLECT_USE_VLM"
  --ui_collect_vlm_text_only "$UI_COLLECT_VLM_TEXT_ONLY"
  --ui_collect_vlm_model "$UI_COLLECT_VLM_MODEL"
  --ui_collect_base_url "$UI_COLLECT_BASE_URL"
  --ui_collect_api_key "$UI_COLLECT_API_KEY"
  --ui_collect_max_items "$UI_COLLECT_MAX_ITEMS"
  --ui_collect_max_vlm_calls "$UI_COLLECT_MAX_VLM_CALLS"
  --ui_collect_min_area "$UI_COLLECT_MIN_AREA"
)

if [[ -n "$DATA_DIR" ]]; then
  CMD+=(--data_dir "$DATA_DIR")
fi

echo "Running auto-search with app=$APP_NAME depth=$DEPTH breadth=$BREADTH ..."
"${CMD[@]}"
