#!/usr/bin/env bash
set -euo pipefail

# Auto Search 运行参数模板
# 使用方式：
# 1) 修改下面变量
# 2) 执行: bash runner/mobiagent/auto-search-run-template.sh

APP_NAME="小红书"
DEPTH=2
BREADTH=2
DEVICE="Harmony"                 # Android | Harmony
SERVICE_IP="166.111.53.96"
DECIDER_PORT=7003
EXPLORER_MODEL="google/gemini-3-flash-preview"
OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
USE_QWEN3="on"                   # on | off
DATA_DIR=""                      # 为空时使用默认输出目录

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
)

if [[ -n "$DATA_DIR" ]]; then
  CMD+=(--data_dir "$DATA_DIR")
fi

echo "Running auto-search with app=$APP_NAME depth=$DEPTH breadth=$BREADTH ..."
"${CMD[@]}"
