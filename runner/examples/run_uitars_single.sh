#!/bin/bash
# 示例: 使用 UI-TARS 执行任务

python run.py \
  --provider uitars \
  --api-base http://localhost:9003/v1 \
  --model UI-TARS-1.5-7B \
  --max-steps 25 \
  --output-dir results \
  --draw \
  --task "在淘宝上搜索电动牙刷" 
