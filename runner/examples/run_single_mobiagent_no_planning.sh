#!/bin/bash
# 示例: 使用MobiAgent执行任务（不启用planning，需手动在任务描述中指定APP）

python run.py \
  --provider mobiagent \
  --api-base http://localhost:9002/v1 \
  --max-steps 30 \
  --output-dir results \
  --task "在淘宝上搜索电动牙刷，选最畅销的那款"