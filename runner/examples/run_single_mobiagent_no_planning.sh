#!/bin/bash
# 示例: 使用MobiAgent执行任务（不启用planning，需手动在任务描述中指定APP）

python run.py \
  --provider mobiagent_step \
  --task "在淘宝上搜索电动牙刷，选最畅销的那款" \
  --service-ip localhost \
  --decider-port 9002 \
  --grounder-port 9002 \
  --max-steps 30 \
  --output-dir results
