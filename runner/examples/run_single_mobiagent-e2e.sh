#!/bin/bash
# 示例1: 使用MobiAgent执行单个任务

python run.py \
  --provider mobiagent \
  --service-ip localhost \
  --decider-port 9002 \
  --grounder-port 9002 \
  --planner-port 8080 \
  --max-steps 30 \
  --output-dir results \
  --enable-planning \
  --use-e2e \
  --task "在淘宝上搜索电动牙刷，选最畅销的那款" \
  --device-type Harmony