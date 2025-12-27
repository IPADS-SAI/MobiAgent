#!/bin/bash
# 示例1: 使用MobiAgent执行单个任务

python run.py \
  --provider mobiagent_step \
  --task "在淘宝上搜索电动牙刷，选最畅销的那款并加入购物车" \
  --service-ip localhost \
  --decider-port 9002 \
  --grounder-port 9002 \
  --planner-port 8080 \
  --enable-planning \
  --max-steps 30 \
  --draw \
  --output-dir results
