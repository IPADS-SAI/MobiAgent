#!/bin/bash
# 示例3: 使用UI-TARS执行任务

python run.py \
  --provider uitars \
  --task "在淘宝上搜索电动牙刷" \
  --model-url http://localhost:8000/v1 \
  --model-name UI-TARS-7B-SFT \
  --max-steps 25 \
  --output-dir results
