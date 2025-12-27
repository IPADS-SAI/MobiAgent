#!/bin/bash
# 示例4: 使用UI-TARS批量执行MobiFlow任务

python run.py \
  --provider uitars \
  --task-file mobiagent/task_mobiflow.json \
  --model-url http://localhost:8000/v1 \
  --model-name UI-TARS-7B-SFT \
  --max-steps 25 \
  --output-dir results/uitars_batch \
  --log-level INFO
