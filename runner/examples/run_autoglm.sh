#!/bin/bash
# 示例: 使用 AutoGLM 执行任务

python run.py \
  --provider autoglm \
  --api-base http://localhost:9003/v1 \
  --model autoglm-phone-9b \
  --max-steps 20 \
  --output-dir results \
  --draw \
  --task "打开小红书搜索博主影视飓风并查看他的主页第一条内容" \
  --device-type Harmony
