#!/bin/bash

python run.py \
    --provider qwen \
    --output-dir results \
    --qwen-api-base http://localhost:8080/v1 \
    --max-steps 20 \
    --draw \
    --task "帮我在微博浏览查看央视新闻今天发了什么，查看前2条微博"