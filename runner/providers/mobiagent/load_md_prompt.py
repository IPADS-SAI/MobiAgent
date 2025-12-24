# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

import os

def load_prompt(md_name, prompt_dir=None):
    """从markdown文件加载prompt模板
    
    Args:
        md_name: markdown文件名
        prompt_dir: prompt目录路径（可选，默认为当前文件的prompts子目录）
    """
    if prompt_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_dir = os.path.join(current_dir, "prompts")
    
    prompt_file = os.path.join(prompt_dir, md_name)

    with open(prompt_file, "r", encoding="utf-8") as f:
        content = f.read()
    content = content.replace("````markdown", "").replace("````", "")
    return content.strip()
