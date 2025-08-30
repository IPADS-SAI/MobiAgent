import os

def load_prompt(md_name):
    """从markdown文件加载应用选择prompt模板"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_file = os.path.join(current_dir, "..", "prompts", md_name)

    with open(prompt_file, "r", encoding="utf-8") as f:
        content = f.read()
    content = content.replace("````markdown", "").replace("````", "")
    return content.strip()