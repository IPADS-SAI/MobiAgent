import os
import json
import re
from openai import OpenAI
import sys
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 添加 MobiAgent 根目录到 sys.path
current_dir = Path(__file__).resolve().parent
mobigent_root = current_dir.parent
sys.path.append(str(mobigent_root))

from utils.local_experience import PromptTemplateSearch
from utils.load_md_prompt import load_prompt

# --- 配置 ---
SERVICE_IP = "localhost"
PLANNER_PORT = 8002  # 默认端口, 如有需要请调整
# DATA_TASK_DIR = current_dir / "data_task"
DATA_TASK_DIR = current_dir.parent / "Dataset/Chinese-Mobile-Use/sft/normal/feizhu"  # 请根据实际路径调整
PROMPT_FILE = current_dir / "prompts" / "rewrite_prompt.md"
# 使用与 mobiagent.py 中相同的 planner
PLANNER_API_KEY = "sk-rfCIGhxrzcdsMV4jC17e406bE56c47CbA5416068A62318D3"
PLANNER_BASE_URL = "http://ipads.chat.gpt:3006/v1"
MODEL = "gemini-2.5-flash"

# --- 初始化 Planner 客户端 ---
try:
    planner_client = OpenAI(
        api_key=PLANNER_API_KEY,
        base_url=PLANNER_BASE_URL,
    )
except Exception as e:
    print(f"初始化 planner 客户端时出错: {e}")
    sys.exit(1)

# --- 加载提示词模版 ---
try:
    rewrite_prompt_template = load_prompt(PROMPT_FILE)
    # 手动转义示例JSON中的花括号，防止format方法报错
    rewrite_prompt_template = rewrite_prompt_template.replace('{\n      "rewritten_task_description"', '{{\n      "rewritten_task_description"')
    rewrite_prompt_template = rewrite_prompt_template.replace('"\n    }', '"\n    }}')
    rewrite_prompt_template = rewrite_prompt_template.replace('{\n  "rewritten_task_description"', '{{\n  "rewritten_task_description"')
    rewrite_prompt_template = rewrite_prompt_template.replace('"\n}', '"\n}}')
except Exception as e:
    print(f"加载提示文件 {PROMPT_FILE} 时出错: {e}")
    sys.exit(1)


def rewrite_task_description(task_description: str, app:str) -> str:
    """
    使用 planner 和模版重写任务描述。
    """
    # 1. 检索本地经验
    try:
        search_engine = PromptTemplateSearch()
        experience_content = search_engine.get_experience(task_description)
        print(f"为 '{task_description}' 检索到的经验:\n{experience_content}")
    except Exception as e:
        print(f"无法检索经验: {e}")
        experience_content = "未找到相关经验。"

    # 2. 构建提示
    prompt = rewrite_prompt_template.format(
        app_name= app,
        task_description=task_description,
        experience_content=experience_content
    )

    # 3. 调用 planner 模型
    while True:
        try:
            response = planner_client.chat.completions.create(
                model= MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            response_content = response.choices[0].message.content
            
            # 从响应中提取 JSON
            match = re.search(r'```json\n(.*?)\n```', response_content, re.DOTALL)
            if match:
                response_json_str = match.group(1)
                response_json = json.loads(response_json_str)
                rewritten_desc = response_json.get("rewritten_task_description")
                if rewritten_desc:
                    return rewritten_desc
            
            print("无法从模型响应中解析重写的描述。正在重试...")

        except Exception as e:
            print(f"调用 planner 时发生错误: {e}。正在重试...")


def process_data_tasks():
    """
    递归遍历 data_task 目录下所有子目录, 重写第一个任务描述, 并将其追加到列表中。
    """
    if not DATA_TASK_DIR.exists():
        print(f"错误: 目录 '{DATA_TASK_DIR}' 未找到。")
        return

    for root, dirs, files in os.walk(DATA_TASK_DIR):
        for file in files:
            if file == "actions.json":
                actions_file = Path(root) / file
                print(f"\n--- 正在处理 {actions_file} ---")
                try:
                    with open(actions_file, 'r+', encoding='utf-8') as f:
                        data = json.load(f)
                        
                        task_description = data.get("task_description")
                        app = data.get("app_name", "任务默认App")
                        
                        if not task_description:
                            print(f"跳过 {actions_file}: 'task_description' 为空或不存在。")
                            continue

                        # 确保 task_description 是一个列表
                        if isinstance(task_description, str):
                            task_descriptions = [task_description]
                        else:
                            task_descriptions = task_description

                        if len(task_descriptions) >= 11:
                            print(f"跳过 {actions_file}: 已改写，任务数已达上限（11）。")
                            continue
                        
                        # 采用第2个任务进行重写
                        first_task = task_descriptions[1] if len(task_descriptions) > 1 else task_descriptions[0]
                        
                        print(f"原始任务: {first_task}")
                        
                        # 重写任务
                        rewritten_task = rewrite_task_description(first_task, app)
                        
                        if rewritten_task and rewritten_task not in task_descriptions:
                            print(f"重写后的任务: {rewritten_task}")
                            task_descriptions.append(rewritten_task)
                            data["task_description"] = task_descriptions
                            
                            # 写回文件
                            f.seek(0)
                            f.truncate()
                            json.dump(data, f, ensure_ascii=False, indent=4)
                            print(f"成功更新 {actions_file}")
                        else:
                            print("跳过更新: 重写的任务为空或已存在。")

                except (json.JSONDecodeError, IOError) as e:
                    print(f"处理 {actions_file} 时出错: {e}")


if __name__ == "__main__":
    process_data_tasks()
