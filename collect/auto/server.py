import uiautomator2 as u2
import time
import os
import shutil
import base64
from PIL import Image
import io
import json
import re
import logging
import sys
import json
from datetime import datetime
from openai import OpenAI
import argparse

from collect.auto.draw_bounds import process_folder

device = None  # 设备连接对象
hierarchy = None  # 层次结构数据
data_index = 1  # 数据索引

operation_history = []  # 操作历史记录
logger = None  # 日志记录器

# 全局配置变量，将由命令行参数设置
model = None
api_key = None
base_url = None
max_steps = 15
client = None

# action_dir 是存储的目录
def get_current_hierarchy_and_screenshot(action_dir, sleep_time = 0):
    global hierarchy
    time.sleep(sleep_time)

    if os.path.exists(action_dir):
        shutil.rmtree(action_dir)
    os.makedirs(action_dir)

    # if not os.path.exists(action_dir):
    #     os.makedirs(action_dir)

    screenshot_path = os.path.join(action_dir, "screenshot.jpg")
    hierarchy_path = os.path.join(action_dir, "hierarchy.xml")

    device.screenshot(screenshot_path)
    hierarchy = device.dump_hierarchy()
    with open(hierarchy_path, "w", encoding="utf-8") as f:
        f.write(hierarchy)

    logger.info(f"操作完成，已重新截图和获取层次结构")

# 将路径 img_path 截图保存为base64编码的字符串
def get_screenshot(img_path, factor=0.4):
    img = Image.open(img_path)
    img = img.resize((int(img.width * factor), int(img.height * factor)), Image.Resampling.LANCZOS)
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    screenshot = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return screenshot

def handle_click(x, y):
    """处理点击操作"""
    device.click(x, y)
    operation_record = {
        "type": "click",
        "timestamp": __import__('datetime').datetime.now().isoformat(),
        "position": {"x": x, "y": y},
        # "clicked_elements": clicked_elements
    }
    operation_history.append(operation_record)

def handle_input(text):
    current_ime = device.current_ime()
    device.shell(['settings', 'put', 'secure', 'default_input_method', 'com.android.adbkeyboard/.AdbIME'])
    time.sleep(1)
    charsb64 = base64.b64encode(text.encode('utf-8')).decode('utf-8')
    device.shell(['am', 'broadcast', '-a', 'ADB_INPUT_B64', '--es', 'msg', charsb64])
    time.sleep(1)
    device.shell(['settings', 'put', 'secure', 'default_input_method', current_ime])
    # time.sleep(1)
    # device.shell(['am', 'broadcast', '-a', 'ADB_INPUT_METHOD_HIDE'])

    operation_record = {
        "type": "input",
        "timestamp": __import__('datetime').datetime.now().isoformat(),
        "text": text,
    }
    operation_history.append(operation_record)
    # get_current_hierarchy_and_screenshot(1.5)

def handle_swipe(direction):
    # device.swipe(action.startX, action.startY, action.endX, action.endY, duration=0.1)
    device.swipe_ext(direction=direction, duration=0.1)
    operation_record = {
        "type": "swipe",
        "timestamp": __import__('datetime').datetime.now().isoformat(),
        "direction": direction
    }
    operation_history.append(operation_record)
    # get_current_hierarchy_and_screenshot(1.5)



from utils.load_md_prompt import load_prompt
app_selection_prompt_template = load_prompt("planner.md")
decider_prompt_template = load_prompt("auto_decider.md")

def get_app_package_name(task_description):
    """根据任务描述获取需要启动的app包名"""
    app_selection_prompt = app_selection_prompt_template.format(task_description=task_description)
    while True:
        response_str = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": app_selection_prompt
                }
            ]
        ).choices[0].message.content
        
        logger.info(f"应用选择响应: \n{response_str}")
        
        # 解析JSON响应
        pattern = re.compile(r"```json\n(.*)\n```", re.DOTALL)
        match = pattern.search(response_str)
        if match:
            break
        
    response = json.loads(match.group(1))
    package_name = response.get("package_name")
    reasoning = response.get("reasoning")
        
    logger.info(f"选择应用原因: {reasoning}")
    logger.info(f"选择的包名: {package_name}")
        
    return package_name

def do_task(task_description, data_dir):
    global logger
    
    logger.info(f"开始执行任务: {task_description}")
    
    # 根据任务描述获取需要启动的应用包名
    package_name = get_app_package_name(task_description)
    logger.info(f"选择启动应用: {package_name}")

    device.app_start(package_name, stop=True)
    time.sleep(3) 
    action_history = []
    reasoning_history = []
    screenshots = []
    while True:
        logger.info('=' * 50)
        logger.info('=' * 50)
        action_count = len(action_history)  # 已有的操作数量
        action_index = action_count + 1     # 接下来的操作索引
        action_dir = os.path.join(data_dir, str(action_index))
        get_current_hierarchy_and_screenshot(action_dir)

        if(action_count > max_steps):
            logger.info(f"任务步骤超过上限({max_steps})，停止执行")
            return

        if action_count == 0:
            history = "(No history)"
        else:
            # history = "\n".join(f"{idx}. {h}" for idx, h in enumerate(action_history, 1))
            history = "\n".join(f"{idx}. {h}" for idx, h in enumerate(reasoning_history, 1))

        # 截图拉框
        # layer_count, bounds_list = process_folder(action_dir, need_clickable=True)
        layer_count, bounds_list = process_folder(action_dir)
        logger.info(f"已处理 {action_dir}，共绘制 {layer_count} 个图层")

        # decider_prompt
        decider_prompt = decider_prompt_template.format(
            task_description = task_description,
            history = history,
            layer_count = layer_count
        )
        message_content = [
            {"type": "text", "text": decider_prompt}
        ]

        # 屏幕截图
        screenshot_path = os.path.join(action_dir, "screenshot.jpg")
        screenshot = get_screenshot(screenshot_path, factor=1.0)
        message_content.append({
            "type": "text",
            "text": f"\n屏幕截图:"
        })
        message_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{screenshot}"}
        })

        # 遍历所有标注图层
        for idx in range(1, layer_count + 1):
            screenshot_path = os.path.join(action_dir, f"layer_{idx}.jpg")
            screenshot = get_screenshot(screenshot_path)
            message_content.append({
                "type": "text", 
                "text": f"\n第{idx}张标注图层:"
            })
            message_content.append({
                "type": "image_url", 
                "image_url": {"url": f"data:image/jpeg;base64,{screenshot}"}
            })

        decider_response_str = client.chat.completions.create(
            model= model,
            messages=[
                {
                    "role": "user",
                    "content": message_content
                }
            ]
        ).choices[0].message.content
        
        logger.info(f"response: \n{decider_response_str}")
        pattern = re.compile(r"```json\n(.*)\n```", re.DOTALL)
        match = pattern.search(decider_response_str)
        if not match:
            logger.error("错误：未找到有效的JSON响应")
            continue
        decider_response = json.loads(match.group(1))

        reasoning = decider_response.get("reasoning")
        action = decider_response.get("action")
        parameters = decider_response.get("parameters")

        if action == "done":
            logger.info("任务完成！")
            action = {
                "reasoning": reasoning,
                "function": {
                    "name": "done",
                    "parameters": {}
                }
            }
            logger.info(f"完成操作: {action}")
            action_history.append(action)
            reasoning_history.append(reasoning)
            break

        elif action == "click":
            target_element = parameters.get("target_element")
            index = parameters.get("index")
            if index is None or index < 0 or index >= len(bounds_list):
                logger.error(f"错误：index {index} 超出范围，有效范围为 0 到 {len(bounds_list)-1}")
                continue
            bounds = bounds_list[index]
            # index, bounds = decide_click_element(data_dir, action_count + 1, task_description, reasoning, target_element)
            logger.info(f"选择点击元素: {target_element} (index: {index}, bounds: {bounds})")
            x = (bounds[0] + bounds[2]) / 2
            y = (bounds[1] + bounds[3]) / 2
            handle_click(x, y)
            action = {
                "reasoning": reasoning,
                 "function": {
                    "name": "click",
                    "parameters": {
                        "position_x": x,
                        "position_y": y,
                        "bounding_box": bounds,
                        "target_element": target_element,
                    }
                }
            }
            logger.info(f"点击操作: {action}")
            action_history.append(action)
            reasoning_history.append(reasoning)

        elif action == "input":
            text = parameters.get("text")
            handle_input(text)
            action = {
                "reasoning": reasoning,
                "function": {
                    "name": "input",
                    "parameters": {
                        "text": text
                    }
                }
            }
            logger.info(f"输入操作: {action}")
            action_history.append(action)
            reasoning_history.append(reasoning)

        elif action == "swipe":
            direction = parameters.get("direction").lower()
            handle_swipe(direction)
            action = {
                "reasoning": reasoning,
                "function": {
                    "name": "swipe",
                    "parameters": {
                        "direction": direction
                    }
                }
            }
            logger.info(f"滑动操作: {action}")
            action_history.append(action)
            reasoning_history.append(reasoning)
        else:
            raise ValueError(f"Unknown action: {action}")

        time.sleep(2.5)
    
    data = {
        "task_description": task_description,
        "action_count": len(action_history),
        "actions": action_history,
    }
    data_file = os.path.join(data_dir, "task_data.json")
    with open(data_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    logger.info(f"任务执行完成，共执行 {len(action_history)} 个操作")
    logger.info(f"任务数据已保存到: {data_file}")
    logger.info("日志记录完成")

def setup_logger(data_dir):
    """设置日志记录器，同时输出到控制台和文件"""
    global logger
    
    # 创建日志目录
    log_file = os.path.join(data_dir, "execution.log")
    
    # 创建logger，使用特定名称避免冲突
    logger_name = f'auto_collect_{id(data_dir)}'
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    # 清除已有的处理器
    logger.handlers.clear()
    
    # 防止日志传播到根logger
    logger.propagate = False
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 创建格式器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def change_auto_data(data_log_path, index):
    parse_error = os.path.join(data_log_path, "parse.error")
    if os.path.exists(parse_error):
        return
    task_data = os.path.join(data_log_path, "task_data.json")
    if not os.path.exists(task_data):
        return

    with open(task_data, 'r', encoding='utf-8') as file:
        task_data = json.load(file)

    app_name = task_data.get("app_name")
    task_type = None
    task_description = task_data.get("task_description")
    actions = task_data.get("actions")

    new_actions = []
    for action in actions:
        action_type = action["function"]["name"].lower()
        if action_type == "click":
            new_action = {
                "type": action_type,
                "position_x": int(action["function"]["parameters"]["position_x"]),
                "position_y": int(action["function"]["parameters"]["position_y"]),
                "bounds": action["function"]["parameters"]["bounding_box"]
            }
            new_actions.append(new_action)
        elif action_type == "swipe":
            new_action = {
                "type": action_type,
                "press_position_x": None,
                "press_position_y": None,
                "release_position_x": None,
                "release_position_y": None,
                "direction": action["function"]["parameters"]["direction"]
            }
            new_actions.append(new_action)
        elif action_type == "input":
            new_action = {
                "type": action_type,
                "text": action["function"]["parameters"]["text"]
            }
            new_actions.append(new_action)
        elif action_type == "done":
            new_action = {
                "type": "done"
            }
            new_actions.append(new_action)
        else:
            raise ValueError(f"Unknown action type: {action_type}")

    data = {
        "app_name": app_name,
        "task_type": task_type,
        "task_description": task_description,
        "action_count": len(new_actions),
        "actions": new_actions
    }

    dest_path_dir = os.path.join(os.path.dirname(__file__), 'data')
    if not os.path.exists(dest_path_dir):
        os.makedirs(dest_path_dir)
    existing_dirs = [d for d in os.listdir(dest_path_dir) if os.path.isdir(os.path.join(dest_path_dir, d)) and d.isdigit()]
    if existing_dirs:
        max_index = max(int(d) for d in existing_dirs) + 1
    else:
        max_index = 1
    dest_path = os.path.join(dest_path_dir, str(max_index))
    os.makedirs(dest_path)
    
    # 复制并重命名图片文件
    for index in range(1, len(new_actions) + 2):  # +2 因为通常有一张额外的截图
        screenshot_src = os.path.join(data_log_path, str(index), "screenshot.jpg")
        if os.path.exists(screenshot_src):
            screenshot_dest = os.path.join(dest_path, f"{index}.jpg")
            shutil.copy2(screenshot_src, screenshot_dest)
            print(f"复制图片: {screenshot_src} -> {screenshot_dest}")
        
    with open(os.path.join(dest_path, "actions.json"), 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Auto collection of GUI data')
    parser.add_argument('--model', type=str, required=True, help='name of the LLM model')
    parser.add_argument('--api_key', type=str, required=True, help='API key for the LLM model')
    parser.add_argument('--base_url', type=str, required=True, help='base URL for the LLM model API')
    parser.add_argument('--max_steps', type=int, default=15, help='maximum steps per task (default: 15)')

    args = parser.parse_args()
    
    # 设置全局配置
    model = args.model
    api_key = args.api_key
    base_url = args.base_url
    max_steps = args.max_steps
    
    # 初始化OpenAI客户端
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    
    device = u2.connect()
    # 创建数据目录
    session_base_dir = os.path.dirname(__file__)
    data_base_dir = os.path.join(session_base_dir, 'data_log')
    if not os.path.exists(data_base_dir):
        os.makedirs(data_base_dir)

    # 读取任务列表
    task_json_path = os.path.join(os.path.dirname(__file__), "task.json")
    with open(task_json_path, "r", encoding="utf-8") as f:
        task_list = json.load(f)

    for task_description in task_list:
        # 遍历现有数据目录，找到最大的索引
        existing_dirs = [d for d in os.listdir(data_base_dir) if os.path.isdir(os.path.join(data_base_dir, d)) and d.isdigit()]
        if existing_dirs:
            data_index = max(int(d) for d in existing_dirs) + 1
        else:
            data_index = 1
        data_log_dir = os.path.join(data_base_dir, str(data_index))
        os.makedirs(data_log_dir)

        # 设置日志记录器
        logger = setup_logger(data_log_dir)
        logger.info("程序启动")
        logger.info(f"数据索引: {data_index}")
        logger.info(f"数据目录: {data_log_dir}")

        do_task(task_description, data_log_dir)
        change_auto_data(data_log_dir, data_index)

