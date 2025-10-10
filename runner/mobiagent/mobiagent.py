from openai import OpenAI
import uiautomator2 as u2
import base64
from PIL import Image
import json
import io
import logging
from abc import ABC, abstractmethod
import time
import re
import os
import argparse
from PIL import Image, ImageDraw, ImageFont
import textwrap
import cv2
import numpy as np
from utils.local_experience import PromptTemplateSearch 
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MAX_STEPS = 35

class Device(ABC):
    @abstractmethod
    def start_app(self, app):
        pass

    @abstractmethod
    def screenshot(self, path):
        pass

    @abstractmethod
    def click(self, x, y):
        pass

    @abstractmethod
    def input(self, text):
        pass

    @abstractmethod
    def swipe(self, direction):
        pass

    @abstractmethod
    def keyevent(self, key):
        pass

    @abstractmethod
    def dump_hierarchy(self):
        pass

class AndroidDevice(Device):
    def __init__(self, adb_endpoint=None):
        super().__init__()
        if adb_endpoint:
            self.d = u2.connect(adb_endpoint)
        else:
            self.d = u2.connect()
        self.app_package_names = {
            "携程": "ctrip.android.view",
            "同城": "com.tongcheng.android",
            "飞猪": "com.taobao.trip",
            "去哪儿": "com.Qunar",
            "华住会": "com.htinns",
            "饿了么": "me.ele",
            "支付宝": "com.eg.android.AlipayGphone",
            "淘宝": "com.taobao.taobao",
            "京东": "com.jingdong.app.mall",
            "美团": "com.sankuai.meituan",
            "滴滴出行": "com.sdu.didi.psnger",
            "微信": "com.tencent.mm",
            "微博": "com.sina.weibo",
            "携程": "ctrip.android.view",
        }

    def start_app(self, app):
        package_name = self.app_package_names.get(app)
        if not package_name:
            raise ValueError(f"App '{app}' is not registered with a package name.")
        self.d.app_start(package_name, stop=True)
        time.sleep(1)
        if not self.d.app_wait(package_name, timeout=10):
            raise RuntimeError(f"Failed to start app '{app}' with package '{package_name}'")
    
    def app_start(self, package_name):
        self.d.app_start(package_name, stop=True)
        time.sleep(1)
        if not self.d.app_wait(package_name, timeout=10):
            raise RuntimeError(f"Failed to start package '{package_name}'")
        
    def screenshot(self, path):
        self.d.screenshot(path)

    def click(self, x, y):
        self.d.click(x, y)

    def input(self, text):
        current_ime = self.d.current_ime()
        self.d.shell(['settings', 'put', 'secure', 'default_input_method', 'com.android.adbkeyboard/.AdbIME'])
        time.sleep(1)
        charsb64 = base64.b64encode(text.encode('utf-8')).decode('utf-8')
        self.d.shell(['am', 'broadcast', '-a', 'ADB_INPUT_B64', '--es', 'msg', charsb64])
        time.sleep(1)
        self.d.shell(['settings', 'put', 'secure', 'default_input_method', current_ime])
        time.sleep(1)

    def swipe(self, direction, scale=0.5):
        # self.d.swipe_ext(direction, scale)
        self.d.swipe_ext(direction=direction, scale=scale)

    def keyevent(self, key):
        self.d.keyevent(key)

    def dump_hierarchy(self):
        return self.d.dump_hierarchy()

decider_client = None
grounder_client = None
planner_client = None

planner_model = ""
decider_model = ""
grounder_model = ""
def init(service_ip, decider_port, grounder_port, planner_port):
    global decider_client, grounder_client, planner_client, general_client, general_model, apps
    decider_client = OpenAI(
        api_key = "0",
        base_url = f"http://{service_ip}:{decider_port}/v1",
    )
    grounder_client = OpenAI(
        api_key = "0",
        base_url = f"http://{service_ip}:{grounder_port}/v1",
    )
    planner_client = OpenAI(
        api_key = "0",
        base_url = f"http://{service_ip}:{planner_port}/v1",
    )

decider_prompt_template = """
You are a phone-use AI agent. Now your task is "{task}".
Your action history is:
{history}
Please provide the next action based on the screenshot and your action history. You should do careful reasoning before providing the action.
Your action space includes:
- Name: click, Parameters: target_element (a high-level description of the UI element to click).
- Name: swipe, Parameters: direction (one of UP, DOWN, LEFT, RIGHT).
- Name: input, Parameters: text (the text to input).
- Name: wait, Parameters: (no parameters, will wait for 1 second).
- Name: done, Parameters: (no parameters).
Your output should be a JSON object with the following format:
{{"reasoning": "Your reasoning here", "action": "The next action (one of click, input, swipe, done)", "parameters": {{"param1": "value1", ...}}}}"""

grounder_prompt_template_no_bbox = '''
Based on the screenshot, user's intent and the description of the target UI element, provide the coordinates of the element using **absolute coordinates**.
User's intent: {reasoning}
Target element's description: {description}
Your output should be a JSON object with the following format:
{{"coordinates": [x, y]}}'''

grounder_prompt_template_bbox = '''
Based on the screenshot, user's intent and the description of the target UI element, provide the bounding box of the element using **absolute coordinates**.
User's intent: {reasoning}
Target element's description: {description}
Your output should be a JSON object with the following format:
{{"bbox": [x1, y1, x2, y2]}}'''

decider_prompt_template_zh = """
你是一个手机使用AI代理。现在你的任务是“{task}”。
你的操作历史如下：
{history}
请根据截图和你的操作历史提供下一步操作。在提供操作之前，你需要进行仔细的推理。
你的操作范围包括：
- 名称：点击（click），参数：目标元素（target_element，对要点击的UI元素的高级描述）。
- 名称：滑动（swipe），参数：方向（direction，UP、DOWN、LEFT、RIGHT中的一个）。
- 名称：输入（input），参数：文本（text，要输入的文本）。
- 名称：等待（wait），参数：（无参数，将等待1秒）。
- 名称：完成（done），参数：（无参数）。
你的输出应该是一个如下格式的JSON对象：
{{"reasoning": "你的推理分析过程在此", "action": "下一步操作（click、input、swipe、done中的一个）", "parameters": {{"param1": "value1", ...}}}}"""

grounder_prompt_template_no_bbox_zh = """
根据截图、用户意图和目标UI元素的描述，使用**绝对坐标**提供该元素的坐标。
用户意图：{reasoning}
目标元素描述：{description}
你的输出应该是一个如下格式的JSON对象：
{{"coordinates": [x, y]}}"""

grounder_prompt_template_bbox_zh = """"
根据截图、用户意图和目标UI元素的描述，使用**绝对坐标**提供该元素的边界框。
用户意图：{reasoning}
目标元素描述：{description}
你的输出应该是一个如下格式的JSON对象：
{{"bbox": [x1, y1, x2, y2]}}"""

screenshot_path = "screenshot.jpg"
factor = 0.5

prices = {}


def get_screenshot(device):
    device.screenshot(screenshot_path)
    # resize the screenshot to reduce the size for processing
    img = Image.open(screenshot_path)
    img = img.resize((int(img.width * factor), int(img.height * factor)), Image.Resampling.LANCZOS)
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    screenshot = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return screenshot

def create_swipe_visualization(data_dir, image_index, direction):
    """为滑动动作创建可视化图像"""
    try:
        # 读取原始截图
        img_path = os.path.join(data_dir, f"{image_index}.jpg")
        if not os.path.exists(img_path):
            return
            
        img = cv2.imread(img_path)
        if img is None:
            return
            
        height, width = img.shape[:2]
        
        # 根据方向计算箭头起点和终点
        center_x, center_y = width // 2, height // 2
        arrow_length = min(width, height) // 4
        
        if direction == "up":
            start_point = (center_x, center_y + arrow_length // 2)
            end_point = (center_x, center_y - arrow_length // 2)
        elif direction == "down":
            start_point = (center_x, center_y - arrow_length // 2)
            end_point = (center_x, center_y + arrow_length // 2)
        elif direction == "left":
            start_point = (center_x + arrow_length // 2, center_y)
            end_point = (center_x - arrow_length // 2, center_y)
        elif direction == "right":
            start_point = (center_x - arrow_length // 2, center_y)
            end_point = (center_x + arrow_length // 2, center_y)
        else:
            return
            
        # 绘制箭头
        cv2.arrowedLine(img, start_point, end_point, (255, 0, 0), 8, tipLength=0.3)  # 蓝色箭头
        
        # 添加文字说明
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"SWIPE {direction.upper()}"
        text_size = cv2.getTextSize(text, font, 1.5, 3)[0]
        text_x = (width - text_size[0]) // 2
        text_y = 50
        cv2.putText(img, text, (text_x, text_y), font, 1.5, (255, 0, 0), 3)  # 蓝色文字
        
        # 保存可视化图像
        swipe_path = os.path.join(data_dir, f"{image_index}_swipe.jpg")
        cv2.imwrite(swipe_path, img)
        
    except Exception as e:
        logging.warning(f"Failed to create swipe visualization: {e}")


def task_in_app(app, old_task, task, device, data_dir, bbox_flag=True):
    history = []
    actions = []
    reacts = []
    while True:     
        if len(actions) >= MAX_STEPS:
            logging.info("Reached maximum steps, stopping the task.")
            break
        
        if len(history) == 0:
            history_str = "(No history)"
        else:
            history_str = "\n".join(f"{idx}. {h}" for idx, h in enumerate(history, 1))

        screenshot = get_screenshot(device)
        # for debug: 查看history
        # print("Current History:")
        # print(history_str)


        decider_prompt = decider_prompt_template.format(
            task=task,
            history=history_str
        )
        # logging.info(f"Decider prompt: \n{decider_prompt}")
        decider_response_str = decider_client.chat.completions.create(
            model="",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{screenshot}"}},
                        {"type": "text", "text": decider_prompt},
                    ]
                }
            ],
            temperature=0
        ).choices[0].message.content

        logging.info(f"Decider response: \n{decider_response_str}")

        # 预处理 decider_response_str，增强健壮性
        def robust_json_loads(s):
            import re
            s = s.strip()
            # 提取 ```json ... ``` 代码块
            codeblock = re.search(r"```json(.*?)```", s, re.DOTALL)
            if codeblock:
                s = codeblock.group(1).strip()
            # 替换中文省略号为英文 ...
            s = s.replace("…", "...")
            # 去除多余换行
            s = s.replace("\r", "").replace("\n", " ")
            # 尝试解析
            try:
                return json.loads(s)
            except Exception as e:
                logging.error(f"解析 decider_response_str 失败: {e}\n原始内容: {s}")
                raise

        decider_response = robust_json_loads(decider_response_str)
        converted_item = {
            "reasoning": decider_response["reasoning"],
            "function": {
                "name": decider_response["action"],
                "parameters": decider_response["parameters"]
            }
        }
        reacts.append(converted_item)
        action = decider_response["action"]

        # compute image index for this loop iteration (1-based)
        image_index = len(actions) + 1
        current_dir = os.getcwd()
        img_path = os.path.join(current_dir, f"screenshot.jpg")
        save_path = os.path.join(data_dir, f"{image_index}.jpg")
        img = Image.open(img_path)
        img.save(save_path)

        # attach index to the most recent react (reasoning)
        if reacts:
            try:
                reacts[-1]["action_index"] = image_index
            except Exception:
                pass

        hierarchy_path = os.path.join(data_dir, f"{image_index}.xml")
        hierarchy = device.dump_hierarchy()
        with open(hierarchy_path, "w", encoding="utf-8") as f:
            f.write(hierarchy)
        
        if action == "done":
            print("Task completed.")
            actions.append({
                "type": "done",
                "action_index": image_index
            })
            break
        if action == "click":
            reasoning = decider_response["reasoning"]
            target_element = decider_response["parameters"]["target_element"]
            grounder_prompt = (grounder_prompt_template_bbox if bbox_flag else grounder_prompt_template_no_bbox).format(reasoning=reasoning, description=target_element)
            # logging.info(f"Grounder prompt: \n{grounder_prompt}")
            
            grounder_response_str = grounder_client.chat.completions.create(
                model="",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{screenshot}"}},
                            {"type": "text", "text": grounder_prompt},
                        ]
                    }
                ],
                temperature=0
            ).choices[0].message.content
            logging.info(f"Grounder response: \n{grounder_response_str}")
            grounder_response = json.loads(grounder_response_str)
            if(bbox_flag):
                bbox = grounder_response["bbox"]

                x1, y1, x2, y2 = [int(coord / factor) for coord in bbox]
                position_x = (x1 + x2) // 2
                position_y = (y1 + y2) // 2
                device.click(position_x, position_y)
                # save action (record index only)
                actions.append({
                    "type": "click",
                    "position_x": position_x,
                    "position_y": position_y,
                    "bounds": [x1, y1, x2, y2],
                    "action_index": image_index
                })
                history.append(decider_response_str)

                current_dir = os.getcwd()
                img_path = os.path.join(current_dir, f"screenshot.jpg")
                save_path = os.path.join(data_dir, f"{image_index}_highlighted.jpg")
                img = Image.open(img_path)
                draw = ImageDraw.Draw(img)
                font = ImageFont.truetype("msyh.ttf", 40)
                text = f"CLICK [{position_x}, {position_y}]"
                text = textwrap.fill(text, width=20)
                text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:]
                draw.text((img.width / 2 - text_width / 2, 0), text, fill="red", font=font)
                img.save(save_path)

                # 拉框
                bounds_path = os.path.join(data_dir, f"{image_index}_bounds.jpg")
                img_bounds = Image.open(save_path)
                draw_bounds = ImageDraw.Draw(img_bounds)
                draw_bounds.rectangle([x1, y1, x2, y2], outline='red', width=5)
                img_bounds.save(bounds_path)

                # 画点
                cv2image = cv2.imread(bounds_path)
                if cv2image is not None:
                    # 在点击位置画圆点
                    cv2.circle(cv2image, (position_x, position_y), 15, (0, 255, 0), -1)  # 绿色实心圆
                    # 保存带点击点的图像
                    click_point_path = os.path.join(data_dir, f"{image_index}_click_point.jpg")
                    cv2.imwrite(click_point_path, cv2image)

            else:
                coordinates = grounder_response["coordinates"]
                x, y = [int(coord / factor) for coord in coordinates]
                device.click(x, y)
                actions.append({
                    "type": "click",
                    "position_x": x,
                    "position_y": y,
                    "action_index": image_index
                })
          

        elif action == "input":
            text = decider_response["parameters"]["text"]
            device.input(text)
            actions.append({
                "type": "input",
                "text": text,
                "action_index": image_index
            })
            history.append(decider_response_str)

        elif action == "swipe":
            direction = decider_response["parameters"]["direction"]

            if direction == "DOWN":
                device.swipe(direction.lower(), 2)
                time.sleep(1)
                # record the swipe as an action (index only)
                actions.append({
                    "type": "swipe",
                    "press_position_x": None,
                    "press_position_y": None,
                    "release_position_x": None,
                    "release_position_y": None,
                    "direction": direction.lower(),
                    "action_index": image_index
                })
                history.append(decider_response_str)
                
                # 为向下滑动创建可视化
                create_swipe_visualization(data_dir, image_index, direction.lower())
                continue

            if direction in ["UP", "LEFT", "RIGHT"]:
                device.swipe(direction.lower())
                actions.append({
                    "type": "swipe",
                    "press_position_x": None,
                    "press_position_y": None,
                    "release_position_x": None,
                    "release_position_y": None,
                    "direction": direction.lower(),
                    "action_index": image_index
                })
                history.append(decider_response_str)
                
                # 为滑动创建可视化
                create_swipe_visualization(data_dir, image_index, direction.lower())

            else:
                raise ValueError(f"Unknown swipe direction: {direction}")
        elif action == "wait":
            print("Waiting for a while...")
            actions.append({
                "type": "wait",
                "action_index": image_index
            })
        else:
            raise ValueError(f"Unknown action: {action}")
        
        time.sleep(1)
    
    data = {
        "app_name": app,
        "task_type": None,
        "old_task_description": old_task,
        "task_description": task,
        "action_count": len(actions),
        "actions": actions
    }

    with open(os.path.join(data_dir, "actions.json"), "w", encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    with open(os.path.join(data_dir, "react.json"), "w", encoding='utf-8') as f:
        json.dump(reacts, f, ensure_ascii=False, indent=4)

from utils.load_md_prompt import load_prompt
planner_prompt_template = load_prompt("planner_oneshot.md")

current_file_path = Path(__file__).resolve()
current_dir = current_file_path.parent
default_template_path = current_dir.parent.parent / "utils" /"experience" / "templates-new.json"

def parse_planner_response(response_str: str):

    # 尝试匹配 ```json ... ``` 代码块
    pattern = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)
    match = pattern.search(response_str)

    json_str = None
    if match:
        json_str = match.group(1)
    else:
        # 如果没有代码块，直接当成 JSON
        json_str = response_str.strip()

    try:
        data = json.loads(json_str)
        return data
    except json.JSONDecodeError as e:
        logging.error(f"解析 JSON 失败: {e}\n内容为:\n{json_str}")
        return None

def get_app_package_name(task_description):
    """单阶段：本地检索经验，调用模型完成应用选择和任务描述生成。"""
    # 本地检索经验
    search_engine = PromptTemplateSearch()
    print("Using template path:", default_template_path)
    experience_content = search_engine.get_experience(task_description, default_template_path, 1)
    print(f"检索到的相关经验:\n{experience_content}")

    # 构建Prompt
    prompt = planner_prompt_template.format(
        task_description=task_description,
        experience_content=experience_content
    )
    
    # 调用模型
    # while True:
    response_str = planner_client.chat.completions.create(
        model = planner_model,
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]
    ).choices[0].message.content
    logging.info(f"Planner 响应: \n{response_str}")
    
    response_json = parse_planner_response(response_str)
    if response_json is None:
        raise ValueError("无法解析模型响应为 JSON。")

    app_name = response_json.get("app_name")
    package_name = response_json.get("package_name")
    final_desc = response_json.get("final_task_description", task_description)
    
    return app_name, package_name, final_desc

# for testing purposes
if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="MobiMind Agent")
    parser.add_argument("--service_ip", type=str, default="localhost", help="Ip for the services (default: localhost)")
    parser.add_argument("--decider_port", type=int, default=8000, help="Port for decider service (default: 8000)")
    parser.add_argument("--grounder_port", type=int, default=8001, help="Port for grounder service (default: 8001)")
    parser.add_argument("--planner_port", type=int, default=8002, help="Port for planner service (default: 8002)")
    
    args = parser.parse_args()

    # 使用命令行参数初始化
    init(args.service_ip, args.decider_port, args.grounder_port, args.planner_port)

    device = AndroidDevice()
    print(f"connect to device")

    data_base_dir = os.path.join(os.path.dirname(__file__), 'data')
    if not os.path.exists(data_base_dir):
        os.makedirs(data_base_dir)

    # 读取任务列表
    task_json_path = os.path.join(os.path.dirname(__file__), "task.json")
    with open(task_json_path, "r", encoding="utf-8") as f:
        task_list = json.load(f)
    
    # print(task_list)

    for task in task_list:
        existing_dirs = [d for d in os.listdir(data_base_dir) if os.path.isdir(os.path.join(data_base_dir, d)) and d.isdigit()]
        if existing_dirs:
            data_index = max(int(d) for d in existing_dirs) + 1
        else:
            data_index = 1
        data_dir = os.path.join(data_base_dir, str(data_index))
        os.makedirs(data_dir)

        task_description = task
        app_name, package_name, new_task_description = get_app_package_name(task_description)
        device.app_start(package_name)
        print(f"Starting task '{new_task_description}' in app '{app_name}'")
        task_in_app(app_name, task_description, new_task_description, device, data_dir, True)