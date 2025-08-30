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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MAX_STEPS = 30

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
        self.d.set_input_ime

    def dump_hierarchy(self):
        return self.d.dump_hierarchy()

decider_client = None
grounder_client = None
planner_client = None

def init(service_ip, decider_port, grounder_port, planner_port):
    global decider_client, grounder_client, planner_client, general_client, general_model, apps
    decider_client = OpenAI(
        api_key = "0",
        base_url = f"http://{service_ip}:{decider_port}/v1",
    )
    grounder_client = OpenAI(
        api_key = "0",
        base_url = f"{service_ip}:{grounder_port}/v1",
    )
    planner_client = OpenAI(
        api_key = "0",
        base_url = f"{service_ip}:{planner_port}/v1",
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


screenshot_path = "screenshot.jpg"
factor = 0.5

prices = {}

app_scale = {
    "去哪儿": 1.0,
    "飞猪": 0.7,
    "华住会": 1.0,
    "携程": 0.9,
    "同城": 1.0,
}

def get_screenshot(device):
    device.screenshot(screenshot_path)
    # resize the screenshot to reduce the size for processing
    img = Image.open(screenshot_path)
    img = img.resize((int(img.width * factor), int(img.height * factor)), Image.Resampling.LANCZOS)
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    screenshot = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return screenshot

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

        decider_prompt = decider_prompt_template.format(
            task=task,
            history=history_str
        )
        # logging.info(f"Decider prompt: \n{decider_prompt}")
        decider_response_str = decider_client.chat.completions.create(
            model="decider",
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

        decider_response = json.loads(decider_response_str)
        converted_item = {
            "reasoning": decider_response["reasoning"],
            "function": {
                "name": decider_response["action"],
                "parameters": decider_response["parameters"]
            }
        }
        reacts.append(converted_item)
        action = decider_response["action"]

        current_dir = os.getcwd()
        img_path = os.path.join(current_dir, f"screenshot.jpg")
        save_path = os.path.join(data_dir, f"{len(actions) + 1}.jpg")
        img = Image.open(img_path)
        img.save(save_path)

        hierarchy_path = os.path.join(data_dir, f"{len(actions) + 1}.xml")
        hierarchy = device.dump_hierarchy()
        with open(hierarchy_path, "w", encoding="utf-8") as f:
            f.write(hierarchy)
        
        if action == "done":
            print("Task completed.")
            actions.append({
                "type": "done"
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
                actions.append({
                    "type": "click",
                    "position_x": position_x,
                    "position_y": position_y,
                    "bounds": [
                        x1, y1, x2, y2
                    ]
                })
                history.append(decider_response_str)

                current_dir = os.getcwd()
                img_path = os.path.join(current_dir, f"screenshot.jpg")
                save_path = os.path.join(data_dir, f"{len(actions)}_highlighted.jpg")
                img = Image.open(img_path)
                draw = ImageDraw.Draw(img)
                font = ImageFont.truetype("msyh.ttf", 40)
                text = f"CLICK [{position_x}, {position_y}]"
                text = textwrap.fill(text, width=20)
                text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:]
                draw.text((img.width / 2 - text_width / 2, 0), text, fill="red", font=font)
                img.save(save_path)

                # 拉框
                bounds_path = os.path.join(data_dir, f"{len(actions)}_bounds.jpg")
                img_bounds = Image.open(save_path)
                draw_bounds = ImageDraw.Draw(img_bounds)
                draw_bounds.rectangle([x1, y1, x2, y2], outline='red', width=5)
                img_bounds.save(bounds_path)

                # # 画点
                # with open(save_path, 'rb') as f:
                #     image_data = f.read()
                # nparr = np.frombuffer(image_data, np.uint8)
                # cv2image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                # if action["type"] == "click":
                #     x = int(action['position_x'])
                #     y = int(action['position_y'])
                #     cv2.circle(cv2image, (x, y), 50, (0, 0, 255), 10)
                # elif action["type"] == "swipe":
                #     x1 = int(action['press_position_x'])
                #     y1 = int(action['press_position_y'])
                #     x2 = int(action['release_position_x'])
                #     y2 = int(action['release_position_y'])
                #     cv2.arrowedLine(cv2image, (x1, y1), (x2, y2), (0, 0, 255), 5)
                # success, encoded_img = cv2.imencode('.jpg', cv2image)

            else:
                coordinates = grounder_response["coordinates"]
                x, y = [int(coord / factor) for coord in coordinates]
                device.click(x, y)
          

        elif action == "input":
            text = decider_response["parameters"]["text"]
            device.input(text)
            actions.append({
                "type": "input",
                "text": text
            })
            history.append(decider_response_str)

        elif action == "swipe":
            direction = decider_response["parameters"]["direction"]

            if direction == "DOWN":
                device.swipe(direction.lower(), 2)
                time.sleep(2)
                continue

            if direction in ["UP", "DOWN", "LEFT", "RIGHT"]:
                device.swipe(direction.lower())
                actions.append({
                    "type": "swipe",
                    "press_position_x": None,
                    "press_position_y": None,
                    "release_position_x": None,
                    "release_position_y": None,
                    "direction": direction.lower()
                })
                history.append(decider_response_str)

            else:
                raise ValueError(f"Unknown swipe direction: {direction}")
        elif action == "wait":
            print("Waiting for a while...")
            actions.append({
                "type": "wait"
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
app_selection_prompt_template = load_prompt("planner.md")

def get_app_package_name(task_description):
    """根据任务描述获取需要启动的app包名和改写后的任务描述"""
    app_selection_prompt = app_selection_prompt_template.format(task_description=task_description)
    while True:
        response_str = planner_client.chat.completions.create(
            model="planner",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": app_selection_prompt},
                    ]
                }
            ]
        ).choices[0].message.content

        logging.info(f"应用选择响应: \n{response_str}")
        
        pattern = re.compile(r"```json\n(.*)\n```", re.DOTALL)
        match = pattern.search(response_str)
        if match:
            break
        
    response = json.loads(match.group(1))
    app_name = response.get("app_name")
    package_name = response.get("package_name")
    new_task_description = response.get("task_description", task_description)  # 如果没有新描述，使用原描述
        
    return app_name, package_name, new_task_description

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