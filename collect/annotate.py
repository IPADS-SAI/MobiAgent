from PIL import Image, ImageDraw, ImageFont
import textwrap
import cv2
import numpy as np
import os

import argparse
from argparse import Namespace

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import base64, re
import json

from utils.parse_omni import extract_all_bounds, find_clicked_element

from utils.load_md_prompt import load_prompt

model = None

direction_mapping = {
    "向上滑动": "UP",
    "向下滑动": "DOWN",
    "向左滑动": "LEFT",
    "向右滑动": "RIGHT",
    "从下往上滑动": "UP",
    "从上往下滑动": "DOWN",
    "从右往左滑动": "LEFT",
    "从左往右滑动": "RIGHT",
    "向上滚动": "UP",
    "向下滚动": "DOWN",
    "向左滚动": "LEFT",
    "向右滚动": "RIGHT",
    "从下往上滚动": "UP",
    "从上往下滚动": "DOWN",
    "从右往左滚动": "LEFT",
    "从左往右滚动": "RIGHT",
}

# 前者是actions.json 后者是react.json 对应的解析内容
def compare_actions(actions, reacts):
    if (len(actions) != len(reacts)):
        raise Exception(f"[Action and React length mismatch] actions: {len(actions)}, reacts: {len(reacts)}")
    
    for i, (action, react) in enumerate(zip(actions, reacts)):  
        # 比较动作类型（忽略大小写）
        action_type = action.get("type", "").lower()
        react_type = react.get("function").get("name", "").lower()

        if action_type != react_type:
            raise Exception(f"[type mismatch] Action {i+1}: action type {action_type}，react type {react_type}")

        reasoning = react["reasoning"]

        # 展示放弃如reasoning中有滚动滑动，强制让类型变成swipe
        # for desc, expected_direction in direction_mapping.items():
        #     if desc in reasoning:y
        #         if react_type != "swipe":
        #             raise Exception(f"[Swipe action is expected] action {i+1} action: {action}, react: {react}, reasoning: {reasoning}")
        #         break
        
        if(action_type == "swipe"):
            action_direction = action["direction"].upper() if "direction" in action else None

        if(react_type == "swipe"):
            # parameters 内的字段可能不是 direction，而是taget啥的
            if "parameters" not in react["function"] or "direction" not in react["function"]["parameters"]:
                raise Exception(f"[Swipe action missing parameters] React {i+1}: {react}")
            
            react_direction = react["function"]["parameters"]["direction"]

            if(action_direction != react_direction):
                raise Exception(f"[direction mismatch] Action {i+1}: action_direction: {action_direction}, react_direction: {react_direction}")

            flag = False
            for desc, expected_direction in direction_mapping.items():
                if desc in reasoning:
                    if react_direction == expected_direction:
                        flag = True
                        break
                    else:
                        raise Exception(f"[Swipe reasoning direction mismatch] Action {i+1}: action_direction: {action_direction}, react: {react}")
            if not flag:
                raise Exception(f"[Swipe reasoning hasn't direction description] Action {i+1}: action_direction: {action_direction}, react: {react}")

change_task_description_prompt = load_prompt("change_task_description.md")
def change_task_description(app_name, original_task):
    count = 6
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "{sys_prompt}"),
                ("user", "{user_message}")
            ])
            chain = prompt | model
            response = chain.invoke({
                "sys_prompt": change_task_description_prompt.replace("{app_name}", app_name if app_name else "").replace("{original_task}", original_task).replace("{count}", str(count)),
                "user_message": f"请将任务'{original_task}'改写成{count}个版本" + (f"（前3条不带应用名称，后3条带应用名称）" if app_name else "（都不带应用名称）")
            })

            # 提取JSON内容
            pattern = re.compile(r"```json\n(.*?)\n```", re.DOTALL)
            match = pattern.search(response.content)
            
            if match is None:
                # 如果没有找到json代码块，尝试直接解析整个响应
                try:
                    data = json.loads(response.content.strip())
                    if isinstance(data, list) and len(data) == count:
                        return data
                except:
                    pass
                print(f"[Generate Task] Attempt {attempt + 1} failed, no valid JSON found in response.")
                continue
            
            json_str = match.group(1)
            data = json.loads(json_str)
            
            if not isinstance(data, list):
                raise Exception("Response is not a list")
                
            if len(data) != count:
                raise Exception(f"Expected {count} tasks, got {len(data)}")
                
            # 验证所有元素都是字符串
            for i, task in enumerate(data):
                if not isinstance(task, str) or not task.strip():
                    raise Exception(f"Task {i+1} is not a valid string")
            
            return data
            
        except Exception as e:
            print(f"[Generate Task] Attempt {attempt + 1} failed, error: {str(e)}")
            if attempt == max_attempts - 1:
                raise Exception(f"Failed to generate tasks after {max_attempts} attempts")
            continue


def add_action_index(actions):
    """为 actions 列表中的每个元素添加 action_index 字段"""
    for i, action in enumerate(actions):
        if isinstance(action, dict):
            action['action_index'] = i + 1
    return actions

def add_bounds_to_action(root, actions):
    for i, action in enumerate(actions):
        flag = False

        if action["type"] == "click":
            if not "bounds" in action:
                print(f"[Add Bounds] {root} Action {i + 1} no bounds, adding bounds")
                flag = True
            elif action["bounds"] is None:
                print(f"[Add Bounds] {root} Action {i + 1} bounds is None, adding bounds")
                flag = True

            # bounds = action.get("bounds", None)
            # if bounds is not None:
            #     x1, y1, x2, y2 = bounds
            #     # if x1 < 50 and x2 > 950:
            #     if x1 < 100 and x2 > 950:
            #         print(f"[Add Bounds] {root} Action {i + 1} bounds is Special")
            #         flag = True

        # flag = action["type"] == "click"

        if flag:
            img_path = os.path.join(root, f"{i + 1}.jpg")
            if not os.path.exists(img_path):
                raise Exception(f"[Add Bounds] Image not found: {img_path}")
                   
            # actions[i]["bounds"] = None         
            # bounds_list = extract_all_bounds(img_path)
            # actions[i]["bounds"] = find_clicked_element(bounds_list, action["position_x"], action["position_y"])

            bounds_list = extract_all_bounds(img_path)
            if "bounds" in action and action["bounds"]:
                bounds_list.append(action["bounds"])
            actions[i]["bounds"] = find_clicked_element(bounds_list, action["position_x"], action["position_y"])

    return actions

def visual_prompt(root, actions):
    print(f"[Visual Prompt] {root} begin")
    for file_name in os.listdir(root):
        # 检查文件是否以 '_highlighted.jpg' 结尾
        if file_name.endswith('_highlighted.jpg') or file_name.endswith('_bounds.jpg'):
            file_path = os.path.join(root, file_name)
            os.remove(file_path)

    jpg_files = [f for f in os.listdir(root) if f.endswith('.jpg')]

    if actions[-1]["type"] == "done":
        if(len(jpg_files)!= len(actions)):
            raise Exception(f"[Visual Prompt] {root} has {len(jpg_files)} images, but {len(actions)} actions with done")
    else:
        if(len(jpg_files)!= len(actions) + 1):
            raise Exception(f"[Visual Prompt] {root} has {len(jpg_files)} images, but {len(actions)} actions without     done")

    for i, action in enumerate(actions):
        img_path = os.path.join(root, f"{i + 1}.jpg")
        save_path = os.path.join(root, f"{i + 1}_highlighted.jpg")

        if not os.path.exists(img_path):
            raise Exception(f"[Visual Prompt] Image not found: {img_path}")

        img = Image.open(img_path)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("msyh.ttf", 40)
        if action["type"] == "click":
            text = f"CLICK [{int(action['position_x'])}, {int(action['position_y'])}]"
        elif action["type"] == "input":
            text = f"INPUT {action['text']}"
        elif action["type"] == "swipe":
            text = f"SWIPE [{int(action['press_position_x'])}, {int(action['press_position_y'])}] to [{int(action['release_position_x'])}, {int(action['release_position_y'])}]"
        elif action["type"] == "done":
            text = f"DONE"
        elif action["type"] == "long_press":
            text = f"LONG PRESS [{int(action['position_x'])}, {int(action['position_y'])}]"
        elif action["type"] == "open_app":
            text = f"OPEN APP {action['app_name']}"
        else:
            raise Exception(f"[Visual Prompt] Unknown action type: {action['type']}")
        
        text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:]
        draw.text((img.width / 2 - text_width / 2, 0), text, fill="red", font=font)
        img.save(save_path)

        # 拉框
        bounds_path = os.path.join(root, f"{i + 1}_bounds.jpg")
        img_bounds = Image.open(save_path)
        draw_bounds = ImageDraw.Draw(img_bounds)
        if action["type"] == "click" or action["type"] == "long_press":
            if "bounds" in action and action["bounds"]:
                draw_bounds.rectangle(action["bounds"], outline='red', width=5)
                img_bounds.save(bounds_path)
        # 画点
        with open(save_path, 'rb') as f:
            image_data = f.read()
        nparr = np.frombuffer(image_data, np.uint8)
        cv2image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if action["type"] == "click":
            x = int(action['position_x'])
            y = int(action['position_y'])
            cv2.circle(cv2image, (x, y), 50, (0, 0, 255), 10)
        elif action["type"] == "swipe":
            x1 = int(action['press_position_x'])
            y1 = int(action['press_position_y'])
            x2 = int(action['release_position_x'])
            y2 = int(action['release_position_y'])
            cv2.arrowedLine(cv2image, (x1, y1), (x2, y2), (0, 0, 255), 5)
        success, encoded_img = cv2.imencode('.jpg', cv2image)
        if success:
            with open(save_path, 'wb') as f:
                f.write(encoded_img.tobytes())
    print(f"[Visual Prompt] done")

def auto_annotate(root, chain, task_description, actions):
    print(f"[Reasoning] root: \"{root}\" task: \"{task_description}\"")

    files = os.listdir(root)
    image_data = []
    highlighted = [file for file in files if file.endswith("_highlighted.jpg")]
    highlighted.sort(key=lambda f: int(f.replace("_highlighted.jpg", "")))
    for file in highlighted:
        img_path = os.path.join(root, file)
        with open(img_path, "rb") as f:
            image_data.append(base64.b64encode(f.read()).decode("utf-8"))

    max_attempts = 3
    for attempt in range(0, max_attempts):
        response = chain.invoke({
            "goal": task_description,
            "screenshot_count": len(image_data),
            "messages": [
                (
                    "user",
                    [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}} for image in image_data]
                )
            ]
        })

        pattern = re.compile(r"```json\n(.*)\n```", re.DOTALL)
        match = pattern.search(response.content)
        if match is None:
            print(f"[Reasoning] Attempt {attempt + 1} failed, no JSON found in response.")
            continue
        
        try:
            json_str = match.group(1)
            data = json.loads(json_str)
            reasoning_count = len(data)
            if(len(image_data) != reasoning_count):
                raise Exception(f"[Invalid reasoning count]")
            react_json = os.path.join(root, "temp.json")
            with open(react_json, "w", encoding="UTF-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

            compare_actions(actions, data)

        except Exception as e:
            print(f"[Reasoning] Attempt {attempt + 1} failed, error: {str(e)}")
            continue
        break

    json_str = match.group(1)
    data = json.loads(json_str)
    reasoning_count = len(data)
    if(len(image_data) != reasoning_count):
        raise Exception(f"[Invalid reasoning count]")

    # # 为 react.json 中的数据添加 action_index
    # for i, item in enumerate(data):
    #     if isinstance(item, dict):
    #         item['action_index'] = i + 1

    compare_actions(actions, data)

    react_json = os.path.join(root, "react.json")
    with open(react_json, "w", encoding="UTF-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"[Reasoning] finished, saved to {react_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Auto annotation of GUI data')
    parser.add_argument('--data_path', type=str, default='data', help='root directory containing the data (default: data)')
    parser.add_argument('--model', type=str, required=True, help='name of the annotation model')
    parser.add_argument('--api_key', type=str, required=True, help='API key of the annotation model')
    parser.add_argument('--base_url', type=str, required=True, help='base URL of the annotation model')

    args = parser.parse_args()
    
    model = ChatOpenAI(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
    )

    from utils.load_md_prompt import load_prompt
    sys_prompt = load_prompt("annotation_en_general.md")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", sys_prompt),
            MessagesPlaceholder(variable_name='messages')
        ]
    )

    chain = prompt | model

    for root, dirs, files in os.walk(args.data_path):
        # 对子目录按数字顺序排序
        dirs.sort(key=lambda x: int(x) if x.isdigit() else float('inf'))
        
        try:
            actions_json = os.path.join(root, "actions.json")
            if not os.path.exists(actions_json):
                raise Exception("No actions.json")
      
            react_json = os.path.join(root, "react.json")
            if os.path.exists(react_json):
                continue
            parse_error = os.path.join(root, "parse.error")
            if os.path.exists(parse_error):
                continue
        
            with open(actions_json, 'r', encoding='utf-8') as file:
                data = json.load(file)
            if "task_description" not in data:
                raise Exception("No task_description in actions.json")
            task_description = data.get("task_description")
            actions = data.get("actions")

            # 不要随意开启这个，ocr有风险
            # actions = add_bounds_to_action(root, actions)
            # data["actions"] = actions
            # with open(actions_json, 'w', encoding='utf-8') as file:
            #     json.dump(data, file, ensure_ascii=False, indent=4)
            
            visual_prompt(root, actions)
            auto_annotate(root, chain, task_description, actions)

            app_name = data.get("app_name")
            if(isinstance(task_description, str)):
                new_tasks = change_task_description(app_name, task_description)
                all_tasks = [task_description] + new_tasks
                data["task_description"] = all_tasks

                with open(actions_json, 'w', encoding='utf-8') as file:
                    json.dump(data, file, ensure_ascii=False, indent=4)

                print(f"[Increase Task] finished, saved to {actions_json}")


        except Exception as e:
            with open(f"{root}/parse.error", 'w', encoding='utf-8', errors='ignore') as file:
                file.write(f"{str(e)}\n")
            with open(f"{args.data_path}/list.error", 'a', encoding='utf-8', errors='ignore') as file:
                file.write(f"root: \"{root}\" {str(e)}\n")

