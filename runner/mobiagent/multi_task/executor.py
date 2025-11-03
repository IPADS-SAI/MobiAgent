"""
Executor模块
包含task_in_app、decider、grounder的逻辑
"""

import base64
import json
import logging
import time
import re
import os
import textwrap
from typing import Dict, List
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    # 尝试相对导入（当作为模块运行时）
    from .models import ActionPlan, GroundResponse
    from .prompts import (
        DECIDER_PROMPT_TEMPLATE,
        GROUNDER_QWEN3_BBOX_PROMPT,
        GROUNDER_PROMPT_BBOX,
        GROUNDER_PROMPT_NO_BBOX
    )
except ImportError:
    # 回退到绝对导入（当直接运行时）
    from models import ActionPlan, GroundResponse
    from prompts import (
        DECIDER_PROMPT_TEMPLATE,
        GROUNDER_QWEN3_BBOX_PROMPT,
        GROUNDER_PROMPT_BBOX,
        GROUNDER_PROMPT_NO_BBOX
    )

# 全局配置
MAX_STEPS = 40
screenshot_path = "screenshot.jpg"
factor = 0.5


def parse_json_response(response_str: str) -> dict:
    """解析JSON响应"""
    print("Parsing JSON response...")
    try:
        return json.loads(response_str)
    except json.JSONDecodeError:
        try:
            # 查找JSON代码块
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_str, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            
            # 查找花括号包围的JSON
            json_match = re.search(r'(\{.*?\})', response_str, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            
            raise ValueError("无法在响应中找到有效的JSON")
        except Exception as e:
            logging.error(f"JSON解析失败: {e}")
            logging.error(f"原始响应: {response_str}")
            raise ValueError(f"无法解析JSON响应: {e}")


def get_screenshot(device):
    """获取设备截图并转换为base64"""
    device.screenshot(screenshot_path)
    img = Image.open(screenshot_path)
    img = img.resize((int(img.width * factor), int(img.height * factor)), Image.Resampling.LANCZOS)
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    screenshot = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return screenshot


def _save_execution_data(data_dir: str, app: str, old_task: str, task: str, actions: List, reacts: List):
    """保存执行数据到JSON文件"""
    # 保存actions.json
    actions_data = {
        "app_name": app,
        "task_type": None,
        "old_task_description": old_task,
        "task_description": task,
        "action_count": len(actions),
        "actions": actions
    }
    
    actions_path = os.path.join(data_dir, "actions.json")
    with open(actions_path, "w", encoding='utf-8') as f:
        json.dump(actions_data, f, ensure_ascii=False, indent=4)
    
    # 保存react.json
    react_path = os.path.join(data_dir, "react.json")
    with open(react_path, "w", encoding='utf-8') as f:
        json.dump(reacts, f, ensure_ascii=False, indent=4)
    
    logging.info(f"Saved execution data to {data_dir}")


def create_swipe_visualization(data_dir, image_index, direction):
    """为滑动动作创建可视化图像"""
    try:
        img_path = os.path.join(data_dir, f"{image_index}.jpg")
        if not os.path.exists(img_path):
            return
            
        img = cv2.imread(img_path)
        if img is None:
            return
            
        height, width = img.shape[:2]
        center_x, center_y = width // 2, height // 2
        arrow_length = min(width, height) // 4
        
        direction_map = {
            "up": (center_x, center_y + arrow_length // 2, center_x, center_y - arrow_length // 2),
            "down": (center_x, center_y - arrow_length // 2, center_x, center_y + arrow_length // 2),
            "left": (center_x + arrow_length // 2, center_y, center_x - arrow_length // 2, center_y),
            "right": (center_x - arrow_length // 2, center_y, center_x + arrow_length // 2, center_y)
        }
        
        if direction not in direction_map:
            return
        
        x1, y1, x2, y2 = direction_map[direction]
        cv2.arrowedLine(img, (x1, y1), (x2, y2), (255, 0, 0), 8, tipLength=0.3)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"SWIPE {direction.upper()}"
        text_size = cv2.getTextSize(text, font, 1.5, 3)[0]
        text_x = (width - text_size[0]) // 2
        cv2.putText(img, text, (text_x, 50), font, 1.5, (255, 0, 0), 3)
        
        swipe_path = os.path.join(data_dir, f"{image_index}_swipe.jpg")
        cv2.imwrite(swipe_path, img)
        
    except Exception as e:
        logging.warning(f"Failed to create swipe visualization: {e}")


def robust_json_loads(s: str) -> dict:
    """健壮的JSON解析"""
    s = s.strip()
    # 提取代码块
    codeblock = re.search(r"```json(.*?)```", s, re.DOTALL)
    if codeblock:
        s = codeblock.group(1).strip()
    # 替换省略号
    s = s.replace("…", "...")
    # 去除多余换行
    s = s.replace("\r", "").replace("\n", " ")
    
    try:
        return json.loads(s)
    except Exception as e:
        logging.error(f"解析 decider_response_str 失败: {e}\n原始内容: {s}")
        raise


import io

def convert_qwen3_coordinates_to_absolute(bbox_or_coords, img_width, img_height, is_bbox=True):
    """
    将 Qwen3 模型返回的相对坐标（0-1000范围）转换为绝对坐标
    
    Args:
        bbox_or_coords: 相对坐标或边界框，范围为 0-1000
        img_width: 图像宽度
        img_height: 图像高度
        is_bbox: 是否为边界框（True）或坐标点（False）
        
    Returns:
        转换后的绝对坐标或边界框
    """
    if is_bbox:
        # bbox: [x1, y1, x2, y2]
        x1, y1, x2, y2 = bbox_or_coords
        x1 = int(x1 / 1000 * img_width)
        x2 = int(x2 / 1000 * img_width)
        y1 = int(y1 / 1000 * img_height)
        y2 = int(y2 / 1000 * img_height)
        return [x1, y1, x2, y2]
    else:
        # coordinates: [x, y]
        x, y = bbox_or_coords
        x = int(x / 1000 * img_width)
        y = int(y / 1000 * img_height)
        return [x, y]

def task_in_app(
    app: str,
    old_task: str,
    task: str,
    device,
    data_dir: str,
    decider_client: OpenAI,
    grounder_client: OpenAI,
    decider_model: str,
    grounder_model: str,
    bbox_flag: bool = True,
    use_qwen3: bool = True
) -> Dict:
    """
    执行单个子任务（在单个APP内）
    
    Args:
        app: 应用名称
        old_task: 原始任务描述
        task: 当前任务描述
        device: 设备对象
        data_dir: 数据存储目录
        decider_client: Decider客户端
        grounder_client: Grounder客户端
        decider_model: Decider模型名称
        grounder_model: Grounder模型名称
        bbox_flag: 是否使用bbox模式
        use_qwen3: 是否使用Qwen3模型进行坐标转换
    
    Returns:
        dict: 包含执行结果的字典
    """
    history = []
    actions = []
    reacts = []
    screenshots_base64 = []
    skip_temp = False
    
    # 创建初始的空数据文件
    _save_execution_data(data_dir, app, old_task, task, actions, reacts)
    
    while True:
        if len(actions) >= MAX_STEPS:
            logging.info("Reached maximum steps, stopping the task.")
            # 保存执行数据
            _save_execution_data(data_dir, app, old_task, task, actions, reacts)
            break
        
        history_str = "(No history)" if len(history) == 0 else "\n".join(
            f"{idx}. {h}" for idx, h in enumerate(history, 1)
        )
        
        if len(actions) < 10:
            print(f"Step {len(actions)+1}, History:\n{history_str}\n")
        
        # 获取截图
        screenshot = get_screenshot(device)
        screenshots_base64.append(screenshot)
        
        # Decider决策
        decider_prompt = DECIDER_PROMPT_TEMPLATE.format(task=task, history=history_str)
        logging.info(f"Decider prompt: \n{decider_prompt}")
        
        decider_start_time = time.time()
        decider_response_str = decider_client.chat.completions.create(
            model=decider_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{screenshot}"}},
                    {"type": "text", "text": decider_prompt},
                ]
            }],
            temperature=0,
            response_format={
                "type": "json_object",
                "schema": ActionPlan.model_json_schema()
            }
        ).choices[0].message.content
        
        decider_end_time = time.time()
        logging.info(f"Decider time taken: {decider_end_time - decider_start_time} seconds")
        logging.info(f"Decider response: \n{decider_response_str}")
        
        parsed_plan = ActionPlan.model_validate_json(decider_response_str)
        logging.info(f"Parsed plan: {parsed_plan}")
        
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
        
        # 计算图像索引
        image_index = len(actions) + 1
        current_dir = os.getcwd()
        img_path = os.path.join(current_dir, screenshot_path)
        save_path = os.path.join(data_dir, f"{image_index}.jpg")
        img = Image.open(img_path)
        img.save(save_path)
        
        # 附加索引
        if reacts:
            try:
                reacts[-1]["action_index"] = image_index
            except Exception:
                pass
        
        # 保存层级结构
        hierarchy_path = os.path.join(data_dir, f"{image_index}.xml")
        hierarchy = device.dump_hierarchy()
        with open(hierarchy_path, "w", encoding="utf-8") as f:
            f.write(hierarchy)
        
        # 处理done动作
        if action == "done":
            print("Task completed.")
            actions.append({"type": "done", "action_index": image_index})
            
            # 保存最终执行结果
            _save_execution_data(data_dir, app, old_task, task, actions, reacts)
            
            return {
                "success": True,
                "reacts": reacts,
                "screenshots": screenshots_base64,
                "actions": actions
            }
        
        # 处理click动作
        if action == "click":
            reasoning = decider_response["reasoning"]
            target_element = decider_response["parameters"]["target_element"]
            
            # 跳过支付操作
            if any(keyword in target_element for keyword in ["支付宝支付", "¥", "立即支付", "免密支付"]):
                print("Skipping click action for payment.")
                skip_temp = True
            
            grounder_prompt = (GROUNDER_QWEN3_BBOX_PROMPT if bbox_flag else GROUNDER_PROMPT_NO_BBOX).format(
                reasoning=reasoning, description=target_element
            )
            
            grounder_start_time = time.time()
            grounder_response_str = grounder_client.chat.completions.create(
                model=grounder_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{screenshot}"}},
                        {"type": "text", "text": grounder_prompt},
                    ]
                }],
                temperature=0,
            ).choices[0].message.content
            
            grounder_end_time = time.time()
            logging.info(f"Grounder time taken: {grounder_end_time - grounder_start_time} seconds")
            logging.info(f"Grounder response: \n{grounder_response_str}")
            
            grounder_response = parse_json_response(grounder_response_str)
            
            if bbox_flag:
                bbox = grounder_response.get("bbox") or grounder_response.get("bbox_2d") or grounder_response.get("bbox-2d")
            
                # 如果使用 Qwen3 模型，进行坐标转换
                if use_qwen3:
                    x1, y1, x2, y2 = [int(coord) for coord in bbox]
                    bbox = convert_qwen3_coordinates_to_absolute(bbox, img.width, img.height, is_bbox=True)
                    x1, y1, x2, y2 = bbox
                else:
                    x1, y1, x2, y2 = [int(coord / factor) for coord in bbox]

                position_x = (x1 + x2) // 2
                position_y = (y1 + y2) // 2
                
                if not skip_temp:
                    device.click(position_x, position_y)
                
                actions.append({
                    "type": "click",
                    "position_x": position_x,
                    "position_y": position_y,
                    "bounds": [x1, y1, x2, y2],
                    "action_index": image_index
                })
                
                history.append(decider_response_str)
                
                # 创建可视化
                _create_click_visualization(data_dir, image_index, position_x, position_y, x1, y1, x2, y2, img)
                
            else:
                coordinates = grounder_response["coordinates"]

                # 如果使用 Qwen3 模型，进行坐标转换
                if use_qwen3:
                    x, y = [int(coord) for coord in coordinates]
                    coords = convert_qwen3_coordinates_to_absolute(coordinates, img.width, img.height, is_bbox=False)
                    x, y = coords
                else:
                     x, y = [int(coord / factor) for coord in coordinates]
                
                if not skip_temp:
                    device.click(x, y)
                actions.append({
                    "type": "click",
                    "position_x": x,
                    "position_y": y,
                    "action_index": image_index
                })
            
            skip_temp = False
            
            # 保存当前执行状态
            _save_execution_data(data_dir, app, old_task, task, actions, reacts)
        
        # 处理input动作
        elif action == "input":
            text = decider_response["parameters"]["text"]
            device.input(text)
            actions.append({"type": "input", "text": text, "action_index": image_index})
            history.append(decider_response_str)
            
            # 保存当前执行状态
            _save_execution_data(data_dir, app, old_task, task, actions, reacts)
        
        # 处理swipe动作
        elif action == "swipe":
            direction = decider_response["parameters"]["direction"]
            
            if direction == "DOWN":
                device.swipe(direction.lower(), 1)
            elif direction in ["UP", "LEFT", "RIGHT"]:
                device.swipe(direction.lower())
            else:
                raise ValueError(f"Unknown swipe direction: {direction}")
            
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
            create_swipe_visualization(data_dir, image_index, direction.lower())
            
            # 保存当前执行状态
            _save_execution_data(data_dir, app, old_task, task, actions, reacts)
            
            if direction == "DOWN":
                continue
        
        # 处理wait动作
        elif action == "wait":
            print("Waiting for a while...")
            actions.append({"type": "wait", "action_index": image_index})
            history.append(decider_response_str)
            
            # 保存当前执行状态
            _save_execution_data(data_dir, app, old_task, task, actions, reacts)
        
        else:
            raise ValueError(f"Unknown action: {action}")

    # 确保最终状态被保存（如果还没有保存的话）
    if actions:  # 如果有执行过动作
        _save_execution_data(data_dir, app, old_task, task, actions, reacts)
    
    return {
        "success": len(actions) < MAX_STEPS,
        "reacts": reacts,
        "screenshots": screenshots_base64,
        "actions": actions
    }


def _create_click_visualization(data_dir, image_index, position_x, position_y, x1, y1, x2, y2, img):
    """创建点击动作的可视化图像"""
    current_dir = os.getcwd()
    img_path = os.path.join(current_dir, screenshot_path)
    save_path = os.path.join(data_dir, f"{image_index}_highlighted.jpg")
    
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("msyh.ttf", 40)
    text = f"CLICK [{position_x}, {position_y}]"
    text = textwrap.fill(text, width=20)
    text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:]
    draw.text((img.width / 2 - text_width / 2, 0), text, fill="red", font=font)
    img.save(save_path)
    
    # 绘制边界框
    bounds_path = os.path.join(data_dir, f"{image_index}_bounds.jpg")
    img_bounds = Image.open(save_path)
    draw_bounds = ImageDraw.Draw(img_bounds)
    draw_bounds.rectangle([x1, y1, x2, y2], outline='red', width=5)
    img_bounds.save(bounds_path)
    
    # 绘制点击点
    cv2image = cv2.imread(bounds_path)
    if cv2image is not None:
        cv2.circle(cv2image, (position_x, position_y), 15, (0, 255, 0), -1)
        click_point_path = os.path.join(data_dir, f"{image_index}_click_point.jpg")
        cv2.imwrite(click_point_path, cv2image)
