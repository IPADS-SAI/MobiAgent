
import os
import time
import json
import base64
import logging
import io
import re
from typing import Dict, List, Optional
from PIL import Image

from base_task import BaseTask
from providers.qwen.utils import process_screenshot
from providers.qwen.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
import sys

# 使用模块级别的logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# base64编码图像的辅助函数
def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

class QwenTask(BaseTask):
    """
    Qwen VLM任务适配器（支持Qwen3-VL、Gemini等通用VLLM）
    使用Set-of-Marks (SoM)方法进行元素交互
    """
    
    def __init__(
        self,
        task_description: str,
        device,
        data_dir: str,
        device_type: str = "Android",
        use_step_loop: bool = True,
        # 统一参数
        api_base: str = None,
        api_key: str = "",
        model: str = None,
        temperature: float = None,
        # 向后兼容的旧参数
        model_name: str = None,
        **kwargs
    ):
        super().__init__(
            task_description=task_description,
            device=device,
            data_dir=data_dir,
            device_type=device_type,
            use_step_loop=use_step_loop,
            **kwargs
        )
        
        # 处理参数优先级: 新参数 > 旧参数 > 默认值
        self.model_name = model if model is not None else model_name 
        self.api_base = api_base
        self.api_key = api_key
        self.temperature = temperature if temperature is not None else 0.0
        
        # 初始化OpenAI客户端
        try:
            from openai import OpenAI
            self.client = OpenAI(
                base_url=self.api_base,
                api_key=self.api_key or "EMPTY"
            )
            logger.info(f"Initialized OpenAI client for {self.model_name} at {self.api_base}")
        except ImportError:
            logger.error("OpenAI package not found. Please install it: pip install openai")
            raise

    def execute_step(self, step_index: int) -> List[Dict]:
        """
        执行单一步骤：观察 -> 思考 -> 执行
        """
        logger.info(f"使用{self.model_name}执行第{step_index}步")
        
        current_screenshot_path = os.path.join(self.data_dir, f"{step_index}.jpg")
        
        # 验证截图是否存在
        if not os.path.exists(current_screenshot_path):
            logger.warning(f"截图不存在于{current_screenshot_path}，正在重新获取")
            self.device.screenshot(current_screenshot_path)
            
        # 处理SoM标注
        logger.info("处理截图以标注UI元素(SoM)...")
        bounds_list, layer_images = process_screenshot(current_screenshot_path)
        layer_count = len(layer_images)
        logger.info(f"生成了{layer_count}层SoM标注图层")
        
        # 保存调试用的图层图像
        for idx, img in enumerate(layer_images):
            layer_path = os.path.join(self.data_dir, f"{step_index}_layer_{idx+1}.jpg")
            img.convert("RGB").save(layer_path)

        # 构造提示词
        # 加载历史记录
        history_str = self._format_history()
        
        user_prompt = USER_PROMPT_TEMPLATE.format(
            task_description=self.task_description,
            history=history_str,
            layer_count=layer_count
        )
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": []}
        ]
        
        # 添加文本内容
        messages[1]["content"].append({"type": "text", "text": user_prompt})
        
        # 添加原始截图
        screenshot_b64 = image_to_base64(Image.open(current_screenshot_path))
        messages[1]["content"].append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{screenshot_b64}"}
        })
        
        # 添加图层标注图像
        for i, img in enumerate(layer_images):
            layer_b64 = image_to_base64(img)
            messages[1]["content"].append({"type": "text", "text": f"第{i+1}层:"})
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{layer_b64}"}
            })

        # 模型推理
        try:
            logger.info(f"[{self.model_name}] 正在发送请求到VLLM... (超时: 60s)")
            start_time = time.time()
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=1024,
                timeout=60.0,
                response_format={"type": "json_object"}
            )
            elapsed = time.time() - start_time
            logger.info(f"[{self.model_name}] VLLM响应已接收，耗时{elapsed:.2f}s")
            
            response_content = completion.choices[0].message.content
            logger.info(f"[{self.model_name}] 原始响应: {response_content}")
            
        except Exception as e:
            logger.error(f"[{self.model_name}] VLLM请求失败: {e}")
            return [{"type": "error", "error": str(e)}]

        # 解析响应
        try:
            action_data = self._parse_json_response(response_content)
            logger.info(f"解析的动作: {action_data}")
            
            # 映射动作到runner的动作格式
            return self._map_to_runner_actions(action_data, bounds_list, step_index)
            
        except Exception as e:
            logger.error(f"解析响应失败: {e}")
            return [{"type": "error", "error": str(e)}]

    def _format_history(self) -> str:
        if not self.actions:
            return "(暂无历史记录)"
        
        formatted = []
        for i, act in enumerate(self.actions[-5:]):
            act_type = act.get("type", "unknown")
            formatted.append(f"步骤 {act.get('action_index', '?')}: {act_type}")
        
        return "\n".join(formatted)

    def _parse_json_response(self, content: str) -> Dict:
        # 从代码块中提取JSON（如果存在）
        json_match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = content
            
        json_str = json_str.strip()
        
        return json.loads(json_str)

    def _map_to_runner_actions(self, action_data: Dict, bounds_list: List, step_index: int) -> List[Dict]:
        action_type = action_data.get("action", "").upper().strip()
        params = action_data.get("params", {})
        reasoning = action_data.get("reasoning", "")
        
        # 获取屏幕尺寸以进行坐标转换
        current_screenshot_path = os.path.join(self.data_dir, f"{step_index}.jpg")
        screen_width, screen_height = 1080, 1920
        if os.path.exists(current_screenshot_path):
            with Image.open(current_screenshot_path) as img:
                screen_width, screen_height = img.size
                logger.info(f"屏幕尺寸: {screen_width}x{screen_height}")
        
        # 规范化动作类型别名
        action_type_map = {
            "TAP": "CLICK",
            "LONG_PRESS": "LONG PRESS",
            "SCROLL": "SWIPE",
            "TYPE": "INPUT",
            "INPUT_TEXT": "INPUT",
            "OPEN": "OPEN_APP",
            "BACK": "KEY_BACK",
            "HOME": "KEY_HOME",
        }
        action_type = action_type_map.get(action_type, action_type)
        
        logger.info(f"动作类型: {action_type}, 参数: {params}")
        
        # 保存推理过程
        self._add_react(reasoning, action_type, params, step_index)
        
        runner_actions = []
        
        # 将相对坐标(0-1000)转换为绝对屏幕坐标的辅助函数
        def convert_coordinates(coord_list):
            """将[x, y]从1000x1000相对坐标转换为绝对屏幕坐标"""
            if not coord_list or len(coord_list) != 2:
                return coord_list
            rel_x, rel_y = coord_list
            abs_x = int(rel_x * screen_width / 1000)
            abs_y = int(rel_y * screen_height / 1000)
            logger.info(f"坐标转换: [{rel_x}, {rel_y}] (相对) -> [{abs_x}, {abs_y}] (绝对)")
            return [abs_x, abs_y]
        
        if action_type in ["CLICK", "LONG PRESS"]:
            index = params.get("index")
            coordinate = params.get("coordinate") or params.get("position")
            
            # 情形1: 使用元素索引
            if index is not None and coordinate is None:
                if 0 <= index < len(bounds_list):
                    bounds = bounds_list[index]
                    # 计算中心坐标
                    center_x = (bounds[0] + bounds[2]) // 2
                    center_y = (bounds[1] + bounds[3]) // 2
                    
                    runner_action = {
                        "type": "click" if action_type == "CLICK" else "long_press",
                        "params": {
                            "coordinate": [center_x, center_y],
                            "index": index,
                            "bounds": bounds,
                            "reasoning": reasoning
                        }
                    }
                    runner_actions.append(runner_action)
                else:
                    logger.warning(f"边界列表大小为{len(bounds_list)}，索引{index}无效")
                    runner_actions.append({"type": "fail", "params": {"reason": f"无效的元素索引{index}"}})
            
            # 情形2: 使用直接坐标
            elif coordinate is not None:
                abs_coordinate = convert_coordinates(coordinate)
                runner_action = {
                    "type": "click" if action_type == "CLICK" else "long_press",
                    "params": {
                        "coordinate": abs_coordinate,
                        "reasoning": reasoning
                    }
                }
                runner_actions.append(runner_action)
            
            else:
                logger.warning(f"CLICK/LONG PRESS动作缺少索引和坐标")
                runner_actions.append({"type": "fail", "params": {"reason": "缺少点击动作的索引或坐标"}})

        elif action_type == "INPUT":
            text = params.get("text", "")
            runner_actions.append({
                "type": "input",
                "params": {"text": text, "reasoning": reasoning}
            })

        elif action_type == "SWIPE":
            start = params.get("start")
            end = params.get("end")
            if start and end:
                # 将相对坐标转换为绝对坐标
                abs_start = convert_coordinates(start)
                abs_end = convert_coordinates(end)
                runner_actions.append({
                    "type": "swipe",
                    "params": {
                        "start_coordinate": abs_start,
                        "end_coordinate": abs_end,
                        "reasoning": reasoning
                    }
                })
            else:
                runner_actions.append({"type": "fail", "params": {"reason": "缺少滑动坐标"}})

        elif action_type == "OPEN_APP":
            app_name = params.get("app_name")
            if app_name:
                runner_actions.append({
                    "type": "open_app",
                    "params": {"app_name": app_name, "reasoning": reasoning}
                })
                logger.info(f"打开应用: {app_name}")
            else:
                logger.warning("OPEN_APP 动作缺少应用名称")
                runner_actions.append({"type": "fail", "params": {"reason": "缺少应用名称"}})

        elif action_type == "KEY_BACK":
            runner_actions.append({
                "type": "back",
                "params": {"reasoning": reasoning}
            })
            logger.info("执行返回动作")

        elif action_type == "KEY_HOME":
            runner_actions.append({
                "type": "home",
                "params": {"reasoning": reasoning}
            })
            logger.info("执行主屏动作")

        elif action_type == "WAIT":
            duration = params.get("duration", 2)
            runner_actions.append({
                "type": "wait",
                "params": {"seconds": duration, "reasoning": reasoning}
            })
            logger.info(f"等待 {duration} 秒")

        elif action_type == "DONE":
            status = params.get("status", "success")
            runner_actions.append({
                "type": "done",
                "params": {"status": status, "reasoning": reasoning}
            })
            logger.info(f"任务完成，状态: {status}")
            
        else:
            logger.warning(f"未知的动作类型: {action_type}")
            runner_actions.append({"type": "fail", "params": {"reason": f"未知的动作{action_type}"}})
            
        return runner_actions
