"""
AutoGLM Task Adapter for MobiAgent Runner Framework.

This adapter integrates AutoGLM model into the unified execution framework.
Reference: Open-AutoGLM/phone_agent/agent.py
"""

import os
import sys
import time
import json
import base64
import logging
from typing import Dict, List, Optional, Any
from PIL import Image

from openai import OpenAI

# 使用模块级别的logger（级别由 setup_logging() 统一配置）
logger = logging.getLogger(__name__)

# 添加父目录到路径以导入base_task
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from base_task import BaseTask

from providers.autoglm.prompts import SYSTEM_PROMPT
from providers.autoglm.action_parser import parse_action, parse_response


class AutoGLMTask(BaseTask):
    """
    AutoGLM 任务适配器（步骤循环模式）
    
    使用 AutoGLM 模型进行手机自动化任务。
    支持的动作包括：Launch, Tap, Type, Swipe, Back, Home, Long Press, Double Tap, Wait, finish
    
    坐标系统：使用相对坐标 (0-1000)，自动转换为绝对屏幕坐标
    """
    
    def __init__(
        self,
        task_description: str,
        device,
        data_dir: str,
        device_type: str = "Android",
        max_steps: int = 30,
        # 统一参数
        api_base: str = None,
        api_key: str = None,
        model: str = None,
        temperature: float = None,
        # AutoGLM 专属参数
        max_tokens: int = 3000,
        top_p: float = 0.85,
        frequency_penalty: float = 0.2,
        # 向后兼容的旧参数
        model_base_url: str = None,
        model_name: str = None,
        **kwargs
    ):
        """
        初始化 AutoGLM 任务
        
        Args:
            task_description: 任务描述
            device: 设备对象
            data_dir: 数据保存目录
            device_type: 设备类型，目前仅支持 Android
            max_steps: 最大步骤数
            api_base: 模型服务基础URL
            api_key: API密钥
            model: 模型名称
            temperature: 生成温度
            max_tokens: 最大输出token数
            top_p: Top-p采样参数
            frequency_penalty: 频率惩罚参数
        """
        super().__init__(
            task_description=task_description,
            device=device,
            data_dir=data_dir,
            device_type=device_type,
            max_steps=max_steps,
            use_step_loop=True,  # 使用步骤循环模式
            **kwargs
        )
        
        # 处理参数优先级: 新参数 > 旧参数 > 默认值
        self.api_base = api_base or model_base_url or "http://localhost:8000/v1"
        self.model_name = model or model_name or "autoglm-phone-9b"
        self.api_key = api_key or "EMPTY"
        self.temperature = temperature if temperature is not None else 0.0
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        
        # 初始化 OpenAI 客户端
        try:
            self.client = OpenAI(
                base_url=self.api_base,
                api_key=self.api_key
            )
            logger.info(f"AutoGLM client initialized: {self.model_name} at {self.api_base}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
        
        # 对话上下文
        self._context: List[Dict[str, Any]] = []
        
        logger.info("AutoGLMTask initialized")
    
    def execute_step(self, step_index: int) -> List[Dict]:
        """
        执行单步操作
        
        Args:
            step_index: 当前步骤索引
            
        Returns:
            动作序列列表
        """
        # 获取截图路径
        screenshot_path = os.path.join(self.data_dir, f"{step_index}.jpg")
        
        # 读取截图
        try:
            with open(screenshot_path, "rb") as f:
                image_data = f.read()
            encoded_image = base64.b64encode(image_data).decode('utf-8')
            
            # 获取图像尺寸
            with Image.open(screenshot_path) as img:
                screen_width, screen_height = img.size
        except Exception as e:
            logger.error(f"Failed to read screenshot: {e}")
            return [{"type": "retry", "params": {"error": str(e)}}]
        
        # 获取当前应用信息（尝试获取）
        try:
            current_app = self._get_current_app()
        except Exception:
            current_app = "Unknown"
        
        # 构建消息
        is_first = len(self._context) == 0
        
        if is_first:
            # 第一步：添加系统提示和用户任务
            self._context.append({
                "role": "system",
                "content": SYSTEM_PROMPT
            })
            
            screen_info = json.dumps({"current_app": current_app}, ensure_ascii=False)
            text_content = f"{self.task_description}\n\n** Screen Info **\n\n{screen_info}"
            
            self._context.append({
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
                    {"type": "text", "text": text_content}
                ]
            })
        else:
            # 后续步骤：添加当前屏幕观察
            screen_info = json.dumps({"current_app": current_app}, ensure_ascii=False)
            text_content = f"** Screen Info **\n\n{screen_info}"
            
            self._context.append({
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
                    {"type": "text", "text": text_content}
                ]
            })
        
        # 调用模型
        try:
            logger.info(f"Calling AutoGLM model (step {step_index})...")
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self._context,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                stream=False
            )
            
            raw_content = response.choices[0].message.content
            elapsed_time = time.time() - start_time
            logger.info(60*'-')
            logger.info(f"Model response ({elapsed_time:.2f}s): {raw_content}")
            logger.info(60*'-')
            
        except Exception as e:
            logger.error(f"Model request failed: {e}")
            return [{"type": "retry", "params": {"error": str(e)}}]
        
        # 解析响应
        try:
            thinking, action_str = parse_response(raw_content)
            logger.info(f"Thinking: {thinking[:100]}...")
            logger.info(f"Action string: {action_str}")
            
            action = parse_action(action_str)
            logger.info(f"Parsed action: {json.dumps(action, ensure_ascii=False)}")
            
        except ValueError as e:
            logger.error(f"Failed to parse response: {e}")
            # 尝试将整个响应作为finish
            action = {"_metadata": "finish", "message": raw_content}
        
        # 从context中移除图像以节省空间
        if len(self._context) > 0:
            last_msg = self._context[-1]
            if isinstance(last_msg.get("content"), list):
                self._context[-1]["content"] = [
                    item for item in last_msg["content"] 
                    if item.get("type") == "text"
                ]
        
        # 添加助手响应到上下文
        self._context.append({
            "role": "assistant",
            "content": f"<think>{thinking}</think><answer>{action_str}</answer>"
        })
        
        # 记录推理过程
        self._add_react(
            reasoning=thinking,
            action=action.get("action", action.get("_metadata", "unknown")),
            parameters=action,
            step_index=step_index
        )
        
        # 转换为 runner 标准格式
        return self._convert_to_runner_actions(action, screen_width, screen_height)
    
    def _get_current_app(self) -> str:
        """获取当前应用名称"""
        try:
            # 使用 adb 获取当前焦点窗口
            import subprocess
            result = subprocess.run(
                ["adb", "shell", "dumpsys", "window"],
                capture_output=True,
                text=True,
                encoding="utf-8"
            )
            output = result.stdout
            
            for line in output.split("\n"):
                if "mCurrentFocus" in line or "mFocusedApp" in line:
                    # 简单返回包名
                    return line.strip()
            
            return "System Home"
        except Exception:
            return "Unknown"
    
    def _convert_to_runner_actions(
        self, 
        action: Dict[str, Any], 
        screen_width: int, 
        screen_height: int
    ) -> List[Dict]:
        """
        将 AutoGLM 动作转换为 runner 标准格式
        
        Args:
            action: AutoGLM 解析后的动作
            screen_width: 屏幕宽度
            screen_height: 屏幕高度
            
        Returns:
            runner 标准动作序列
        """
        action_type = action.get("_metadata")
        
        # 处理 finish 动作
        if action_type == "finish":
            return [{
                "type": "done",
                "params": {
                    "status": "success",
                    "message": action.get("message", "Task completed")
                }
            }]
        
        if action_type != "do":
            logger.warning(f"Unknown action metadata: {action_type}")
            return [{"type": "retry", "params": {}}]
        
        action_name = action.get("action", "")
        
        # Launch -> open_app
        if action_name == "Launch":
            app_name = action.get("app", "")
            return [{
                "type": "open_app",
                "params": {"app_name": app_name}
            }]
        
        # Tap -> click
        elif action_name == "Tap":
            element = action.get("element", [500, 500])
            x, y = self._convert_relative_to_absolute(element, screen_width, screen_height)
            return [{
                "type": "click",
                "params": {"coordinate": [x, y]}
            }]
        
        # Type / Type_Name -> input_text
        elif action_name in ["Type", "Type_Name"]:
            text = action.get("text", "")
            return [{
                "type": "input_text",
                "params": {"text": text}
            }]
        
        # Swipe -> scroll
        elif action_name == "Swipe":
            start = action.get("start", [500, 700])
            end = action.get("end", [500, 300])
            start_x, start_y = self._convert_relative_to_absolute(start, screen_width, screen_height)
            end_x, end_y = self._convert_relative_to_absolute(end, screen_width, screen_height)
            return [{
                "type": "scroll",
                "params": {
                    "start_coordinate": [start_x, start_y],
                    "end_coordinate": [end_x, end_y]
                }
            }]
        
        # Back -> back
        elif action_name == "Back":
            return [{"type": "back", "params": {}}]
        
        # Home -> home
        elif action_name == "Home":
            return [{"type": "home", "params": {}}]
        
        # Double Tap -> doubleclick
        elif action_name == "Double Tap":
            element = action.get("element", [500, 500])
            x, y = self._convert_relative_to_absolute(element, screen_width, screen_height)
            return [{
                "type": "doubleclick",
                "params": {"coordinate": [x, y]}
            }]
        
        # Long Press -> longclick
        elif action_name == "Long Press":
            element = action.get("element", [500, 500])
            x, y = self._convert_relative_to_absolute(element, screen_width, screen_height)
            return [{
                "type": "longclick",
                "params": {"coordinate": [x, y]}
            }]
        
        # Wait -> wait
        elif action_name == "Wait":
            duration_str = action.get("duration", "2 seconds")
            try:
                duration = float(duration_str.replace("seconds", "").strip())
            except ValueError:
                duration = 2.0
            return [{
                "type": "wait",
                "params": {"seconds": duration}
            }]
        
        # Take_over -> wait (需要用户介入，暂时作为等待处理)
        elif action_name == "Take_over":
            logger.warning(f"Take_over requested: {action.get('message', '')}")
            return [{
                "type": "wait",
                "params": {"seconds": 5}
            }]
        
        # Note / Call_API / Interact -> wait (暂不实现具体功能)
        elif action_name in ["Note", "Call_API", "Interact"]:
            logger.info(f"Action {action_name} treated as wait")
            return [{
                "type": "wait",
                "params": {"seconds": 1}
            }]
        
        else:
            logger.warning(f"Unknown action: {action_name}")
            return [{"type": "retry", "params": {}}]
    
    def _convert_relative_to_absolute(
        self, 
        element: List[int], 
        screen_width: int, 
        screen_height: int
    ) -> tuple:
        """
        将相对坐标 (0-1000) 转换为绝对像素坐标
        
        Args:
            element: [x, y] 相对坐标 (0-1000)
            screen_width: 屏幕宽度
            screen_height: 屏幕高度
            
        Returns:
            (x, y) 绝对坐标
        """
        x = int(element[0] / 1000 * screen_width)
        y = int(element[1] / 1000 * screen_height)
        return x, y
