import sys
import os
import time
import base64
import json
import logging
from typing import Dict, List, Optional
from openai import OpenAI

# 使用模块级别的logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 导入UI-TARS所需的模块
runner_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
agent_root = os.path.dirname(os.path.dirname(os.path.dirname(runner_dir)))

from base_task import BaseTask
from .ui_tars_helper import UITarsHelper, PROMPT_TEMPLATE

class UITARSTask(BaseTask):
    """
    UI-TARS任务适配器（独立实现）
    """
    
    def __init__(
        self,
        task_description: str,
        device,
        data_dir: str,
        device_type: str = "Android",
        max_steps: int = 40,
        # 统一参数
        api_base: str = None,
        model: str = "UI-TARS-1.5-7B",
        temperature: float = 0.0,
        # UI-TARS 专属参数
        step_delay: float = 2.0,
        device_ip: Optional[str] = None,
        # 向后兼容的旧参数
        model_base_url: str = None,
        model_name: str = None,
        **kwargs
    ):
        super().__init__(
            task_description=task_description,
            device=device,
            data_dir=data_dir,
            device_type=device_type,
            max_steps=max_steps,
            use_step_loop=True,
            **kwargs
        )
        
        # 处理参数优先级: 新参数 > 旧参数 > 默认值
        self.model_base_url = api_base or model_base_url or "http://localhost:8000/v1"
        self.model_name = model or model_name or "UI-TARS-1.5-7B"
        self.temperature = temperature if temperature is not None else 0.0
        self.step_delay = step_delay
        # 初始化OpenAI客户端
        try:
            self.client = OpenAI(
                base_url=self.model_base_url,
                api_key="EMPTY"
            )
            logger.info(f"已连接到模型服务: {self.model_base_url}")
        except Exception as e:
            logger.error(f"初始化OpenAI客户端失败: {e}")
            raise

        # UI-TARS特定的历史记录
        self.history = []
        
        logger.info("UITARSTask (步骤循环模式)已初始化")

    def execute_step(self, step_index: int) -> List[Dict]:
        """
        执行单一步骤
        """
        screenshot_path = os.path.join(self.data_dir, f"{step_index}.jpg")
        
        # 读取截图
        if not os.path.exists(screenshot_path):
            self.device.screenshot(screenshot_path)
            
        with open(screenshot_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        image_data_url = f"data:image/jpeg;base64,{image_data}"
        
        # 获取图像尺寸
        try:
            from PIL import Image
            with Image.open(screenshot_path) as img:
                width, height = img.size
        except Exception as e:
            logger.warning(f"获取图像尺寸失败: {e}")
            width, height = 1080, 2340
            
        # 构建消息
        messages = self._build_messages(image_data_url)
        
        # 调用模型
        try:
            logger.info("调用UI-TARS模型...")
            start_time = time.time()
            chat_completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=512,
                stream=False
            )
            response = chat_completion.choices[0].message.content
            logger.info(f"模型响应 ({time.time() - start_time:.2f}s): {response}")
        except Exception as e:
            logger.error(f"模型调用失败: {e}")
            return [{"type": "retry", "params": {"error": str(e)}}]

        # 使用辅助函数解析响应
        try:
            thought, raw_action, actions = UITarsHelper.parse_response(
                response, width, height, model_name=self.model_name
            )
            
            # 记录历史
            self.history.append({
                "thought": thought,
                "raw_action": raw_action
            })
            
            # 添加推理记录
            self._add_react(thought, raw_action, {}, step_index)
            
            return actions
            
        except Exception as e:
            logger.error(f"解析失败: {e}")
            return [{"type": "retry", "params": {"error": f"解析失败: {e}"}}]

    def _build_messages(self, image_data_url: str) -> List[Dict]:
        """为模型构建消息"""
        system_prompt = PROMPT_TEMPLATE.format(
            language="Chinese",
            instruction=self.task_description
        )
        
        messages = [
            {"role": "user", "content": system_prompt}
        ]
        
        # 添加历史记录
        # 格式:
        # User: ...
        # Assistant: Thought: ... \n Action: ...
        # ...
        # User: <当前图像>
        
        for record in self.history:
            if record['thought'] and record['raw_action']:
                content = f"Thought: {record['thought']}\nAction: {record['raw_action']}"
                messages.append({"role": "assistant", "content": content})
                
        # 添加当前观察
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_url}}
            ]
        })
        
        return messages
