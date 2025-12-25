# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

import logging
from typing import Dict, Optional


class TaskManager:
    """
    任务管理器，负责根据provider类型创建和执行相应的任务
    """
    
    def __init__(
        self,
        provider: str,
        task_description: str,
        device,
        data_dir: str,
        device_type: str = "Android",
        max_steps: int = 40,
        draw: bool = False,
        **kwargs
    ):
        """
        初始化任务管理器
        
        Args:
            provider: 模型提供者名称 ("mobiagent", "uitars", 等)
            task_description: 任务描述
            device: 设备对象
            data_dir: 数据保存目录
            device_type: 设备类型
            max_steps: 最大步骤数
            draw: 是否在截图上绘制操作
            log_level: 日志级别
            **kwargs: 传递给具体任务类的其他参数
        """
        self.provider = provider
        self.task_description = task_description
        self.device = device
        self.data_dir = data_dir
        self.device_type = device_type
        self.max_steps = max_steps
        self.kwargs = kwargs
        
        # 导入provider
        self.task_map = self._get_task_map()
        
        # 创建任务实例
        if provider not in self.task_map:
            raise ValueError(
                f"Unknown provider: {provider}. "
                f"Available providers: {list(self.task_map.keys())}"
            )
        
        task_class = self.task_map[provider]
        self.task = task_class(
            task_description=task_description,
            device=device,
            data_dir=data_dir,
            device_type=device_type,
            max_steps=max_steps,
            draw=draw,
            **kwargs
        )
        
        logging.info(f"TaskManager initialized with provider: {provider}")
    
    def _get_task_map(self) -> Dict:
        """
        获取provider到任务类的映射
        
        Returns:
            provider名称到任务类的字典
        """
        try:
            # from providers.mobiagent_task import MobiAgentTask  # Legacy, removed
            from providers.uitars.uitars_task import UITARSTask
            from providers.mobiagent.mobile_task import MobiAgentStepTask
            from providers.qwen.qwen_task import QwenTask
            from providers.autoglm.autoglm_task import AutoGLMTask
            
            task_map = {
                "mobiagent": MobiAgentStepTask,  # 别名
                "mobiagent_step": MobiAgentStepTask,
                "uitars": UITARSTask,
                "qwen": QwenTask,
                "autoglm": AutoGLMTask,
            }
            
            return task_map
            
        except ImportError as e:
            logging.warning(f"Failed to import some providers: {e}")
            # 返回部分可用的providers
            task_map = {}
            
            try:
                from providers.mobiagent.mobile_task import MobiAgentStepTask
                task_map["mobiagent_step"] = MobiAgentStepTask
            except ImportError:
                pass
            try:
                from providers.uitars.uitars_task import UITARSTask
                task_map["uitars"] = UITARSTask
            except ImportError:
                pass

            try:
                from providers.qwen.qwen_task import QwenTask
                task_map["qwen"] = QwenTask
            except ImportError:
                pass
            
            return task_map
    
    def execute(self) -> Dict:
        """
        执行任务
        
        Returns:
            任务执行结果字典
        """
        logging.info(f"Executing task with provider: {self.provider}")
        result = self.task.execute()
        return result
    
    def get_task_info(self) -> Dict:
        """
        获取任务信息
        
        Returns:
            任务信息字典
        """
        return {
            "provider": self.provider,
            "task_description": self.task_description,
            "device_type": self.device_type,
            "max_steps": self.max_steps,
            "data_dir": self.data_dir
        }
