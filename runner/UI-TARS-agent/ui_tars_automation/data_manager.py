#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据管理模块 - 负责保存截图、XML、操作记录等数据
"""

import os
import json
import time
import datetime
from typing import Dict, Any, Optional
from .config import ExecutionConfig

class DataManager:
    """数据管理器"""
    
    def __init__(self, config: ExecutionConfig, task_description: str):
        self.config = config
        self.task_description = task_description
        self.task_start_time = datetime.datetime.now()
        self.task_dir = None
        self.current_step = 0
        
        if config.save_data:
            self._create_task_directory()
    
    def _create_task_directory(self):
        """创建任务目录"""
        # 生成任务目录名：基于时间戳和任务描述前几个字
        timestamp = self.task_start_time.strftime("%Y%m%d_%H%M%S")
        task_name = "".join(c for c in self.task_description[:20] if c.isalnum() or c in ('_', '-'))
        if not task_name:
            task_name = "task"
        
        self.task_dir = os.path.join(
            self.config.data_base_dir,
            f"{timestamp}_{task_name}"
        )
        
        # 创建目录结构
        os.makedirs(self.task_dir, exist_ok=True)
        
        # 保存任务基本信息
        task_info = {
            "task_description": self.task_description,
            "start_time": self.task_start_time.isoformat(),
            "config": {
                "model_base_url": self.config.model_base_url,
                "model_name": self.config.model_name,
                "max_steps": self.config.max_steps,
                "language": self.config.language
            }
        }
        
        with open(os.path.join(self.task_dir, "task_info.json"), "w", encoding="utf-8") as f:
            json.dump(task_info, f, ensure_ascii=False, indent=2)
    
    def start_new_step(self, step_number: int):
        """开始新的步骤"""
        self.current_step = step_number
        if self.config.save_data:
            step_dir = os.path.join(self.task_dir, str(step_number))
            os.makedirs(step_dir, exist_ok=True)
    
    def save_screenshot(self, device, step_number: int) -> Optional[str]:
        """保存截图"""
        if not self.config.save_data or not self.config.save_screenshots:
            return None
        
        screenshot_path = os.path.join(
            self.task_dir, 
            str(step_number), 
            f"screenshot_{step_number}.jpg"
        )
        
        try:
            device.screenshot(screenshot_path)
            return screenshot_path
        except Exception as e:
            print(f"保存截图失败: {e}")
            return None
    
    def save_xml(self, device, step_number: int) -> Optional[str]:
        """保存XML层次结构"""
        if not self.config.save_data or not self.config.save_xml:
            return None
        
        xml_path = os.path.join(
            self.task_dir,
            str(step_number),
            f"hierarchy_{step_number}.xml"
        )
        
        try:
            hierarchy = device.dump_hierarchy()
            with open(xml_path, "w", encoding="utf-8") as f:
                f.write(hierarchy)
            return xml_path
        except Exception as e:
            print(f"保存XML失败: {e}")
            return None
    
    def save_step_data(self, step_number: int, step_data: Dict[str, Any]):
        """保存单步数据"""
        if not self.config.save_data:
            return
        
        step_file = os.path.join(
            self.task_dir,
            str(step_number),
            f"step_{step_number}.json"
        )
        
        try:
            with open(step_file, "w", encoding="utf-8") as f:
                json.dump(step_data, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            print(f"保存步骤数据失败: {e}")
    
    def save_execution_summary(self, execution_data: Dict[str, Any]):
        """保存执行总结"""
        if not self.config.save_data:
            return
        
        # 保存actions.json格式的数据
        actions_data = {
            "task_description": self.task_description,
            "start_time": self.task_start_time.isoformat(),
            "end_time": datetime.datetime.now().isoformat(),
            "action_count": execution_data.get("total_steps", 0),
            "success": execution_data.get("success", False),
            "actions": []
        }
        
        # 转换action历史为标准格式
        for action in execution_data.get("action_history", []):
            action_record = {
                "step": action["step"],
                "thought": action["thought"],
                "raw_action": action["raw_action"],
                "action_type": action["parsed_action"]["action_type"],
                "action_params": action["parsed_action"]["action_params"],
                "result": {
                    "success": action["result"].success,
                    "message": action["result"].message,
                    "error": action["result"].error
                },
                "screenshot_path": action.get("screenshot_path"),
                "xml_path": action.get("xml_path")
            }
            actions_data["actions"].append(action_record)
        
        # 保存actions.json
        actions_path = os.path.join(self.task_dir, "actions.json")
        with open(actions_path, "w", encoding="utf-8") as f:
            json.dump(actions_data, f, ensure_ascii=False, indent=2)
        
        # 保存react.json格式的数据（简化版）
        react_data = []
        for action in execution_data.get("action_history", []):
            react_record = {
                "reasoning": action["thought"],
                "function": {
                    "name": action["parsed_action"]["action_type"],
                    "parameters": action["parsed_action"]["action_params"]
                },
                "action_index": action["step"]
            }
            react_data.append(react_record)
        
        react_path = os.path.join(self.task_dir, "react.json")
        with open(react_path, "w", encoding="utf-8") as f:
            json.dump(react_data, f, ensure_ascii=False, indent=2)
        
        print(f"执行数据已保存到: {self.task_dir}")
    
    def get_task_directory(self) -> Optional[str]:
        """获取任务目录路径"""
        return self.task_dir
