from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional
import os, glob

import yaml  # type: ignore


@dataclass
class StepRule:
    """步骤规则数据类"""
    name: str  # 步骤名称
    must_have_keywords: List[str]  # 必须包含的关键词列表
    any_of_keywords: Optional[List[str]] = None  # 任选其一的关键词列表
    forbidden_keywords: Optional[List[str]] = None  # 禁止出现的关键词列表
    # 图标约束
    all_of_icons: Optional[List[str]] = None  # 必须全部存在的图标列表
    any_of_icons: Optional[List[str]] = None  # 任选其一的图标列表


@dataclass
class TaskRule:
    """任务规则数据类"""
    task_name: str  # 任务名称标识
    steps: List[StepRule]  # 有序的步骤列表
    min_actions: int = 0  # 最小动作数量要求


def load_tasks_from_dir(rules_dir: str) -> Dict[str, TaskRule]:
    """
    从目录中加载所有任务规则
    
    Args:
        rules_dir: 规则文件目录路径
        
    Returns:
        任务规则字典，键为任务名称，值为任务规则对象
    """
    tasks: Dict[str, TaskRule] = {}
    for path in glob.glob(os.path.join(rules_dir, "*.y*ml")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if not data:
                continue
            
            # 支持单个任务对象转换为列表格式
            if isinstance(data, dict) and "task_name" in data:
                data = [data]
            if not isinstance(data, list):
                continue
            
            # 解析每个任务定义
            for td in data:
                name = td.get("task_name")
                steps_raw = td.get("steps", [])
                min_actions = int(td.get("min_actions", 0))
                steps: List[StepRule] = []
                
                for s in steps_raw:
                    steps.append(StepRule(
                        name=s.get("name"),
                        must_have_keywords=list(s.get("must_have_keywords", [])),
                        any_of_keywords=list(s.get("any_of_keywords", [])) if s.get("any_of_keywords") else None,
                        forbidden_keywords=list(s.get("forbidden_keywords", [])) if s.get("forbidden_keywords") else None,
                        all_of_icons=list(s.get("all_of_icons", [])) if s.get("all_of_icons") else None,
                        any_of_icons=list(s.get("any_of_icons", [])) if s.get("any_of_icons") else None,
                    ))
                
                task = TaskRule(task_name=name, steps=steps, min_actions=min_actions)
                tasks[name] = task
        except Exception:
            # 忽略无法解析的文件
            continue
    return tasks


def load_task_by_name(rules_dir: str, task_name: str) -> TaskRule:
    """
    根据任务名称加载特定任务规则
    
    Args:
        rules_dir: 规则文件目录路径
        task_name: 任务名称
        
    Returns:
        任务规则对象
        
    Raises:
        ValueError: 任务未找到时抛出异常
    """
    tasks = load_tasks_from_dir(rules_dir)
    if task_name not in tasks:
        raise ValueError(f"任务 '{task_name}' 在规则目录中未找到: {rules_dir}")
    return tasks[task_name]
