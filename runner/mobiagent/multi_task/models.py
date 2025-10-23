"""
数据模型定义
包含所有Pydantic模型和数据结构
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from enum import Enum


class ActionType(str, Enum):
    """定义所有可能的用户界面动作"""
    CLICK = "click"
    INPUT = "input"
    SWIPE = "swipe"
    DONE = "done"
    WAIT = "wait"


class ActionPlan(BaseModel):
    """定义一个包含推理、动作和参数的结构化计划"""
    reasoning: str = Field(description="描述执行此动作的思考过程和理由")
    action: ActionType = Field(description="要执行的下一个动作")
    parameters: Dict[str, str] = Field(
        description="执行动作所需要的参数，以键值对形式提供",
        default_factory=dict
    )


class GroundResponse(BaseModel):
    """Grounder返回的坐标或边界框"""
    coordinates: list[int] = Field(
        description="点击坐标 [x, y]",
        default=None
    )
    bbox: list[int] = Field(
        description="边界框 [x1, y1, x2, y2]",
        default=None
    )
    bbox_2d: list[int] = Field(
        description="边界框 [x1, y1, x2, y2]",
        default=None
    )


class Subtask(BaseModel):
    """定义单个子任务的结构"""
    subtask_id: int = Field(description="子任务的唯一标识符")
    app_name: str = Field(description="执行该子任务所需的应用名称")
    package_name: str = Field(description="应用的包名")
    description: str = Field(description="子任务的详细描述")
    artifact_schema: Dict[str, str] = Field(
        description="该子任务需要提取的结构化数据格式，key为字段名，value为字段描述",
        default_factory=dict
    )
    depends_on: List[int] = Field(
        description="该子任务依赖的前置子任务ID列表",
        default_factory=list
    )


class Plan(BaseModel):
    """定义完整的任务规划"""
    task_description: str = Field(description="原始任务描述")
    subtasks: List[Subtask] = Field(description="子任务列表")


class Artifact(BaseModel):
    """定义子任务执行后提取的结构化数据"""
    subtask_id: int = Field(description="对应的子任务ID")
    data: Dict[str, Any] = Field(description="提取的结构化数据")
    success: bool = Field(description="子任务是否成功完成")
    summary: str = Field(description="子任务执行的自然语言总结")


class State(BaseModel):
    """维护整个任务执行过程中的状态"""
    task_description: str = Field(description="原始任务描述")
    plan: Optional[Plan] = Field(default=None, description="当前的任务规划")
    current_subtask_index: int = Field(default=0, description="当前执行到的子任务索引")
    artifacts: Dict[str, Artifact] = Field(default_factory=dict, description="已收集的所有artifacts，key为subtask_id")
    completed_subtasks: List[int] = Field(default_factory=list, description="已完成的子任务ID列表")
