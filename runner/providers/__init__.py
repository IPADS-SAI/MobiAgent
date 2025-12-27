# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

"""
Provider包，包含不同模型的任务适配器
"""

from .uitars.uitars_task import UITARSTask
from .mobiagent.mobile_task import MobiAgentStepTask

__all__ = [
    'MobiAgentTask',
    'UITARSTask',
    'MobiAgentStepTask',
]
