#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UI-TARS 自动化框架包
"""

# 首先设置日志
from .logger import setup_logging

# 然后导入主要模块
from .framework import UITarsAutomationFramework
from .config import ExecutionConfig, ActionResult
from .action_parser import ActionParser
from .data_manager import DataManager
from .coordinate_processor import CoordinateProcessor

__version__ = "1.0.0"
__all__ = [
    "UITarsAutomationFramework",
    "ExecutionConfig", 
    "ActionResult",
    "ActionParser",
    "DataManager",
    "CoordinateProcessor",
    "setup_logging"
]
