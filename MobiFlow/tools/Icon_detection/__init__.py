"""
图标检测工具包
提供基于OpenCV模板匹配的图标检测功能
"""

from .icon_detector import IconDetector, IconPathResolver
from .config import IconDetectionConfig, get_default_config, set_default_config
from .icon_detection import (
    IconDetectionService,
    get_icon_detection_service, 
    detect_icons_simple,
    detect_single_icon
)

__version__ = "1.0.0"
__all__ = [
    # 核心检测器
    'IconDetector',
    'IconPathResolver',
    
    # 配置管理
    'IconDetectionConfig',
    'get_default_config',
    'set_default_config',
    
    # 服务接口
    'IconDetectionService',
    'get_icon_detection_service',
    
    # 简化接口
    'detect_icons_simple',
    'detect_single_icon',
]
