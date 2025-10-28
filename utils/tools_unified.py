"""
工具模块初始化文件
提供统一的工具导入接口
"""

from .ocr_engine import OCREngine, OCRResult, OCRWord, ocr_image
from .icon_detection import (
    IconDetector, 
    IconPathResolver, 
    IconDetectionService,
    get_icon_detection_service,
    detect_icons_simple,
    detect_single_icon
)
from .config import (
    OCRConfig,
    IconDetectionConfig, 
    ToolsConfig,
    ConfigManager,
    get_config_manager,
    get_ocr_config,
    get_icon_detection_config,
    get_weights_dir,
    save_config
)

__all__ = [
    # OCR相关
    'OCREngine',
    'OCRResult', 
    'OCRWord',
    'ocr_image',
    
    # 图标检测相关
    'IconDetector',
    'IconPathResolver',
    'IconDetectionService', 
    'get_icon_detection_service',
    'detect_icons_simple',
    'detect_single_icon',
    
    # 配置相关
    'OCRConfig',
    'IconDetectionConfig',
    'ToolsConfig', 
    'ConfigManager',
    'get_config_manager',
    'get_ocr_config',
    'get_icon_detection_config',
    'get_weights_dir',
    'save_config'
]

# 版本信息
__version__ = '1.0.0'