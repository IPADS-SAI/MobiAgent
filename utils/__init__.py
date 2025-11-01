"""
Utils模块 - 统一的工具集合
包含OCR引擎、图标检测、配置管理、权重管理、高级OCR处理等功能
"""

from .ocr_engine import OCREngine, OCRResult, OCRWord, ocr_image
from .icon_detection import IconDetector, IconPathResolver, IconDetectionService
from .config import OCRConfig, IconDetectionConfig, ToolsConfig, ConfigManager
from .weights_manager import WeightsManager

# 高级OCR处理器
from .advanced_ocr import (
    ProcessedText, AdvancedOCRProcessor, get_advanced_ocr_processor,
    extract_text_from_xml, create_frame_ocr_function, create_frame_texts_function,
    create_standard_ocr_functions, extract_text_from_image, match_text_in_frame,
    process_frame_text, smart_text_search
)

# 统一工具接口 (解决命名冲突，改为 tools_unified)
try:
    from .tools_unified import Tools
    __tools_available = True
except ImportError:
    __tools_available = False

# 导出的公共接口
__all__ = [
    # 基础OCR引擎
    "OCREngine", "OCRResult", "OCRWord", "ocr_image",
    # 图标检测
    "IconDetector", "IconPathResolver", "IconDetectionService",
    # 配置管理
    "OCRConfig", "IconDetectionConfig", "ToolsConfig", "ConfigManager",
    # 权重管理
    "WeightsManager",
    # 高级OCR处理器
    "ProcessedText", "AdvancedOCRProcessor", "get_advanced_ocr_processor",
    "extract_text_from_xml", "create_frame_ocr_function", "create_frame_texts_function",
    "create_standard_ocr_functions", "extract_text_from_image", "match_text_in_frame",
    "process_frame_text", "smart_text_search",
]

# 条件导出 Tools
if __tools_available:
    __all__.append("Tools")