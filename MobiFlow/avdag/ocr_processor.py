"""
OCR处理器兼容性包装器
为了向后兼容，这个文件将导入重定向到新的utils.advanced_ocr模块
"""

# 发出废弃警告
import warnings
warnings.warn(
    "avdag.ocr_processor已经被废弃，请改用utils.advanced_ocr模块",
    DeprecationWarning,
    stacklevel=2
)

# 重新导出所有函数和类，保持兼容性
try:
    import sys
    import os
    
    # 添加项目根目录到路径
    current_dir = os.path.dirname(__file__)
    project_root = os.path.join(current_dir, "..", "..")
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from utils.advanced_ocr import (
        ProcessedText,
        AdvancedOCRProcessor as OCRProcessor,
        get_advanced_ocr_processor as get_ocr_processor,
        extract_text_from_xml,
        extract_text_from_xml_simple,
        create_frame_ocr_function,
        create_frame_texts_function,
        create_standard_ocr_functions,
        extract_text_from_image,
        match_text_in_frame,
        process_frame_text,
        smart_text_search
    )
    
    # 为了完全兼容，保留原有的函数名
    def extract_text_from_xml_simple_regex(xml_content: str) -> ProcessedText:
        """兼容性函数，调用新的extract_text_from_xml_simple"""
        return extract_text_from_xml_simple(xml_content)
    
    # 兼容性别名
    OCRText = ProcessedText
    
except ImportError as e:
    # 如果新模块不可用，提供最小的回退实现
    from dataclasses import dataclass
    from typing import List, Dict, Any
    
    @dataclass
    class ProcessedText:
        original: str = ''
        cleaned: str = ''
        no_spaces: str = ''
        words: List[str] = None
        chars: List[str] = None
        
        def __post_init__(self):
            if self.words is None:
                self.words = []
            if self.chars is None:
                self.chars = []
    
    class OCRProcessor:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"新的OCR模块不可用: {e}")
    
    def get_ocr_processor(*args, **kwargs):
        raise ImportError(f"新的OCR模块不可用: {e}")
    
    def extract_text_from_xml(xml_content: str) -> ProcessedText:
        return ProcessedText()
    
    def extract_text_from_xml_simple(xml_content: str) -> ProcessedText:
        return ProcessedText()
    
    def create_standard_ocr_functions():
        def dummy_ocr(frame): return None
        def dummy_texts(frame): return []
        return dummy_ocr, dummy_texts
    
    # 其他函数的回退实现
    def extract_text_from_image(image_path: str) -> str:
        return ""
    
    def match_text_in_frame(frame: Dict[str, Any], keyword: str) -> bool:
        return False
    
    def process_frame_text(frame: Dict[str, Any]) -> ProcessedText:
        return ProcessedText()
    
    def smart_text_search(text: str, keyword: str) -> bool:
        return False
    
    # 兼容性别名
    OCRText = ProcessedText
    extract_text_from_xml_simple_regex = extract_text_from_xml_simple


# 导出所有兼容性接口
__all__ = [
    "ProcessedText", "OCRText", "OCRProcessor", "get_ocr_processor",
    "extract_text_from_xml", "extract_text_from_xml_simple", "extract_text_from_xml_simple_regex",
    "create_frame_ocr_function", "create_frame_texts_function", "create_standard_ocr_functions",
    "extract_text_from_image", "match_text_in_frame", "process_frame_text", "smart_text_search"
]