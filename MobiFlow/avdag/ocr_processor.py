"""
OCR处理模块 - 为验证系统提供图像文字识别功能

此模块封装了app_trajectory_analyzer的OCR引擎，提供统一的文字识别接口。
支持PaddleOCR和Tesseract两种引擎，可根据需要选择或自动切换。
"""

from __future__ import annotations
import os
import sys
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import threading

from .logger import get_ocr_logger

# 动态添加app_trajectory_analyzer路径
def _add_ocr_path():
    """添加OCR引擎路径到Python搜索路径"""
    current_dir = os.path.dirname(__file__)
    tools_dir = os.path.join(current_dir, "..", "tools", "app_trajectory_analyzer", "src")
    if os.path.exists(tools_dir) and tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)

# 尝试导入OCR引擎
_add_ocr_path()

try:
    from analyzer.ocr_engine import OCREngine
    _ocr_available = True
except ImportError:
    _ocr_available = False


@dataclass
class ProcessedText:
    """OCR处理后的文本结果"""
    original: str          # 原始OCR文本
    cleaned: str          # 清理后的文本（移除特殊符号）
    no_spaces: str        # 无空格版本（用于连续匹配）
    words: List[str]      # 分词结果
    chars: List[str]      # 字符列表


class OCRProcessor:
    """OCR处理器，提供图像文字识别和文本处理功能"""

    def __init__(self, use_paddle: bool = True, lang: str = "chi_sim+eng"):
        """初始化OCR处理器"""
        self.use_paddle = use_paddle
        self.lang = lang
        self._engine: Optional[Any] = None
        self._engine_paddle: Optional[Any] = None
        self._engine_tess: Optional[Any] = None
        self._available = _ocr_available
        self._lock = threading.Lock()
        self._cache_words: Dict[str, List[str]] = {}
        self._logger = get_ocr_logger()

        if self._available:
            try:
                paddle_success = False
                tesseract_success = False
                
                # 尝试初始化 PaddleOCR
                try:
                    self._engine_paddle = OCREngine(use_paddle=True, lang=self.lang)
                    paddle_success = True
                    self._logger.info("PaddleOCR引擎初始化成功")
                except Exception as e:
                    self._logger.warning(f"PaddleOCR引擎初始化失败: {e}")
                    self._engine_paddle = None
                    
                # 尝试初始化 Tesseract
                try:
                    self._engine_tess = OCREngine(use_paddle=False, lang=self.lang)
                    tesseract_success = True
                    self._logger.info("Tesseract引擎初始化成功")
                except Exception as e:
                    self._logger.warning(f"Tesseract引擎初始化失败: {e}")
                    self._engine_tess = None
                    
                # 设置主引擎：优先使用PaddleOCR，如果失败则使用Tesseract
                if self.use_paddle and paddle_success:
                    self._engine = self._engine_paddle
                    self._logger.info("使用PaddleOCR作为主引擎")
                elif tesseract_success:
                    self._engine = self._engine_tess
                    self._logger.info("使用Tesseract作为主引擎")
                elif paddle_success:
                    self._engine = self._engine_paddle
                    self._logger.info("回退到PaddleOCR作为主引擎")
                else:
                    self._logger.error("所有OCR引擎初始化失败")
                    self._available = False
                    self._engine = None
                    
            except Exception as e:
                self._logger.error(f"初始化OCR引擎失败: {e}")
                self._available = False

    def is_available(self) -> bool:
        """检查OCR功能是否可用"""
        return self._available and (self._engine_paddle is not None or self._engine_tess is not None)

    def process_text(self, raw_text: str) -> ProcessedText:
        """将原始文本标准化并生成多视图便于匹配"""
        if not raw_text or not raw_text.strip():
            return ProcessedText(original='', cleaned='', no_spaces='', words=[], chars=[])

        # 正规化：全半角、大小写、常见混淆字符
        def to_half_width(s: str) -> str:
            res = []
            for ch in s:
                code = ord(ch)
                if code == 0x3000:
                    code = 32
                elif 0xFF01 <= code <= 0xFF5E:
                    code -= 0xFEE0
                res.append(chr(code))
            return ''.join(res)

        def normalize_confusions(s: str) -> str:
            mapping = {
                'Ｉ': 'I', 'Ｌ': 'L', 'Ｏ': 'O', 'Ｓ': 'S', 'Ｂ': 'B',
                '０': '0', '１': '1', '２': '2', '５': '5', '６': '6', '８': '8', '９': '9',
                # 常见 OCR 易混：
                'O': '0', 'o': '0', 'l': '1', 'I': '1', '丨': '1', '｜': '1',
                'Z': '2', 'S': '5', 'B': '8',
            }
            return ''.join(mapping.get(c, c) for c in s)

        # 1. 清理文本：保留中文、字母、数字、空格
        raw_text_norm = normalize_confusions(to_half_width(raw_text)).casefold()
        cleaned = re.sub(r'[^\u4e00-\u9fff\w\s]', ' ', raw_text_norm)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # 2. 无空格版本：用于连续匹配
        no_spaces = re.sub(r'\s+', '', cleaned)
        
        # 3. 提取词语：按空格分割
        words = [w.strip() for w in cleaned.split() if w.strip()]
        
        # 4. 提取字符
        chars = list(no_spaces)
        return ProcessedText(original=raw_text, cleaned=cleaned, no_spaces=no_spaces, words=words, chars=chars)

    def smart_text_contains(self, processed_text: ProcessedText, keyword: str) -> bool:
        """
        智能文本匹配，支持多种匹配策略
        
        Args:
            processed_text: 处理后的文本
            keyword: 要搜索的关键词
            
        Returns:
            bool: 是否匹配
        """
        if not keyword or not processed_text:
            return False
        
        def to_half_width(s: str) -> str:
            res = []
            for ch in s:
                code = ord(ch)
                if code == 0x3000:
                    code = 32
                elif 0xFF01 <= code <= 0xFF5E:
                    code -= 0xFEE0
                res.append(chr(code))
            return ''.join(res)

        def normalize_confusions(s: str) -> str:
            mapping = {
                'Ｏ': 'O', 'ｏ': 'o', 'Ｉ': 'I', 'Ｌ': 'L', 'Ｓ': 'S', 'Ｂ': 'B',
                '０': '0', '１': '1', '２': '2', '５': '5', '６': '6', '８': '8', '９': '9',
                'O': '0', 'o': '0', 'l': '1', 'I': '1', '丨': '1', '｜': '1', 'Z': '2', 'S': '5', 'B': '8',
            }
            return ''.join(mapping.get(c, c) for c in s)

        # 处理关键词，应用相同的正规化
        keyword_norm = normalize_confusions(to_half_width(keyword)).casefold()
        keyword_clean = re.sub(r'[^\u4e00-\u9fff\w\s]', ' ', keyword_norm)
        keyword_clean = re.sub(r'\s+', ' ', keyword_clean).strip()
        keyword_no_spaces = re.sub(r'\s+', '', keyword_clean)
        
        # 匹配策略 1: 精确匹配（带空格）
        if keyword_clean in processed_text.cleaned:
            return True
            
        # 匹配策略 2: 连续匹配（无空格）
        if keyword_no_spaces in processed_text.no_spaces:
            return True
            
        # 匹配策略 3: 分词匹配
        keyword_words = [w.strip() for w in keyword_clean.split() if w.strip()]
        if keyword_words and all(any(kw in word for word in processed_text.words) for kw in keyword_words):
            return True
            
        # 匹配策略 4: 模糊匹配（80%相似度）
        try:
            from difflib import SequenceMatcher
            similarity = SequenceMatcher(None, keyword_no_spaces, processed_text.no_spaces).ratio()
            if similarity >= 0.8:
                return True
        except ImportError:
            pass
            
        return False

    def extract_text_from_image(self, image_path: str, enable_hybrid: bool = True) -> Tuple[str, Optional[str]]:
        """
        从图像中提取文字
        
        Args:
            image_path: 图像文件路径
            enable_hybrid: 是否启用混合识别（Paddle + Tesseract）
            
        Returns:
            Tuple[str, Optional[str]]: (主识别结果, 备用识别结果)
        """
        if not self.is_available():
            self._logger.warning("OCR引擎不可用")
            return "", None
            
        if not os.path.exists(image_path):
            self._logger.error(f"图像文件不存在: {image_path}")
            return "", None
            
        try:
            # 验证图像文件
            with Image.open(image_path) as img:
                img.verify()
        except Exception as e:
            self._logger.error(f"打开图片失败: {image_path}: {e}")
            return "", None
            
        primary_text = ""
        secondary_text = None
        
        # 使用主引擎识别
        if self._engine:
            try:
                result = self._engine.run(image_path)
                primary_text = result.get_text() if result else ""
                self._logger.debug(f"主引擎识别结果: {len(primary_text)} 字符")
            except Exception as e:
                self._logger.error(f"主引擎识别失败: {e}")
                
        # 混合识别：使用备用引擎
        if enable_hybrid and self._engine_paddle and self._engine_tess:
            backup_engine = self._engine_tess if self._engine == self._engine_paddle else self._engine_paddle
            try:
                result = backup_engine.run(image_path)
                secondary_text = result.get_text() if result else ""
                self._logger.debug(f"备用引擎识别结果: {len(secondary_text)} 字符")
                
                # 如果主引擎失败但备用引擎成功，使用备用结果
                if not primary_text and secondary_text:
                    primary_text = secondary_text
                    self._logger.info("使用备用引擎结果作为主结果")
                    
            except Exception as e:
                self._logger.warning(f"备用引擎识别失败: {e}")
                
        return primary_text, secondary_text

    def recognize_image(self, image_path: str) -> Optional[ProcessedText]:
        """
        识别图像并返回处理后的文本结果（兼容性方法）
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            ProcessedText: 处理后的文本结果，失败时返回None
        """
        text, _ = self.extract_text_from_image(image_path)
        if text:
            return self.process_text(text)
        return None

    def get_word_list(self, image_path: str) -> List[str]:
        """
        从图像中获取词语列表（兼容性方法）
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            List[str]: 词语列表
        """
        text, backup_text = self.extract_text_from_image(image_path)
        words = []
        
        if text:
            processed = self.process_text(text)
            words.extend(processed.words)
            if processed.cleaned:
                words.append(processed.cleaned)
            if processed.no_spaces:
                words.append(processed.no_spaces)
                
        if backup_text and backup_text != text:
            processed_backup = self.process_text(backup_text)
            words.extend(processed_backup.words)
            if processed_backup.cleaned:
                words.append(processed_backup.cleaned)
            if processed_backup.no_spaces:
                words.append(processed_backup.no_spaces)
                
        return list(set(words))

    def extract_xml_text(self, xml_content: str) -> str:
        """从XML内容中提取可视文本"""
        if not xml_content:
            return ""
            
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            self._logger.warning(f"XML解析失败: {e}")
            return ""
            
        texts = []
        
        def extract_text_recursive(element):
            """递归提取元素文本"""
            # 获取元素的text属性
            text_attr = element.get('text', '').strip()
            if text_attr:
                texts.append(text_attr)
                
            # 获取元素的content-desc属性（Android无障碍描述）
            content_desc = element.get('content-desc', '').strip()
            if content_desc and content_desc != text_attr:
                texts.append(content_desc)
                
            # 递归处理子元素
            for child in element:
                extract_text_recursive(child)
                
        extract_text_recursive(root)
        return ' '.join(texts)

    def process_frame_text(self, frame: Dict[str, Any]) -> ProcessedText:
        """
        处理帧中的所有文本信息
        
        Args:
            frame: 包含图像、XML等信息的帧数据
            
        Returns:
            ProcessedText: 处理后的综合文本
        """
        all_texts = []
        
        # 1. 直接文本信息
        if 'text' in frame and frame['text']:
            all_texts.append(str(frame['text']))
            
        # 2. XML文本提取
        if 'xml_text' in frame and frame['xml_text']:
            xml_text = self.extract_xml_text(frame['xml_text'])
            if xml_text:
                all_texts.append(xml_text)
                
        # 3. OCR文本提取
        if 'image' in frame and frame['image'] and os.path.exists(frame['image']):
            ocr_text, _ = self.extract_text_from_image(frame['image'])
            if ocr_text:
                all_texts.append(ocr_text)
                
        # 4. 任务描述和推理文本
        for field in ['task_description', 'reasoning', 'action']:
            if field in frame and frame[field]:
                all_texts.append(str(frame[field]))
                
        # 合并所有文本
        combined_text = ' '.join(all_texts)
        return self.process_text(combined_text)

    def match_keyword_in_frame(self, frame: Dict[str, Any], keyword: str, enable_ocr: bool = True) -> bool:
        """
        在帧中搜索关键词
        
        Args:
            frame: 帧数据
            keyword: 要搜索的关键词
            enable_ocr: 是否启用OCR识别
            
        Returns:
            bool: 是否找到关键词
        """
        if not keyword:
            return False
            
        # 首先在现有文本字段中搜索
        text_fields = ['text', 'task_description', 'reasoning']
        for field in text_fields:
            if field in frame and frame[field]:
                processed = self.process_text(str(frame[field]))
                if self.smart_text_contains(processed, keyword):
                    self._logger.debug(f"在字段 {field} 中找到关键词: {keyword}")
                    return True
                    
        # 在XML文本中搜索
        if 'xml_text' in frame and frame['xml_text']:
            xml_text = self.extract_xml_text(frame['xml_text'])
            if xml_text:
                processed = self.process_text(xml_text)
                if self.smart_text_contains(processed, keyword):
                    self._logger.debug(f"在XML文本中找到关键词: {keyword}")
                    return True
                    
        # 使用OCR在图像中搜索
        if enable_ocr and 'image' in frame and frame['image']:
            if os.path.exists(frame['image']):
                ocr_text, backup_text = self.extract_text_from_image(frame['image'])
                
                # 在主OCR结果中搜索
                if ocr_text:
                    processed = self.process_text(ocr_text)
                    if self.smart_text_contains(processed, keyword):
                        self._logger.debug(f"在OCR文本中找到关键词: {keyword}")
                        return True
                        
                # 在备用OCR结果中搜索
                if backup_text and backup_text != ocr_text:
                    processed = self.process_text(backup_text)
                    if self.smart_text_contains(processed, keyword):
                        self._logger.debug(f"在备用OCR文本中找到关键词: {keyword}")
                        return True
                        
        return False

    def get_text_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的相似度"""
        if not text1 or not text2:
            return 0.0
            
        processed1 = self.process_text(text1)
        processed2 = self.process_text(text2)
        
        try:
            from difflib import SequenceMatcher
            return SequenceMatcher(None, processed1.no_spaces, processed2.no_spaces).ratio()
        except ImportError:
            # 简单的字符重叠度计算
            chars1 = set(processed1.no_spaces)
            chars2 = set(processed2.no_spaces)
            if not chars1 and not chars2:
                return 1.0
            if not chars1 or not chars2:
                return 0.0
            return len(chars1 & chars2) / len(chars1 | chars2)


# 全局OCR处理器实例（单例模式）
_global_ocr_processor: Optional[OCRProcessor] = None
_ocr_lock = threading.Lock()

def get_ocr_processor(use_paddle: bool = True, lang: str = "chi_sim+eng") -> OCRProcessor:
    """获取全局OCR处理器实例"""
    global _global_ocr_processor
    
    with _ocr_lock:
        if _global_ocr_processor is None:
            _global_ocr_processor = OCRProcessor(use_paddle=use_paddle, lang=lang)
    
    return _global_ocr_processor


def get_global_ocr_processor() -> OCRProcessor:
    """获取全局OCR处理器实例（兼容性函数）"""
    return get_ocr_processor()


def extract_text_from_xml(xml_content: str) -> ProcessedText:
    """
    从XML内容中提取文字（兼容性函数）
    
    Args:
        xml_content: XML文件内容字符串
        
    Returns:
        ProcessedText: 处理后的文本结果
    """
    processor = get_ocr_processor()
    xml_text = processor.extract_xml_text(xml_content)
    return processor.process_text(xml_text)


def extract_text_from_xml_simple(xml_content: str) -> ProcessedText:
    """
    从XML内容中使用正则表达式简单提取文字（备用方案）
    
    Args:
        xml_content: XML文件内容字符串
        
    Returns:
        ProcessedText: 处理后的文本结果
    """
    if not xml_content or not xml_content.strip():
        return ProcessedText(
            original='',
            cleaned='',
            no_spaces='',
            words=[],
            chars=[]
        )
    
    text_contents = []
    
    # 提取text属性
    text_pattern = r'text="([^"]*[a-zA-Z\u4e00-\u9fff]+[^"]*)"'
    text_matches = re.findall(text_pattern, xml_content)
    text_contents.extend([t.strip() for t in text_matches if t.strip()])
    
    # 提取content-desc属性
    desc_pattern = r'content-desc="([^"]*[a-zA-Z\u4e00-\u9fff]+[^"]*)"'
    desc_matches = re.findall(desc_pattern, xml_content)
    text_contents.extend([d.strip() for d in desc_matches if d.strip()])
    
    # 提取hint属性
    hint_pattern = r'hint="([^"]*[a-zA-Z\u4e00-\u9fff]+[^"]*)"'
    hint_matches = re.findall(hint_pattern, xml_content)
    text_contents.extend([h.strip() for h in hint_matches if h.strip()])
    
    # 去重
    unique_texts = list(set([t for t in text_contents if t]))
    combined_text = ' '.join(unique_texts)
    
    # 使用OCRProcessor的文本处理功能
    processor = get_ocr_processor()
    return processor.process_text(combined_text)


def create_frame_ocr_function(processor: OCRProcessor) -> callable:
    """
    创建适用于Frame的OCR函数，用于集成到VerifierOptions中
    
    Args:
        processor: OCR处理器实例
        
    Returns:
        callable: 接受Frame参数的OCR函数
    """
    def frame_ocr(frame: Dict[str, Any]) -> Optional[str]:
        """
        从Frame中提取并识别图像文字
        
        Args:
            frame: 包含图像路径的Frame字典
            
        Returns:
            str: 识别的文字，失败时返回XML文本或None
        """
        # 获取图像路径
        image_path = frame.get("image")
        if not image_path or not os.path.exists(image_path):
            # 退化到改进的XML提取
            xml_text = frame.get("xml_text", "")
            if xml_text:
                xml_processed = extract_text_from_xml(xml_text)
                if xml_processed.cleaned:
                    frame['_xml_processed'] = xml_processed
                    processor._logger.info(f"图像不可用，使用改进XML提取: {xml_processed.cleaned[:100]}...")
                    processor._logger.debug(f"XML提取词语数: {len(xml_processed.words)}")
                    return f"{xml_processed.cleaned} {xml_processed.no_spaces} {' '.join(xml_processed.words)}"
                else:
                    processor._logger.warning("图像不可用且XML提取失败")
                    return xml_text[:200] if xml_text else None
            return None
        
        # 使用OCR识别
        ocr_text, backup_text = processor.extract_text_from_image(image_path)
        xml_text = frame.get("xml_text", "")
        merged_parts: List[str] = []
        
        if ocr_text:
            # 处理OCR文本
            processed = processor.process_text(ocr_text)
            frame['_ocr_processed'] = processed
            merged_parts.extend([processed.cleaned, processed.no_spaces] + processed.words)
            processor._logger.debug(f"识别图像 {os.path.basename(image_path)} -> 词语数: {len(processed.words)}")
            
        # 融合XML文本
        if xml_text:
            xml_processed = extract_text_from_xml(xml_text)
            if xml_processed.cleaned:
                frame['_xml_processed'] = xml_processed
                merged_parts.extend([xml_processed.cleaned, xml_processed.no_spaces] + xml_processed.words)
                processor._logger.debug(f"融合XML文本 -> 词语数: {len(xml_processed.words)}")
                
        if merged_parts:
            return ' '.join(list(dict.fromkeys([p for p in merged_parts if p])))
            
        # 全部失败
        processor._logger.warning(f"图像 {os.path.basename(image_path)} 未识别到文字且无可用XML")
        return None
    
    return frame_ocr


def create_frame_texts_function(processor: OCRProcessor) -> callable:
    """
    创建适用于Frame的文本列表提取函数
    
    Args:
        processor: OCR处理器实例
        
    Returns:
        callable: 接受Frame参数，返回文本列表的函数
    """
    def frame_texts(frame: Dict[str, Any]) -> List[str]:
        """
        从Frame中提取文本列表，优先使用OCR识别
        
        Args:
            frame: 包含图像路径的Frame字典
            
        Returns:
            List[str]: 文本列表
        """
        # 获取图像路径
        image_path = frame.get("image")
        if not image_path or not os.path.exists(image_path):
            # 退化到改进的XML提取
            xml_text = frame.get("xml_text", "")
            if xml_text:
                xml_processed = extract_text_from_xml(xml_text)
                if xml_processed.words:
                    frame['_xml_processed'] = xml_processed
                    # 返回多种格式的文本
                    result_texts = xml_processed.words.copy()
                    if xml_processed.cleaned:
                        result_texts.append(xml_processed.cleaned)
                    if xml_processed.no_spaces:
                        result_texts.append(xml_processed.no_spaces)
                    processor._logger.info(f"图像不可用，使用改进XML提取，得到 {len(result_texts)} 个文本片段")
                    return list(set(result_texts))
                else:
                    # 降级到传统方式
                    texts = frame.get("xml_texts", [])
                    if not texts and xml_text:
                        texts = [xml_text]
                    processor._logger.warning("XML解析失败，使用原始XML文本")
                    return texts
            else:
                texts = frame.get("xml_texts", [])
                processor._logger.warning("无图像也无XML文本")
                return texts
        
        # 使用OCR获取文本
        ocr_text, backup_text = processor.extract_text_from_image(image_path)
        words = []
        
        if ocr_text:
            processed = processor.process_text(ocr_text)
            words.extend(processed.words)
            if processed.cleaned:
                words.append(processed.cleaned)
            if processed.no_spaces:
                words.append(processed.no_spaces)
                
        # 融合 XML
        xml_text = frame.get("xml_text", "")
        if xml_text:
            xml_processed = extract_text_from_xml(xml_text)
            if xml_processed.words:
                frame['_xml_processed'] = xml_processed
                words.extend(xml_processed.words)
                if xml_processed.cleaned:
                    words.append(xml_processed.cleaned)
                if xml_processed.no_spaces:
                    words.append(xml_processed.no_spaces)
                    
        if words:
            processor._logger.debug(f"从图像 {os.path.basename(image_path)} (含XML融合) 提取 {len(set(words))} 个文本片段")
            return list(set(words))
            
        # 兜底
        texts = frame.get("xml_texts", [])
        if not texts and xml_text:
            texts = [xml_text]
        if texts:
            processor._logger.info(f"使用兜底XML文本 {len(texts)} 段")
        else:
            processor._logger.warning("无可用文本")
        return texts
    
    return frame_texts


def create_standard_ocr_functions() -> tuple[callable, callable]:
    """
    创建标准的OCR函数对，用于快速集成
    
    Returns:
        tuple: (frame_ocr_function, frame_texts_function)
    """
    processor = get_global_ocr_processor()
    return (
        create_frame_ocr_function(processor),
        create_frame_texts_function(processor)
    )


# 便捷函数
def extract_text_from_image(image_path: str) -> str:
    """便捷函数：从图像提取文本"""
    processor = get_ocr_processor()
    text, _ = processor.extract_text_from_image(image_path)
    return text

def match_text_in_frame(frame: Dict[str, Any], keyword: str) -> bool:
    """便捷函数：在帧中匹配文本"""
    processor = get_ocr_processor()
    return processor.match_keyword_in_frame(frame, keyword)

def process_frame_text(frame: Dict[str, Any]) -> ProcessedText:
    """便捷函数：处理帧文本"""
    processor = get_ocr_processor()
    return processor.process_frame_text(frame)


# 导出的公共接口
__all__ = [
    "ProcessedText",
    "OCRProcessor", 
    "create_frame_ocr_function",
    "create_frame_texts_function",
    "get_global_ocr_processor",
    "get_ocr_processor",
    "create_standard_ocr_functions",
    "extract_text_from_xml",
    "extract_text_from_xml_simple",
    "extract_text_from_image",
    "match_text_in_frame",
    "process_frame_text"
]
