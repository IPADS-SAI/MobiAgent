from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any

from PIL import Image

# 尝试导入日志系统（如果可用）
try:
    import sys
    from pathlib import Path
    # 添加avdag路径以导入日志系统
    current_dir = Path(__file__).resolve()
    avdag_path = current_dir.parent.parent.parent.parent / "avdag"
    if avdag_path.exists():
        sys.path.insert(0, str(avdag_path.parent))
        from avdag.logger import get_ocr_logger
        _has_logger = True
    else:
        _has_logger = False
except ImportError:
    _has_logger = False

def _get_logger():
    """获取日志器，如果不可用则返回None"""
    if _has_logger:
        try:
            return get_ocr_logger()
        except:
            pass
    return None

def _log_info(msg: str):
    """记录信息日志"""
    logger = _get_logger()
    if logger:
        logger.info(msg)
    else:
        print(f"[OCR] {msg}")

def _log_warning(msg: str):
    """记录警告日志"""
    logger = _get_logger()
    if logger:
        logger.warning(msg)
    else:
        print(f"[OCR] {msg}")

def _log_error(msg: str):
    """记录错误日志"""
    logger = _get_logger()
    if logger:
        logger.error(msg)
    else:
        print(f"[OCR] {msg}")

def _log_debug(msg: str):
    """记录调试日志"""
    logger = _get_logger()
    if logger:
        logger.debug(msg)
    else:
        print(f"[OCR] {msg}")

try:
    from paddleocr import PaddleOCR  # type: ignore
    import paddle

    _has_paddle = True
except Exception:  # pragma: no cover
    PaddleOCR = None
    _has_paddle = False

try:
    import pytesseract  # type: ignore
    _has_tesseract = True
    
    # 检测Tesseract是否正确安装
    def _check_tesseract_installation():
        try:
            # 尝试获取Tesseract版本信息
            version = pytesseract.get_tesseract_version()
            _log_info(f"检测到Tesseract版本: {version}")
            return True
        except Exception as e:
            _log_error(f"Tesseract未正确安装或配置: {e}")
            # 尝试自动配置Tesseract路径（Windows）
            if os.name == 'nt':  # Windows
                possible_paths = [
                    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                    r"D:\Program Files\Tesseract-OCR\tesseract.exe",
                    r"D:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        pytesseract.pytesseract.tesseract_cmd = path
                        _log_info(f"设置Tesseract路径: {path}")
                        try:
                            version = pytesseract.get_tesseract_version()
                            _log_info(f"Tesseract配置成功，版本: {version}")
                            return True
                        except Exception:
                            continue
            return False
    
    _has_tesseract = _check_tesseract_installation()
    
except Exception:  # pragma: no cover
    pytesseract = None  # type: ignore
    _has_tesseract = False

# 全局PaddleOCR实例缓存，避免重复初始化和下载模型
_global_paddle_instance = None


@dataclass
class OCRWord:
    """OCR识别的单个词语结果"""
    text: str  # 识别的文字内容
    bbox: Tuple[int, int, int, int]  # 边界框坐标 (x1, y1, x2, y2)
    conf: float  # 置信度分数


@dataclass
class OCRResult:
    """OCR识别的完整结果"""
    words: List[OCRWord]  # 所有识别出的词语列表

    def get_text(self) -> str:
        """获取所有文字内容的拼接字符串"""
        return " ".join([w.text for w in self.words])

    def find(self, keyword: str, fuzzy: bool = True) -> bool:
        """
        在OCR结果中查找关键词
        
        Args:
            keyword: 要查找的关键词
            fuzzy: 是否使用模糊匹配
            
        Returns:
            是否找到关键词
        """
        text = self.get_text()
        if not fuzzy:
            return keyword in text
        try:
            from rapidfuzz import fuzz  # type: ignore
            return fuzz.partial_ratio(keyword, text) >= 80
        except Exception:
            return keyword in text


class OCREngine:
    """OCR文字识别引擎，支持Tesseract和PaddleOCR"""
    
    def __init__(self, lang: str = "chi_sim+eng", use_paddle: Optional[bool] = None):
        """
        初始化OCR引擎
        
        Args:
            lang: Tesseract的语言设置，默认中英文混合
            use_paddle: 是否使用PaddleOCR，None表示自动选择
        """
        self.lang = lang
        if use_paddle is None:
            use_paddle = _has_paddle
        self.use_paddle = use_paddle
        # self.use_paddle = False  # 强制关闭PaddleOCR，避免版本兼容问题
        self._paddle: Optional[Any] = None
        
        # 使用全局单例来避免重复初始化PaddleOCR
        if self.use_paddle and _has_paddle:
            self._paddle = self._get_paddle_instance()

    def _get_paddle_instance(self) -> Optional[Any]:
        """获取全局PaddleOCR实例，如果不存在则创建"""
        global _global_paddle_instance
        if _global_paddle_instance is None:
            try:
                # 判断 Paddle 是否编译了 CUDA
                use_gpu = paddle.device.is_compiled_with_cuda()
                device = "gpu" if use_gpu else "cpu"
                paddle.set_device(device)

                _log_info(f"正在初始化PaddleOCR实例（设备: {device.upper()}，首次使用需要下载模型）...")

                # 注意：3.1.1版本不能同时设置 use_angle_cls 和 use_textline_orientation
                _global_paddle_instance = PaddleOCR(
                    lang="ch",              # 中文模型
                    use_textline_orientation=True      # 推荐启用角度分类
                )
                _log_info(f"PaddleOCR初始化成功（使用{device.upper()}）")

            except Exception as e:
                _log_error(f"PaddleOCR初始化失败: {e}")
                try:
                    _log_info("尝试使用默认参数初始化PaddleOCR（CPU）...")
                    paddle.set_device("cpu")
                    _global_paddle_instance = PaddleOCR(lang="ch")
                    _log_info("PaddleOCR默认参数初始化成功（CPU）")
                except Exception as e2:
                    _log_error(f"PaddleOCR默认参数初始化也失败: {e2}")
                    _global_paddle_instance = None
        return _global_paddle_instance

    def _to_pil(self, img: Any) -> Image.Image:
        """将输入图像转换为PIL图像对象"""
        if isinstance(img, str):
            return Image.open(img).convert("RGB")
        # 可选的numpy数组支持
        try:
            import numpy as np  # type: ignore
            if isinstance(img, np.ndarray):
                from PIL import Image as _Image
                return _Image.fromarray(img)
        except Exception:
            pass
        if isinstance(img, Image.Image):
            return img
        raise TypeError("不支持的图像类型")

    def _resize_image_if_needed(self, img: Any, max_side: int = 4000) -> Any:
        """
        如果图像尺寸超过最大边长限制，则缩放图像
        
        Args:
            img: 输入图像（可以是路径、PIL图像或numpy数组）
            max_side: 最大边长限制
            
        Returns:
            处理后的图像（保持原始类型）
        """
        # 如果是字符串路径，先转换为PIL图像进行尺寸检查
        if isinstance(img, str):
            try:
                pil_img = Image.open(img).convert("RGB")
                w, h = pil_img.size
                if max(w, h) <= max_side:
                    return img  # 尺寸合适，返回原始路径
                
                # 需要缩放，计算新尺寸
                scale = max_side / max(w, h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                
                resized_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                _log_debug(f"图像尺寸从 {w}x{h} 缩放到 {new_w}x{new_h}")
                
                # 转换为numpy数组返回（PaddleOCR支持）
                import numpy as np
                return np.array(resized_img)
            except Exception as e:
                _log_error(f"图像缩放失败: {e}")
                return img
        
        # 如果是PIL图像
        if isinstance(img, Image.Image):
            w, h = img.size
            if max(w, h) <= max_side:
                return img
            
            scale = max_side / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            _log_debug(f"图像尺寸从 {w}x{h} 缩放到 {new_w}x{new_h}")
            return resized_img
        
        # 如果是numpy数组
        try:
            import numpy as np
            if isinstance(img, np.ndarray):
                h, w = img.shape[:2]
                if max(w, h) <= max_side:
                    return img
                
                scale = max_side / max(w, h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                
                # 转换为PIL进行缩放，然后转回numpy
                pil_img = Image.fromarray(img)
                resized_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                _log_debug(f"图像尺寸从 {w}x{h} 缩放到 {new_w}x{new_h}")
                return np.array(resized_img)
        except Exception:
            pass
        
        return img

    def _enhance_image_for_tesseract(self, img: Image.Image) -> Image.Image:
        """
        为Tesseract优化图像质量
        
        Args:
            img: PIL图像
            
        Returns:
            增强后的PIL图像
        """
        try:
            # 转换为灰度图
            if img.mode != 'L':
                img = img.convert('L')
            
            # 增加对比度
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.5)
            
            # 锐化
            from PIL import ImageFilter
            img = img.filter(ImageFilter.SHARPEN)
            
            # 如果图像太小，放大一些（Tesseract对小字识别较差）
            w, h = img.size
            if min(w, h) < 100:
                scale = 200 / min(w, h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                _log_debug(f"为Tesseract放大小图像: {w}x{h} -> {new_w}x{new_h}")
            
            return img
        except Exception as e:
            _log_error(f"图像增强失败: {e}")
            return img

    def run(self, img: Any) -> OCRResult:
        """
        运行OCR识别
        
        Args:
            img: 输入图像，可以是文件路径、PIL图像或numpy数组
            
        Returns:
            OCR识别结果
        """
        # 1) PaddleOCR路径 - 只支持str路径和numpy数组
        if self._paddle is not None:
            try:
                import numpy as np  # type: ignore
                _log_debug("尝试使用PaddleOCR识别")
                # 预处理图像尺寸
                processed_img = self._resize_image_if_needed(img, max_side=4000)
                
                # 准备PaddleOCR的输入：str或numpy数组
                paddle_input = None
                if isinstance(processed_img, str):
                    paddle_input = processed_img
                else:
                    # 转换为numpy数组
                    if isinstance(processed_img, np.ndarray):
                        paddle_input = processed_img
                    else:
                        pil = self._to_pil(processed_img)
                        paddle_input = np.array(pil)
                
                # 尝试新的predict API
                try:
                    results = self._paddle.predict(paddle_input)
                    if not results or len(results) == 0:
                        _log_warning("PaddleOCR predict返回空结果")
                        return OCRResult(words=[])
                    
                    result_data = results[0]
                    if not isinstance(result_data, dict):
                        _log_warning("PaddleOCR predict返回格式异常")
                        return OCRResult(words=[])
                    
                    texts = result_data.get("rec_texts", [])
                    scores = result_data.get("rec_scores", [])
                    bboxes = result_data.get("det_polygons", [])
                    
                    words: List[OCRWord] = []
                    for i, (text, score) in enumerate(zip(texts, scores)):
                        if i < len(bboxes):
                            box = bboxes[i]
                            x1 = int(min(p[0] for p in box))
                            y1 = int(min(p[1] for p in box))
                            x2 = int(max(p[0] for p in box))
                            y2 = int(max(p[1] for p in box))
                        else:
                            x1, y1, x2, y2 = 0, 0, 100, 20
                        words.append(OCRWord(text=text, bbox=(x1, y1, x2, y2), conf=float(score)))
                    return OCRResult(words=words)
                except (AttributeError, KeyError):
                    # 回退到旧的ocr API
                    res = self._paddle.ocr(paddle_input, cls=True)
                    words: List[OCRWord] = []
                    if res and res[0]:
                        for line in res[0]:
                            box = line[0]
                            x1 = int(min(p[0] for p in box))
                            y1 = int(min(p[1] for p in box))
                            x2 = int(max(p[0] for p in box))
                            y2 = int(max(p[1] for p in box))
                            text = line[1][0]
                            conf = float(line[1][1]) if line[1][1] is not None else 0.0
                            words.append(OCRWord(text=text, bbox=(x1, y1, x2, y2), conf=conf))
                    return OCRResult(words=words)
            except Exception as e:
                _log_error(f"PaddleOCR识别失败: {e}")
                _log_warning("该类型图片被处理后，不支持PaddleOCR")
                pass
        
        # 2) Tesseract路径 - 需要PIL图像
        if _has_tesseract and pytesseract is not None:
            try:
                _log_debug(f"尝试使用Tesseract识别，语言设置: {self.lang}")
                # 预处理图像尺寸
                processed_img = self._resize_image_if_needed(img, max_side=4000)
                pil = self._to_pil(processed_img)
                
                # 增强图像质量以提高识别率
                enhanced_img = self._enhance_image_for_tesseract(pil)
                
                # 使用image_to_data获取详细信息
                data = pytesseract.image_to_data(enhanced_img, lang=self.lang, output_type=pytesseract.Output.DICT)
                words: List[OCRWord] = []
                
                if not data or not data.get("text"):
                    _log_warning("Tesseract未识别到任何文字")
                    return OCRResult(words=[])
                
                n = len(data.get("text", []))
                recognized_count = 0
                for i in range(n):
                    txt = (data["text"][i] or "").strip()
                    if not txt:
                        continue
                    
                    try:
                        conf = float(data.get("conf", [0])[i])
                    except Exception:
                        conf = 0.0
                    
                    # 过滤置信度过低的结果
                    if conf < 30:  # Tesseract置信度阈值
                        continue
                    
                    try:
                        x = int(data.get("left", [0])[i])
                        y = int(data.get("top", [0])[i])
                        w = int(data.get("width", [0])[i])
                        h = int(data.get("height", [0])[i])
                    except Exception:
                        x, y, w, h = 0, 0, 100, 20
                    
                    words.append(OCRWord(text=txt, bbox=(x, y, x + w, y + h), conf=conf))
                    recognized_count += 1
                
                _log_debug(f"Tesseract识别成功，识别到 {recognized_count} 个文字片段")
                return OCRResult(words=words)
            except Exception as e:
                _log_error(f"Tesseract识别失败: {e}")
                pass
        
        # 3) 回退空结果
        return OCRResult(words=[])
