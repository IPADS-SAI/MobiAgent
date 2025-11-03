"""
OCR文字识别引擎
支持PaddleOCR和Tesseract两种OCR引擎，提供统一的接口
"""

from __future__ import annotations
import os
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any, Union
from pathlib import Path

from PIL import Image

# 配置日志
logger = logging.getLogger(__name__)

# 尝试导入OCR引擎
try:
    from paddleocr import PaddleOCR
    import paddle
    _has_paddle = True
except ImportError:
    PaddleOCR = None
    paddle = None
    _has_paddle = False

try:
    import pytesseract
    _has_tesseract = True
    
    # 检测Tesseract是否正确安装
    def _check_tesseract_installation():
        try:
            version = pytesseract.get_tesseract_version()
            logger.info(f"检测到Tesseract版本: {version}")
            return True
        except Exception as e:
            logger.error(f"Tesseract未正确安装或配置: {e}")
            # 尝试自动配置Tesseract路径（Windows）
            if os.name == 'nt':
                possible_paths = [
                    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                    r"D:\Program Files\Tesseract-OCR\tesseract.exe",
                    r"D:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        pytesseract.pytesseract.tesseract_cmd = path
                        logger.info(f"设置Tesseract路径: {path}")
                        try:
                            version = pytesseract.get_tesseract_version()
                            logger.info(f"Tesseract配置成功，版本: {version}")
                            return True
                        except Exception:
                            continue
            return False
    
    _has_tesseract = _check_tesseract_installation()
    
except ImportError:
    pytesseract = None
    _has_tesseract = False

# 全局PaddleOCR实例缓存
_global_paddle_instance = None


@dataclass
class OCRWord:
    """OCR识别的单个词语结果"""
    text: str
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    conf: float


@dataclass
class OCRResult:
    """OCR识别的完整结果"""
    words: List[OCRWord]

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
            from rapidfuzz import fuzz
            return fuzz.partial_ratio(keyword, text) >= 80
        except ImportError:
            return keyword in text


class OCREngine:
    """OCR文字识别引擎，支持Tesseract和PaddleOCR"""
    
    def __init__(self, 
                 lang: str = "chi_sim+eng", 
                 use_paddle: Optional[bool] = None,
                 paddle_config: Optional[dict] = None):
        """
        初始化OCR引擎
        
        Args:
            lang: Tesseract的语言设置，默认中英文混合
            use_paddle: 是否使用PaddleOCR，None表示自动选择
            paddle_config: PaddleOCR配置参数
        """
        self.lang = lang
        if use_paddle is None:
            use_paddle = _has_paddle
        self.use_paddle = use_paddle
        self.paddle_config = paddle_config or {}
        self._paddle: Optional[Any] = None
        
        if self.use_paddle and _has_paddle:
            self._paddle = self._get_paddle_instance()

    def _get_paddle_instance(self) -> Optional[Any]:
        """获取全局PaddleOCR实例"""
        global _global_paddle_instance
        if _global_paddle_instance is None:
            try:
                # 判断是否支持GPU
                use_gpu = paddle and paddle.device.is_compiled_with_cuda()
                device = "gpu" if use_gpu else "cpu"
                if paddle:
                    paddle.set_device(device)

                logger.info(f"正在初始化PaddleOCR实例（设备: {device.upper()}）...")

                # 合并配置参数
                config = {
                    "lang": "ch",
                    "use_textline_orientation": True,
                    **self.paddle_config
                }

                _global_paddle_instance = PaddleOCR(**config)
                logger.info(f"PaddleOCR初始化成功（使用{device.upper()}）")

            except Exception as e:
                logger.error(f"PaddleOCR初始化失败: {e}")
                try:
                    logger.info("尝试使用默认参数初始化PaddleOCR...")
                    if paddle:
                        paddle.set_device("cpu")
                    _global_paddle_instance = PaddleOCR(lang="ch")
                    logger.info("PaddleOCR默认参数初始化成功")
                except Exception as e2:
                    logger.error(f"PaddleOCR默认参数初始化也失败: {e2}")
                    _global_paddle_instance = None
        return _global_paddle_instance

    def _to_pil(self, img: Any) -> Image.Image:
        """将输入图像转换为PIL图像对象"""
        if isinstance(img, str):
            return Image.open(img).convert("RGB")
        
        try:
            import numpy as np
            if isinstance(img, np.ndarray):
                return Image.fromarray(img)
        except ImportError:
            pass
            
        if isinstance(img, Image.Image):
            return img
        raise TypeError("不支持的图像类型")

    def _resize_image_if_needed(self, img: Any, max_side: int = 4000) -> Any:
        """如果图像尺寸超过最大边长限制，则缩放图像"""
        if isinstance(img, str):
            try:
                pil_img = Image.open(img).convert("RGB")
                w, h = pil_img.size
                if max(w, h) <= max_side:
                    return img
                
                # 需要缩放
                scale = max_side / max(w, h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                
                resized_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                logger.debug(f"图像尺寸从 {w}x{h} 缩放到 {new_w}x{new_h}")
                
                try:
                    import numpy as np
                    return np.array(resized_img)
                except ImportError:
                    return resized_img
            except Exception as e:
                logger.error(f"图像缩放失败: {e}")
                return img
        
        if isinstance(img, Image.Image):
            w, h = img.size
            if max(w, h) <= max_side:
                return img
            
            scale = max_side / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            logger.debug(f"图像尺寸从 {w}x{h} 缩放到 {new_w}x{new_h}")
            return resized_img
        
        try:
            import numpy as np
            if isinstance(img, np.ndarray):
                h, w = img.shape[:2]
                if max(w, h) <= max_side:
                    return img
                
                scale = max_side / max(w, h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                
                pil_img = Image.fromarray(img)
                resized_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                logger.debug(f"图像尺寸从 {w}x{h} 缩放到 {new_w}x{new_h}")
                return np.array(resized_img)
        except ImportError:
            pass
        
        return img

    def _enhance_image_for_tesseract(self, img: Image.Image) -> Image.Image:
        """为Tesseract优化图像质量"""
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
            
            # 如果图像太小，放大
            w, h = img.size
            if min(w, h) < 100:
                scale = 200 / min(w, h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                logger.debug(f"为Tesseract放大小图像: {w}x{h} -> {new_w}x{new_h}")
            
            return img
        except Exception as e:
            logger.error(f"图像增强失败: {e}")
            return img

    def run(self, img: Any) -> OCRResult:
        """
        运行OCR识别
        
        Args:
            img: 输入图像，可以是文件路径、PIL图像或numpy数组
            
        Returns:
            OCR识别结果
        """
        # 1) PaddleOCR路径
        if self._paddle is not None:
            try:
                logger.debug("尝试使用PaddleOCR识别")
                processed_img = self._resize_image_if_needed(img, max_side=4000)
                
                # 准备PaddleOCR的输入
                paddle_input = None
                if isinstance(processed_img, str):
                    paddle_input = processed_img
                else:
                    try:
                        import numpy as np
                        if isinstance(processed_img, np.ndarray):
                            paddle_input = processed_img
                        else:
                            pil = self._to_pil(processed_img)
                            paddle_input = np.array(pil)
                    except ImportError:
                        pil = self._to_pil(processed_img)
                        # 如果无法转换为numpy，保存临时文件
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                            pil.save(tmp.name)
                            paddle_input = tmp.name
                
                # 尝试新的predict API
                try:
                    results = self._paddle.predict(paddle_input)
                    if results and len(results) > 0:
                        result_data = results[0]
                        if isinstance(result_data, dict):
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
                logger.error(f"PaddleOCR识别失败: {e}")
        
        # 2) Tesseract路径
        if _has_tesseract and pytesseract is not None:
            try:
                logger.debug(f"尝试使用Tesseract识别，语言设置: {self.lang}")
                processed_img = self._resize_image_if_needed(img, max_side=4000)
                pil = self._to_pil(processed_img)
                
                # 增强图像质量
                enhanced_img = self._enhance_image_for_tesseract(pil)
                
                # 使用image_to_data获取详细信息
                data = pytesseract.image_to_data(enhanced_img, lang=self.lang, output_type=pytesseract.Output.DICT)
                words: List[OCRWord] = []
                
                if data and data.get("text"):
                    n = len(data.get("text", []))
                    recognized_count = 0
                    for i in range(n):
                        txt = (data["text"][i] or "").strip()
                        if not txt:
                            continue
                        
                        try:
                            conf = float(data.get("conf", [0])[i])
                        except (IndexError, ValueError):
                            conf = 0.0
                        
                        # 过滤置信度过低的结果
                        if conf < 30:
                            continue
                        
                        try:
                            x = int(data.get("left", [0])[i])
                            y = int(data.get("top", [0])[i])
                            w = int(data.get("width", [0])[i])
                            h = int(data.get("height", [0])[i])
                        except (IndexError, ValueError):
                            x, y, w, h = 0, 0, 100, 20
                        
                        words.append(OCRWord(text=txt, bbox=(x, y, x + w, y + h), conf=conf))
                        recognized_count += 1
                    
                    logger.debug(f"Tesseract识别成功，识别到 {recognized_count} 个文字片段")
                    return OCRResult(words=words)
            except Exception as e:
                logger.error(f"Tesseract识别失败: {e}")
        
        # 3) 无可用引擎时返回空结果
        available_engines = []
        if self._paddle is not None:
            available_engines.append("PaddleOCR")
        if _has_tesseract and pytesseract is not None:
            available_engines.append("Tesseract")
            
        if available_engines:
            logger.warning(f"可用引擎 {available_engines} 识别失败，返回空结果")
        else:
            logger.warning("无可用OCR引擎，返回空结果")
        return OCRResult(words=[])

    @staticmethod
    def get_available_engines() -> List[str]:
        """获取可用的OCR引擎列表"""
        engines = []
        if _has_paddle:
            engines.append("PaddleOCR")
        if _has_tesseract:
            engines.append("Tesseract")
        return engines

    @property
    def current_engine(self) -> str:
        """获取当前使用的OCR引擎"""
        if self.use_paddle and _has_paddle:
            return "PaddleOCR"
        elif _has_tesseract:
            return "Tesseract"
        else:
            return "None"


# 便捷函数
def ocr_image(img: Union[str, Image.Image], 
              engine: str = "auto",
              lang: str = "chi_sim+eng") -> OCRResult:
    """
    便捷的OCR函数
    
    Args:
        img: 输入图像
        engine: OCR引擎 ("auto", "paddle", "tesseract")
        lang: 语言设置
        
    Returns:
        OCR识别结果
    """
    if engine == "auto":
        use_paddle = None
    elif engine == "paddle":
        use_paddle = True
    elif engine == "tesseract":
        use_paddle = False
    else:
        raise ValueError(f"不支持的OCR引擎: {engine}")
    
    ocr_engine = OCREngine(lang=lang, use_paddle=use_paddle)
    return ocr_engine.run(img)