from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Iterable, Set

# 尝试导入日志系统
try:
    import sys
    from pathlib import Path
    current_dir = Path(__file__).resolve()
    avdag_path = current_dir.parent.parent.parent.parent / "avdag"
    if avdag_path.exists():
        sys.path.insert(0, str(avdag_path.parent))
        from avdag.logger import get_logger
        _logger = get_logger("vision")
        _has_logger = True
    else:
        _has_logger = False
except ImportError:
    _has_logger = False

def _log_debug(msg: str):
    if _has_logger:
        _logger.debug(msg)
    else:
        print(f"[Vision] {msg}")

try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore
    _HAS_CV2 = False
try:
    import numpy as np  # type: ignore
    _HAS_NP = True
except Exception:  # pragma: no cover
    np = None  # type: ignore
    _HAS_NP = False
from PIL import Image, ImageChops, ImageStat

from .ocr_engine import OCREngine, OCRResult


@dataclass
class Detection:
    """检测结果数据类"""
    name: str  # 检测目标名称
    bbox: Tuple[int, int, int, int]  # 边界框坐标 (x1, y1, x2, y2)
    score: float  # 检测置信度分数


def imread(path: str):
    """
    读取图像文件
    
    Args:
        path: 图像文件路径
        
    Returns:
        图像对象（OpenCV或PIL格式）
        
    Raises:
        FileNotFoundError: 文件不存在
    """
    if _HAS_CV2:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        return img
    # 回退到PIL
    pil = Image.open(path).convert("RGB")
    return pil


def to_pil(img: Any) -> Image.Image:
    """
    将图像转换为PIL格式
    
    Args:
        img: 输入图像（OpenCV、PIL或numpy数组）
        
    Returns:
        PIL图像对象
        
    Raises:
        TypeError: 不支持的图像类型
    """
    if isinstance(img, Image.Image):
        return img
    if _HAS_CV2:
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 如果已经是numpy数组（RGB），尝试直接转换
    try:
        import numpy as _np  # type: ignore
        if isinstance(img, _np.ndarray):
            return Image.fromarray(img)
    except Exception:
        pass
    raise TypeError("to_pil不支持的图像类型")


def ocr_on_image(img_or_path: Any, engine: OCREngine) -> OCRResult:
    """
    对图像执行OCR识别
    
    Args:
        img_or_path: 图像或图像路径
        engine: OCR引擎实例
        
    Returns:
        OCR识别结果
    """
    # 直接传递给OCR引擎，让引擎内部处理格式转换
    return engine.run(img_or_path)


def find_keywords(ocr: OCRResult, keywords: Iterable[str], fuzzy: bool = True) -> Set[str]:
    """
    在OCR结果中查找关键词
    
    Args:
        ocr: OCR识别结果
        keywords: 要查找的关键词列表
        fuzzy: 是否使用模糊匹配
        
    Returns:
        找到的关键词集合
    """
    found: Set[str] = set()
    text = ocr.get_text()
    for k in keywords:
        if not k:
            continue
        if ocr.find(k, fuzzy=fuzzy):
            found.add(k)
    return found


def template_match(img: Any, template: Any, threshold: float = 0.85) -> List[Detection]:
    """
    模板匹配检测
    
    Args:
        img: 目标图像
        template: 模板图像
        threshold: 匹配阈值
        
    Returns:
        检测结果列表
    """
    if not _HAS_CV2 or not _HAS_NP:
        return []
    ih, iw = img.shape[:2]
    th, tw = template.shape[:2]
    if ih < th or iw < tw:
        return []
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    ys, xs = np.where(res >= threshold)
    dets: List[Detection] = []
    for (x, y) in zip(xs, ys):
        dets.append(Detection(name="template", bbox=(int(x), int(y), int(x+tw), int(y+th)), score=float(res[y, x])))
    
    # 非极大值抑制（简单版本）
    dets_sorted = sorted(dets, key=lambda d: d.score, reverse=True)
    kept: List[Detection] = []
    
    def iou(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
        """计算两个边界框的交并比"""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
        inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / max(area_a + area_b - inter, 1e-6)
    
    for d in dets_sorted:
        if all(iou(d.bbox, k.bbox) < 0.5 for k in kept):
            kept.append(d)
    return kept


def _load_icon_template(path: str):
    """加载图标模板文件"""
    if _HAS_CV2:
        t = cv2.imread(path, cv2.IMREAD_COLOR)
        return t
    # 回退：作为PIL加载，在匹配时禁用模板匹配
    return Image.open(path).convert('RGB')


def build_icon_bank(icons_dir: str) -> Dict[str, Any]:
    """
    从目录中加载图标模板
    
    Args:
        icons_dir: 图标模板目录路径
        
    Returns:
        图标模板字典，键为文件名（不含扩展名），值为模板图像
        
    Note:
        接受 搜索.png, 购物车.jpg 等文件，键为文件名（不含扩展名）
    """
    if not icons_dir or not os.path.isdir(icons_dir):
        return {}
    bank: Dict[str, Any] = {}
    for fn in os.listdir(icons_dir):
        name, ext = os.path.splitext(fn)
        if ext.lower() not in {'.png', '.jpg', '.jpeg'}:
            continue
        try:
            bank[name] = _load_icon_template(os.path.join(icons_dir, fn))
        except Exception:
            continue
    return bank


def detect_icons_in_image(img: Any, icon_bank: Dict[str, Any], threshold: float = 0.85) -> Set[str]:
    """
    在图像中检测图标
    
    Args:
        img: 目标图像
        icon_bank: 图标模板字典
        threshold: 检测阈值
        
    Returns:
        检测到的图标名称集合
    """
    if not icon_bank:
        return set()
    present: Set[str] = set()
    if not _HAS_CV2 or not _HAS_NP:
        return present
    base = img if not isinstance(img, Image.Image) else cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    for name, tmpl in icon_bank.items():
        _log_debug(f"检测图标: {name}")
        if tmpl is None:
            continue
        # 检查PIL图像的有效性
        if isinstance(tmpl, Image.Image) and (tmpl.size[0] < 10 or tmpl.size[1] < 10):
            continue  # 跳过无效模板
        # 检查numpy数组的有效性
        if _HAS_NP and hasattr(tmpl, 'shape') and (tmpl.shape[0] < 10 or tmpl.shape[1] < 10):
            continue  # 跳过无效模板
        dets = template_match(base, tmpl, threshold=threshold)
        if dets:
            present.add(name)
    return present


def red_badge_score(img: Any) -> float:
    """
    计算红色徽章分数的启发式算法（如购物车徽章）
    
    专注于右上角区域检测红色/橙色圆点。
    仅使用PIL操作，无需cv2/numpy。
    
    Args:
        img: 输入图像
        
    Returns:
        归一化分数 [0,1]
    """
    pil = to_pil(img)
    w, h = pil.size
    # 专注于右上角区域（状态栏+头部）
    box = (int(w*0.7), int(0), w, int(h*0.2))
    region = pil.crop(box)
    region = region.resize((200, 120))
    px = region.load()
    red_count = 0
    total = region.size[0] * region.size[1]
    for y in range(region.size[1]):
        for x in range(region.size[0]):
            r, g, b = px[x, y]
            if r >= 180 and g <= 100 and b <= 100 and (r - max(g, b)) >= 50:
                red_count += 1
    return red_count / max(total, 1)


def search_box_score(img: Any) -> float:
    """
    计算搜索框分数的启发式算法
    
    检测顶部区域的大型亮色水平条带（典型的搜索/文本输入框）。
    仅使用PIL操作。
    
    Args:
        img: 输入图像
        
    Returns:
        分数 [0,1]
    """
    pil = to_pil(img).convert('L')
    w, h = pil.size
    # 专注于顶部中心区域，避免极端边缘
    left = int(w * 0.05)
    right = int(w * 0.95)
    top = int(h * 0.02)
    bottom = int(h * 0.25)
    region = pil.crop((left, top, right, bottom))
    region = region.resize((200, 120))
    px = region.load()
    W, H = region.size
    
    # 预计算每行的亮度掩码
    bright_thresh = 220
    row_bright_frac = []
    for y in range(H):
        bright = 0
        for x in range(W):
            if px[x, y] >= bright_thresh:
                bright += 1
        row_bright_frac.append(bright / max(W, 1))
    
    # 找到高亮度覆盖率的最佳连续条带
    best = 0.0
    y = 0
    while y < H:
        if row_bright_frac[y] < 0.55:  # 当行相当亮时开始
            y += 1
            continue
        y0 = y
        while y < H and row_bright_frac[y] >= 0.5:
            y += 1
        y1 = y  # 排他性
        height = y1 - y0
        if 6 <= height <= 40:  # 调整大小后的合理框高度
            avg_cov = sum(row_bright_frac[y0:y1]) / max(height, 1)
            score = avg_cov * min(1.0, height / 25.0)
            if score > best:
                best = score
    return float(best)


def ssim(a: Any, b: Any) -> float:
    """
    计算两幅图像的结构相似性指数
    
    Args:
        a, b: 两幅待比较的图像
        
    Returns:
        SSIM值 [0,1]，值越高表示越相似
    """
    # 优先使用skimage+cv2，回退到基于PIL的近似
    if _HAS_CV2 and _HAS_NP:
        try:
            from skimage.metrics import structural_similarity as ssim_fn  # type: ignore
            a_arr = a if isinstance(a, type(getattr(a, 'shape', None))) else a
            b_arr = b if isinstance(b, type(getattr(b, 'shape', None))) else b
            if isinstance(a, Image.Image):
                a_arr = cv2.cvtColor(np.array(a), cv2.COLOR_RGB2GRAY)
            else:
                a_arr = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
            if isinstance(b, Image.Image):
                b_arr = cv2.cvtColor(np.array(b), cv2.COLOR_RGB2GRAY)
            else:
                b_arr = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
            score, _ = ssim_fn(a_arr, b_arr, full=True)
            return float(score)
        except Exception:
            pass
    
    # PIL回退：归一化平均绝对差异相似性
    a_pil = to_pil(a)
    b_pil = to_pil(b)
    a_pil = a_pil.convert('L').resize((300, 600))
    b_pil = b_pil.convert('L').resize((300, 600))
    diff = ImageChops.difference(a_pil, b_pil)
    stat = ImageStat.Stat(diff)
    mad = stat.mean[0] / 255.0
    return 1.0 - mad  # 粗略近似


def estimate_scroll_direction(prev: Any, curr: Any) -> Optional[str]:
    """
    估计滚动方向
    
    Args:
        prev: 前一帧图像
        curr: 当前帧图像
        
    Returns:
        滚动方向："UP"、"DOWN" 或 None（无明显滚动）
    """
    if _HAS_CV2 and _HAS_NP:
        try:
            # 启发式：计算密集ORB匹配和中位数dy
            orb = cv2.ORB_create(nfeatures=1000)
            prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY) if not isinstance(prev, Image.Image) else cv2.cvtColor(np.array(prev), cv2.COLOR_RGB2GRAY)
            curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY) if not isinstance(curr, Image.Image) else cv2.cvtColor(np.array(curr), cv2.COLOR_RGB2GRAY)
            kp1, des1 = orb.detectAndCompute(prev_gray, None)
            kp2, des2 = orb.detectAndCompute(curr_gray, None)
            if des1 is None or des2 is None:
                return None
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            if not matches:
                return None
            dys = []
            for m in matches:
                y1 = kp1[m.queryIdx].pt[1]
                y2 = kp2[m.trainIdx].pt[1]
                dys.append(y2 - y1)
            if not dys:
                return None
            from statistics import median
            med_dy = float(median(dys))
            if abs(med_dy) < 2.0:
                return None
            return "UP" if med_dy < 0 else "DOWN"
        except Exception:
            return None
    
    # 回退：比较上下三分之一区域的平均强度差异
    try:
        a = to_pil(prev).convert('L').resize((300, 600))
        b = to_pil(curr).convert('L').resize((300, 600))
        top_diff = ImageChops.difference(a.crop((0,0,300,200)), b.crop((0,0,300,200)))
        bot_diff = ImageChops.difference(a.crop((0,400,300,600)), b.crop((0,400,300,600)))
        top_mean = ImageStat.Stat(top_diff).mean[0]
        bot_mean = ImageStat.Stat(bot_diff).mean[0]
        if abs(top_mean - bot_mean) < 1.0:
            return None
        return "UP" if top_mean > bot_mean else "DOWN"
    except Exception:
        return None


# 键盘识别的常见标记
KEYBOARD_TOKENS = {"空格", "拼音", "英文", "123", "ABC", "符"}

def has_keyboard(ocr: OCRResult) -> bool:
    """
    检测OCR结果中是否包含键盘相关文字
    
    Args:
        ocr: OCR识别结果
        
    Returns:
        是否检测到键盘
    """
    text = ocr.get_text()
    return any(tok in text for tok in KEYBOARD_TOKENS)
