"""
图标检测器模块
使用OpenCV模板匹配实现图标检测，支持多尺度匹配和相似度阈值
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Union
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class IconDetector:
    """
    基于OpenCV模板匹配的图标检测器
    支持多尺度匹配和可配置的相似度阈值
    """
    
    def __init__(self, 
                 default_threshold: float = 0.8,
                 scale_range: Tuple[float, float] = (0.5, 2.0),
                 scale_step: float = 0.1,
                 method: int = cv2.TM_CCOEFF_NORMED):
        """
        初始化图标检测器
        
        Args:
            default_threshold: 默认相似度阈值
            scale_range: 缩放范围 (min_scale, max_scale)
            scale_step: 缩放步长
            method: OpenCV模板匹配方法
        """
        self.default_threshold = default_threshold
        self.scale_range = scale_range
        self.scale_step = scale_step
        self.method = method
        self._icon_cache = {}  # 缓存加载的图标模板
        
    def load_icon_template(self, icon_path: str) -> Optional[np.ndarray]:
        """
        加载图标模板
        
        Args:
            icon_path: 图标文件路径
            
        Returns:
            图标模板的numpy数组，加载失败返回None
        """
        if icon_path in self._icon_cache:
            return self._icon_cache[icon_path]
            
        if not os.path.exists(icon_path):
            logger.warning(f"图标文件不存在: {icon_path}")
            return None
            
        try:
            # 读取图标并转为灰度图
            template = cv2.imread(icon_path, cv2.IMREAD_GRAYSCALE)
            if template is None:
                logger.warning(f"无法读取图标文件: {icon_path}")
                return None
                
            # 缓存模板
            self._icon_cache[icon_path] = template
            logger.debug(f"成功加载图标模板: {icon_path}, 尺寸: {template.shape}")
            return template
            
        except Exception as e:
            logger.error(f"加载图标模板失败 {icon_path}: {e}")
            return None
    
    def match_template_multiscale(self, 
                                  image: np.ndarray, 
                                  template: np.ndarray,
                                  threshold: float) -> List[Dict]:
        """
        多尺度模板匹配
        
        Args:
            image: 目标图像（灰度图）
            template: 模板图像（灰度图）
            threshold: 相似度阈值
            
        Returns:
            匹配结果列表，每个结果包含位置、尺度、相似度等信息
        """
        matches = []
        h, w = template.shape
        
        # 生成缩放比例列表
        scales = np.arange(self.scale_range[0], self.scale_range[1] + self.scale_step, self.scale_step)
        
        for scale in scales:
            # 缩放模板
            scaled_w = int(w * scale)
            scaled_h = int(h * scale)
            
            # 如果缩放后的模板大于图像，跳过
            if scaled_w > image.shape[1] or scaled_h > image.shape[0]:
                continue
                
            scaled_template = cv2.resize(template, (scaled_w, scaled_h))
            
            # 模板匹配
            result = cv2.matchTemplate(image, scaled_template, self.method)
            
            # 查找满足阈值的匹配点
            locations = np.where(result >= threshold)
            
            for pt in zip(*locations[::-1]):  # 转换为(x, y)格式
                similarity = result[pt[1], pt[0]]
                matches.append({
                    'position': pt,
                    'scale': scale,
                    'similarity': float(similarity),
                    'bbox': (pt[0], pt[1], scaled_w, scaled_h),  # (x, y, w, h)
                    'template_size': (w, h)
                })
        
        # 按相似度排序
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        return matches
    
    def non_maximum_suppression(self, matches: List[Dict], overlap_threshold: float = 0.3) -> List[Dict]:
        """
        非极大值抑制，去除重叠的检测框
        
        Args:
            matches: 匹配结果列表
            overlap_threshold: 重叠阈值
            
        Returns:
            去重后的匹配结果
        """
        if not matches:
            return []
            
        # 按相似度排序
        matches = sorted(matches, key=lambda x: x['similarity'], reverse=True)
        
        selected = []
        
        for current in matches:
            x1, y1, w1, h1 = current['bbox']
            
            # 检查与已选择的框是否重叠
            is_overlap = False
            for selected_match in selected:
                x2, y2, w2, h2 = selected_match['bbox']
                
                # 计算重叠区域
                overlap_x1 = max(x1, x2)
                overlap_y1 = max(y1, y2)
                overlap_x2 = min(x1 + w1, x2 + w2)
                overlap_y2 = min(y1 + h1, y2 + h2)
                
                if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                    overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                    area1 = w1 * h1
                    area2 = w2 * h2
                    union_area = area1 + area2 - overlap_area
                    
                    iou = overlap_area / union_area if union_area > 0 else 0
                    
                    if iou > overlap_threshold:
                        is_overlap = True
                        break
            
            if not is_overlap:
                selected.append(current)
        
        return selected
    
    def detect_icon(self, 
                    image: Union[np.ndarray, str], 
                    icon_path: str,
                    threshold: Optional[float] = None,
                    apply_nms: bool = True) -> List[Dict]:
        """
        在图像中检测指定图标
        
        Args:
            image: 目标图像（numpy数组或文件路径）
            icon_path: 图标模板路径
            threshold: 相似度阈值，None时使用默认值
            apply_nms: 是否应用非极大值抑制
            
        Returns:
            检测结果列表
        """
        # 处理输入图像
        if isinstance(image, str):
            if not os.path.exists(image):
                logger.error(f"图像文件不存在: {image}")
                return []
            target_image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            if target_image is None:
                logger.error(f"无法读取图像文件: {image}")
                return []
        else:
            target_image = image
            if len(target_image.shape) == 3:
                target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
        
        # 加载图标模板
        template = self.load_icon_template(icon_path)
        if template is None:
            return []
        
        # 使用指定阈值或默认阈值
        use_threshold = threshold if threshold is not None else self.default_threshold
        
        # 执行多尺度匹配
        matches = self.match_template_multiscale(target_image, template, use_threshold)
        
        # 应用非极大值抑制
        if apply_nms and matches:
            matches = self.non_maximum_suppression(matches)
        
        logger.debug(f"图标检测完成，找到 {len(matches)} 个匹配")
        return matches
    
    def detect_icons_batch(self,
                          image: Union[np.ndarray, str],
                          icon_paths: List[str],
                          threshold: Optional[float] = None) -> Dict[str, List[Dict]]:
        """
        批量检测多个图标
        
        Args:
            image: 目标图像
            icon_paths: 图标路径列表
            threshold: 相似度阈值
            
        Returns:
            每个图标的检测结果字典
        """
        results = {}
        
        for icon_path in icon_paths:
            icon_name = os.path.basename(icon_path)
            results[icon_name] = self.detect_icon(image, icon_path, threshold)
        
        return results
    
    def clear_cache(self):
        """清空图标模板缓存"""
        self._icon_cache.clear()
        logger.debug("图标模板缓存已清空")


class IconPathResolver:
    """图标路径解析器，负责根据配置查找图标文件"""
    
    def __init__(self, base_paths: List[str]):
        """
        初始化路径解析器
        
        Args:
            base_paths: 图标搜索基础路径列表
        """
        self.base_paths = [Path(p) for p in base_paths]
    
    def resolve_icon_path(self, icon_name: str, app_id: Optional[str] = None) -> Optional[str]:
        """
        解析图标路径
        
        Args:
            icon_name: 图标名称
            app_id: 应用ID，用于确定子目录
            
        Returns:
            图标文件的完整路径，找不到返回None
        """
        # 常见的图标文件扩展名
        extensions = ['.png', '.jpg', '.jpeg', '.bmp']
        
        # 搜索路径优先级
        search_paths = []
        
        # 1. 如果有app_id，优先在对应目录下搜索
        if app_id:
            app_name = self._extract_app_name(app_id)
            for base_path in self.base_paths:
                search_paths.append(base_path / app_name)
        
        # 2. 在所有基础路径下搜索
        search_paths.extend(self.base_paths)
        
        # 在每个搜索路径下查找图标文件
        for search_path in search_paths:
            if not search_path.exists():
                continue
                
            for ext in extensions:
                # 尝试直接匹配
                icon_path = search_path / f"{icon_name}{ext}"
                if icon_path.exists():
                    return str(icon_path)
                
                # 尝试递归搜索
                for icon_file in search_path.rglob(f"{icon_name}{ext}"):
                    return str(icon_file)
        
        logger.warning(f"未找到图标文件: {icon_name} (app_id: {app_id})")
        return None
    
    def _extract_app_name(self, app_id: str) -> str:
        """从app_id提取应用名称"""
        # 例如: com.tencent.mm -> weixin 或 mm
        if 'tencent.mm' in app_id:
            return 'weixin'
        elif 'bilibili' in app_id:
            return 'bilibili'
        elif 'xiecheng' in app_id or 'ctrip' in app_id:
            return 'xiecheng'
        else:
            return app_id
            # 默认取最后一段
            # return app_id.split('.')[-1]
    
    def list_available_icons(self, app_id: Optional[str] = None) -> List[str]:
        """
        列出可用的图标
        
        Args:
            app_id: 应用ID，用于筛选特定应用的图标
            
        Returns:
            可用图标名称列表
        """
        icons = set()
        
        search_paths = []
        if app_id:
            app_name = self._extract_app_name(app_id)
            for base_path in self.base_paths:
                app_path = base_path / app_name
                if app_path.exists():
                    search_paths.append(app_path)
        
        if not search_paths:
            search_paths = [p for p in self.base_paths if p.exists()]
        
        for search_path in search_paths:
            for icon_file in search_path.rglob('*'):
                if icon_file.is_file() and icon_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']:
                    # 移除扩展名作为图标名称
                    icon_name = icon_file.stem
                    icons.add(icon_name)
        
        return sorted(list(icons))
