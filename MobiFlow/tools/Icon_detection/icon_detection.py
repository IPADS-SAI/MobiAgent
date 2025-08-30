"""
图标检测主接口模块
提供简单易用的图标检测接口
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Optional, Union, Any
import os

from .icon_detector import IconDetector, IconPathResolver
from .config import IconDetectionConfig, get_default_config

logger = logging.getLogger(__name__)


class IconDetectionService:
    """图标检测服务类，提供高级接口"""
    
    def __init__(self, config: Optional[IconDetectionConfig] = None):
        """
        初始化图标检测服务
        
        Args:
            config: 图标检测配置，为None时使用默认配置
        """
        self.config = config or get_default_config()
        self.detector = IconDetector(
            default_threshold=self.config.default_threshold,
            scale_range=self.config.scale_range,
            scale_step=self.config.scale_step
        )
        self.path_resolver = IconPathResolver(self.config.get_icon_paths())
    
    def detect_icons(self, 
                    image: Union[np.ndarray, str],
                    icon_names: List[str],
                    app_id: Optional[str] = None,
                    threshold: Optional[float] = None,
                    match_mode: str = 'any') -> Dict[str, Any]:
        """
        检测图像中的图标
        
        Args:
            image: 目标图像（numpy数组或文件路径）
            icon_names: 要检测的图标名称列表
            app_id: 应用ID，用于确定图标搜索路径
            threshold: 相似度阈值
            match_mode: 匹配模式，'any'表示匹配任意一个，'all'表示必须匹配所有
            
        Returns:
            检测结果字典，包含成功状态、匹配的图标、详细结果等
        """
        logger.debug(f"开始图标检测，图标列表: {icon_names}, 匹配模式: {match_mode}")
        
        result = {
            'success': False,
            'matched_icons': [],
            'unmatched_icons': [],
            'details': {},
            'total_matches': 0,
            'match_mode': match_mode
        }
        
        # 预处理图像
        processed_image = self._preprocess_image(image)
        if processed_image is None:
            result['error'] = '无法处理输入图像'
            return result
        
        # 逐个检测图标
        for icon_name in icon_names:
            # 解析图标路径
            icon_path = self.path_resolver.resolve_icon_path(icon_name, app_id)
            if icon_path is None:
                result['unmatched_icons'].append(icon_name)
                result['details'][icon_name] = {
                    'found': False,
                    'error': '图标文件未找到',
                    'matches': []
                }
                continue
            
            # 执行检测
            matches = self.detector.detect_icon(
                processed_image, 
                icon_path, 
                threshold
            )
            
            # 记录结果
            is_found = len(matches) > 0
            result['details'][icon_name] = {
                'found': is_found,
                'icon_path': icon_path,
                'matches': matches,
                'match_count': len(matches)
            }
            
            if is_found:
                result['matched_icons'].append(icon_name)
                result['total_matches'] += len(matches)
                logger.debug(f"图标 {icon_name} 检测到 {len(matches)} 个匹配")
            else:
                result['unmatched_icons'].append(icon_name)
                logger.debug(f"图标 {icon_name} 未检测到")
        
        # 根据匹配模式判断成功状态
        if match_mode == 'any':
            result['success'] = len(result['matched_icons']) > 0
        elif match_mode == 'all':
            result['success'] = len(result['unmatched_icons']) == 0
        else:
            logger.warning(f"未知的匹配模式: {match_mode}")
            result['success'] = False
        
        logger.debug(f"图标检测完成，成功: {result['success']}, "
                    f"匹配: {len(result['matched_icons'])}, "
                    f"未匹配: {len(result['unmatched_icons'])}")
        
        return result
    
    def _preprocess_image(self, image: Union[np.ndarray, str]) -> Optional[np.ndarray]:
        """
        预处理图像
        
        Args:
            image: 输入图像
            
        Returns:
            处理后的灰度图像，失败返回None
        """
        try:
            if isinstance(image, str):
                if not os.path.exists(image):
                    logger.error(f"图像文件不存在: {image}")
                    return None
                img = cv2.imread(image)
                if img is None:
                    logger.error(f"无法读取图像文件: {image}")
                    return None
            else:
                img = image.copy()
            
            # 转换为灰度图
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            return img
            
        except Exception as e:
            logger.error(f"图像预处理失败: {e}")
            return None
    
    def get_available_icons(self, app_id: Optional[str] = None) -> List[str]:
        """
        获取可用的图标列表
        
        Args:
            app_id: 应用ID
            
        Returns:
            可用图标名称列表
        """
        return self.path_resolver.list_available_icons(app_id)
    
    def validate_icons(self, icon_names: List[str], app_id: Optional[str] = None) -> Dict[str, bool]:
        """
        验证图标是否存在
        
        Args:
            icon_names: 图标名称列表
            app_id: 应用ID
            
        Returns:
            图标名称到存在状态的映射
        """
        result = {}
        for icon_name in icon_names:
            icon_path = self.path_resolver.resolve_icon_path(icon_name, app_id)
            result[icon_name] = icon_path is not None
        return result


# 全局服务实例
_default_service = None

def get_icon_detection_service(config: Optional[IconDetectionConfig] = None) -> IconDetectionService:
    """获取图标检测服务实例"""
    global _default_service
    if _default_service is None or config is not None:
        _default_service = IconDetectionService(config)
    return _default_service


def detect_icons_simple(image: Union[np.ndarray, str],
                       icon_names: List[str],
                       app_id: Optional[str] = None,
                       threshold: Optional[float] = None,
                       match_mode: str = 'any') -> bool:
    """
    简化的图标检测接口
    
    Args:
        image: 目标图像
        icon_names: 图标名称列表
        app_id: 应用ID
        threshold: 相似度阈值
        match_mode: 匹配模式 ('any' 或 'all')
        
    Returns:
        检测是否成功
    """
    service = get_icon_detection_service()
    result = service.detect_icons(image, icon_names, app_id, threshold, match_mode)
    return result['success']


def detect_single_icon(image: Union[np.ndarray, str],
                      icon_name: str,
                      app_id: Optional[str] = None,
                      threshold: Optional[float] = None) -> bool:
    """
    检测单个图标
    
    Args:
        image: 目标图像
        icon_name: 图标名称
        app_id: 应用ID
        threshold: 相似度阈值
        
    Returns:
        是否检测到图标
    """
    return detect_icons_simple(image, [icon_name], app_id, threshold, 'any')
