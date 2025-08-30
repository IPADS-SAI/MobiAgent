"""
图标检测配置管理器
"""

import os
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class IconDetectionConfig:
    """图标检测配置类"""
    
    def __init__(self, 
                 icon_base_paths: Optional[List[str]] = None,
                 default_threshold: float = 0.8,
                 scale_range: tuple = (0.5, 2.0),
                 scale_step: float = 0.1,
                 nms_threshold: float = 0.3):
        """
        初始化图标检测配置
        
        Args:
            icon_base_paths: 图标基础搜索路径列表
            default_threshold: 默认相似度阈值
            scale_range: 缩放范围
            scale_step: 缩放步长
            nms_threshold: 非极大值抑制阈值
        """
        # 设置默认图标路径
        if icon_base_paths is None:
            # 尝试从环境变量或项目结构推断
            project_root = self._find_project_root()
            icon_base_paths = [
                os.path.join(project_root, 'task_configs', 'icons'),
            ]
        
        self.icon_base_paths = [Path(p) for p in icon_base_paths if os.path.exists(p)]
        self.default_threshold = default_threshold
        self.scale_range = scale_range
        self.scale_step = scale_step
        self.nms_threshold = nms_threshold
        
        # 验证配置
        self._validate_config()
    
    def _find_project_root(self) -> str:
        """查找项目根目录"""
        current_dir = Path(__file__).parent
        
        # 向上查找，直到找到包含特定标识文件的目录
        markers = ['pyproject.toml', 'requirements.txt', '.git']
        
        for _ in range(10):  # 最多向上查找10级
            for marker in markers:
                if (current_dir / marker).exists():
                    return str(current_dir)
            current_dir = current_dir.parent
            
        # 如果找不到，返回当前目录的上两级（假设在tools/Icon_detection中）
        return str(Path(__file__).parent.parent.parent)
    
    def _validate_config(self):
        """验证配置有效性"""
        if not self.icon_base_paths:
            logger.warning("未找到有效的图标路径")
        
        if not (0.0 <= self.default_threshold <= 1.0):
            raise ValueError(f"默认阈值必须在0-1之间，当前值: {self.default_threshold}")
        
        if self.scale_range[0] <= 0 or self.scale_range[1] <= self.scale_range[0]:
            raise ValueError(f"无效的缩放范围: {self.scale_range}")
        
        if self.scale_step <= 0:
            raise ValueError(f"缩放步长必须大于0，当前值: {self.scale_step}")
    
    def get_icon_paths(self) -> List[str]:
        """获取所有图标搜索路径"""
        return [str(p) for p in self.icon_base_paths]
    
    def add_icon_path(self, path: str):
        """添加新的图标搜索路径"""
        path_obj = Path(path)
        if path_obj.exists() and path_obj not in self.icon_base_paths:
            self.icon_base_paths.append(path_obj)
            logger.info(f"添加图标搜索路径: {path}")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'icon_base_paths': [str(p) for p in self.icon_base_paths],
            'default_threshold': self.default_threshold,
            'scale_range': self.scale_range,
            'scale_step': self.scale_step,
            'nms_threshold': self.nms_threshold
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'IconDetectionConfig':
        """从字典创建配置实例"""
        return cls(
            icon_base_paths=config_dict.get('icon_base_paths'),
            default_threshold=config_dict.get('default_threshold', 0.8),
            scale_range=tuple(config_dict.get('scale_range', (0.5, 2.0))),
            scale_step=config_dict.get('scale_step', 0.1),
            nms_threshold=config_dict.get('nms_threshold', 0.3)
        )


# 全局默认配置实例
_default_config = None

def get_default_config() -> IconDetectionConfig:
    """获取默认配置实例"""
    global _default_config
    if _default_config is None:
        _default_config = IconDetectionConfig()
    return _default_config

def set_default_config(config: IconDetectionConfig):
    """设置默认配置"""
    global _default_config
    _default_config = config
