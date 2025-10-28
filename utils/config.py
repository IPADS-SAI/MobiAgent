"""
统一配置管理
管理OCR引擎和图标检测的配置参数
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class OCRConfig:
    """OCR引擎配置"""
    lang: str = "chi_sim+eng"  # Tesseract语言设置
    use_paddle: Optional[bool] = None  # 是否使用PaddleOCR，None为自动选择
    paddle_config: Optional[Dict[str, Any]] = None  # PaddleOCR配置参数
    tesseract_config: Optional[Dict[str, Any]] = None  # Tesseract配置参数
    
    def __post_init__(self):
        if self.paddle_config is None:
            self.paddle_config = {}
        if self.tesseract_config is None:
            self.tesseract_config = {}


@dataclass 
class IconDetectionConfig:
    """图标检测配置"""
    icon_base_paths: List[str] = None  # 图标搜索基础路径列表
    default_threshold: float = 0.8  # 默认相似度阈值
    scale_range: tuple = (0.5, 2.0)  # 缩放范围
    scale_step: float = 0.1  # 缩放步长
    nms_threshold: float = 0.3  # 非极大值抑制阈值
    
    def __post_init__(self):
        if self.icon_base_paths is None:
            # 设置默认图标路径
            project_root = self._find_project_root()
            self.icon_base_paths = [
                os.path.join(project_root, 'task_configs', 'icons'),
                os.path.join(project_root, 'assets', 'icons'),
            ]
            # 过滤存在的路径
            self.icon_base_paths = [p for p in self.icon_base_paths if os.path.exists(p)]
    
    def _find_project_root(self) -> str:
        """查找项目根目录"""
        current_dir = Path(__file__).parent
        
        # 向上查找，直到找到包含特定标识文件的目录
        markers = ['pyproject.toml', 'requirements.txt', '.git', 'README.md']
        
        for _ in range(10):  # 最多向上查找10级
            for marker in markers:
                if (current_dir / marker).exists():
                    return str(current_dir)
            current_dir = current_dir.parent
            
        # 如果找不到，返回当前目录的上级（在utils中）
        return str(Path(__file__).parent.parent)


@dataclass
class ToolsConfig:
    """工具统一配置"""
    ocr: OCRConfig = None
    icon_detection: IconDetectionConfig = None
    weights_dir: str = None  # 模型权重目录
    
    def __post_init__(self):
        if self.ocr is None:
            self.ocr = OCRConfig()
        if self.icon_detection is None:
            self.icon_detection = IconDetectionConfig()
        if self.weights_dir is None:
            project_root = self._find_project_root()
            self.weights_dir = os.path.join(project_root, 'weights')
    
    def _find_project_root(self) -> str:
        """查找项目根目录"""
        current_dir = Path(__file__).parent
        
        # 向上查找，直到找到包含特定标识文件的目录
        markers = ['pyproject.toml', 'requirements.txt', '.git', 'README.md']
        
        for _ in range(10):  # 最多向上查找10级
            for marker in markers:
                if (current_dir / marker).exists():
                    return str(current_dir)
            current_dir = current_dir.parent
            
        # 如果找不到，返回当前目录的上级（在utils中）
        return str(Path(__file__).parent.parent)


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_file: 配置文件路径，None时使用默认路径
        """
        if config_file is None:
            project_root = self._find_project_root()
            config_file = os.path.join(project_root, 'config', 'tools_config.json')
        
        self.config_file = config_file
        self.config = self._load_config()
    
    def _find_project_root(self) -> str:
        """查找项目根目录"""
        current_dir = Path(__file__).parent
        
        # 向上查找，直到找到包含特定标识文件的目录
        markers = ['pyproject.toml', 'requirements.txt', '.git', 'README.md']
        
        for _ in range(10):  # 最多向上查找10级
            for marker in markers:
                if (current_dir / marker).exists():
                    return str(current_dir)
            current_dir = current_dir.parent
            
        # 如果找不到，返回当前目录的上级（在utils中）
        return str(Path(__file__).parent.parent)
    
    def _load_config(self) -> ToolsConfig:
        """加载配置"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)
                return self._dict_to_config(config_dict)
            except Exception as e:
                logger.warning(f"加载配置文件失败: {e}，使用默认配置")
        
        return ToolsConfig()
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> ToolsConfig:
        """将字典转换为配置对象"""
        ocr_dict = config_dict.get('ocr', {})
        icon_dict = config_dict.get('icon_detection', {})
        
        ocr_config = OCRConfig(
            lang=ocr_dict.get('lang', 'chi_sim+eng'),
            use_paddle=ocr_dict.get('use_paddle'),
            paddle_config=ocr_dict.get('paddle_config', {}),
            tesseract_config=ocr_dict.get('tesseract_config', {})
        )
        
        icon_config = IconDetectionConfig(
            icon_base_paths=icon_dict.get('icon_base_paths'),
            default_threshold=icon_dict.get('default_threshold', 0.8),
            scale_range=tuple(icon_dict.get('scale_range', (0.5, 2.0))),
            scale_step=icon_dict.get('scale_step', 0.1),
            nms_threshold=icon_dict.get('nms_threshold', 0.3)
        )
        
        return ToolsConfig(
            ocr=ocr_config,
            icon_detection=icon_config,
            weights_dir=config_dict.get('weights_dir')
        )
    
    def _config_to_dict(self, config: ToolsConfig) -> Dict[str, Any]:
        """将配置对象转换为字典"""
        return {
            'ocr': asdict(config.ocr),
            'icon_detection': asdict(config.icon_detection),
            'weights_dir': config.weights_dir
        }
    
    def save_config(self):
        """保存配置到文件"""
        try:
            # 确保配置目录存在
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
            config_dict = self._config_to_dict(self.config)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, ensure_ascii=False, indent=2)
            logger.info(f"配置已保存到: {self.config_file}")
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
    
    def get_ocr_config(self) -> OCRConfig:
        """获取OCR配置"""
        return self.config.ocr
    
    def get_icon_detection_config(self) -> IconDetectionConfig:
        """获取图标检测配置"""
        return self.config.icon_detection
    
    def get_weights_dir(self) -> str:
        """获取权重文件目录"""
        return self.config.weights_dir
    
    def update_ocr_config(self, **kwargs):
        """更新OCR配置"""
        for key, value in kwargs.items():
            if hasattr(self.config.ocr, key):
                setattr(self.config.ocr, key, value)
            else:
                logger.warning(f"未知的OCR配置参数: {key}")
    
    def update_icon_detection_config(self, **kwargs):
        """更新图标检测配置"""
        for key, value in kwargs.items():
            if hasattr(self.config.icon_detection, key):
                setattr(self.config.icon_detection, key, value)
            else:
                logger.warning(f"未知的图标检测配置参数: {key}")
    
    def set_weights_dir(self, weights_dir: str):
        """设置权重文件目录"""
        self.config.weights_dir = weights_dir
    
    def add_icon_path(self, path: str):
        """添加图标搜索路径"""
        if path not in self.config.icon_detection.icon_base_paths:
            if os.path.exists(path):
                self.config.icon_detection.icon_base_paths.append(path)
                logger.info(f"添加图标搜索路径: {path}")
            else:
                logger.warning(f"图标路径不存在: {path}")
    
    def remove_icon_path(self, path: str):
        """移除图标搜索路径"""
        if path in self.config.icon_detection.icon_base_paths:
            self.config.icon_detection.icon_base_paths.remove(path)
            logger.info(f"移除图标搜索路径: {path}")


# 全局配置管理器实例
_global_config_manager = None

def get_config_manager(config_file: Optional[str] = None) -> ConfigManager:
    """获取全局配置管理器实例"""
    global _global_config_manager
    if _global_config_manager is None or config_file is not None:
        _global_config_manager = ConfigManager(config_file)
    return _global_config_manager

def get_ocr_config() -> OCRConfig:
    """获取OCR配置"""
    return get_config_manager().get_ocr_config()

def get_icon_detection_config() -> IconDetectionConfig:
    """获取图标检测配置"""
    return get_config_manager().get_icon_detection_config()

def get_weights_dir() -> str:
    """获取权重文件目录"""
    return get_config_manager().get_weights_dir()

def save_config():
    """保存当前配置"""
    get_config_manager().save_config()

# 环境变量支持
def _load_from_env():
    """从环境变量加载配置"""
    config_manager = get_config_manager()
    
    # OCR配置
    if os.getenv('OCR_LANG'):
        config_manager.update_ocr_config(lang=os.getenv('OCR_LANG'))
    
    if os.getenv('OCR_USE_PADDLE'):
        use_paddle = os.getenv('OCR_USE_PADDLE').lower() in ('true', '1', 'yes')
        config_manager.update_ocr_config(use_paddle=use_paddle)
    
    # 图标检测配置
    if os.getenv('ICON_BASE_PATHS'):
        paths = os.getenv('ICON_BASE_PATHS').split(':')
        config_manager.update_icon_detection_config(icon_base_paths=paths)
    
    if os.getenv('ICON_DEFAULT_THRESHOLD'):
        threshold = float(os.getenv('ICON_DEFAULT_THRESHOLD'))
        config_manager.update_icon_detection_config(default_threshold=threshold)
    
    # 权重目录
    if os.getenv('WEIGHTS_DIR'):
        config_manager.set_weights_dir(os.getenv('WEIGHTS_DIR'))

# 自动加载环境变量
_load_from_env()