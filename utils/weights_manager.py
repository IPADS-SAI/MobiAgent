"""
模型权重管理器
统一管理各种模型权重文件的路径和加载
"""

import os
import logging
from typing import Dict, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class WeightsManager:
    """模型权重管理器"""
    
    def __init__(self, weights_dir: Optional[str] = None):
        """
        初始化权重管理器
        
        Args:
            weights_dir: 权重文件根目录，None时自动查找
        """
        if weights_dir is None:
            weights_dir = self._find_weights_dir()
        
        self.weights_dir = Path(weights_dir)
        if not self.weights_dir.exists():
            logger.warning(f"权重目录不存在: {self.weights_dir}")
        
        # 预定义的模型路径映射
        self.model_paths = {
            # OCR相关模型（PaddleOCR会自动下载，这里预留）
            'paddle_ocr_det': 'paddle_ocr/det',
            'paddle_ocr_rec': 'paddle_ocr/rec', 
            'paddle_ocr_cls': 'paddle_ocr/cls',
            
            # 图标检测相关模型
            'icon_detect': 'icon_detect',
            'icon_caption_florence': 'icon_caption_florence', 
            
            # 视觉分析相关模型
            'owlvit_base_patch32': 'owlvit-base-patch32',
            
            # 其他模型可以在此添加
        }
    
    def _find_weights_dir(self) -> str:
        """查找权重目录"""
        current_dir = Path(__file__).parent
        
        # 向上查找，直到找到包含weights目录的路径
        for _ in range(10):  # 最多向上查找10级
            weights_path = current_dir / 'weights'
            if weights_path.exists():
                return str(weights_path)
            current_dir = current_dir.parent
        
        # 如果找不到，返回项目根目录下的weights
        project_root = Path(__file__).parent.parent
        return str(project_root / 'weights')
    
    def get_model_path(self, model_name: str, create_if_missing: bool = True) -> Optional[str]:
        """
        获取模型权重路径
        
        Args:
            model_name: 模型名称
            create_if_missing: 如果目录不存在是否创建
            
        Returns:
            模型权重路径，找不到返回None
        """
        if model_name not in self.model_paths:
            logger.warning(f"未知的模型名称: {model_name}")
            # 尝试直接拼接路径
            model_path = self.weights_dir / model_name
        else:
            model_path = self.weights_dir / self.model_paths[model_name]
        
        if not model_path.exists():
            if create_if_missing:
                try:
                    model_path.mkdir(parents=True, exist_ok=True)
                    logger.info(f"创建模型目录: {model_path}")
                except Exception as e:
                    logger.error(f"创建模型目录失败: {e}")
                    return None
            else:
                logger.warning(f"模型路径不存在: {model_path}")
                return None
        
        return str(model_path)
    
    def list_available_models(self) -> Dict[str, str]:
        """
        列出所有可用的模型
        
        Returns:
            模型名称到路径的映射
        """
        available = {}
        
        for model_name, relative_path in self.model_paths.items():
            model_path = self.weights_dir / relative_path
            if model_path.exists():
                available[model_name] = str(model_path)
        
        return available
    
    def add_model_path(self, model_name: str, relative_path: str):
        """
        添加新的模型路径映射
        
        Args:
            model_name: 模型名称
            relative_path: 相对于weights_dir的路径
        """
        self.model_paths[model_name] = relative_path
        logger.info(f"添加模型路径映射: {model_name} -> {relative_path}")
    
    def get_icon_detect_model_path(self) -> Optional[str]:
        """获取图标检测模型路径"""
        return self.get_model_path('icon_detect')
    
    def get_icon_caption_model_path(self) -> Optional[str]:
        """获取图标标注模型路径"""
        return self.get_model_path('icon_caption_florence')
    
    def get_owlvit_model_path(self) -> Optional[str]:
        """获取OwlViT模型路径"""
        return self.get_model_path('owlvit_base_patch32')
    
    def validate_models(self) -> Dict[str, bool]:
        """
        验证所有模型是否存在
        
        Returns:
            模型名称到存在状态的映射
        """
        validation_results = {}
        
        for model_name in self.model_paths:
            model_path = self.get_model_path(model_name, create_if_missing=False)
            validation_results[model_name] = model_path is not None and Path(model_path).exists()
        
        return validation_results
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        获取模型详细信息
        
        Args:
            model_name: 模型名称
            
        Returns:
            模型信息字典
        """
        model_path = self.get_model_path(model_name, create_if_missing=False)
        if not model_path:
            return {'exists': False}
        
        path_obj = Path(model_path)
        info = {
            'exists': True,
            'path': str(path_obj),
            'is_directory': path_obj.is_dir(),
            'size_mb': 0,
            'files': []
        }
        
        try:
            if path_obj.is_dir():
                # 计算目录大小和文件列表
                total_size = 0
                files = []
                for file_path in path_obj.rglob('*'):
                    if file_path.is_file():
                        size = file_path.stat().st_size
                        total_size += size
                        files.append({
                            'name': file_path.name,
                            'relative_path': str(file_path.relative_to(path_obj)),
                            'size_mb': round(size / (1024 * 1024), 2)
                        })
                info['size_mb'] = round(total_size / (1024 * 1024), 2)
                info['files'] = files
            else:
                # 单个文件
                size = path_obj.stat().st_size
                info['size_mb'] = round(size / (1024 * 1024), 2)
                info['files'] = [{'name': path_obj.name, 'size_mb': info['size_mb']}]
        except Exception as e:
            logger.error(f"获取模型信息失败: {e}")
            info['error'] = str(e)
        
        return info


# 全局权重管理器实例
_global_weights_manager = None

def get_weights_manager(weights_dir: Optional[str] = None) -> WeightsManager:
    """获取全局权重管理器实例"""
    global _global_weights_manager
    if _global_weights_manager is None or weights_dir is not None:
        _global_weights_manager = WeightsManager(weights_dir)
    return _global_weights_manager

def get_model_path(model_name: str) -> Optional[str]:
    """获取模型路径的便捷函数"""
    return get_weights_manager().get_model_path(model_name)

def list_available_models() -> Dict[str, str]:
    """列出可用模型的便捷函数"""
    return get_weights_manager().list_available_models()

def validate_all_models() -> Dict[str, bool]:
    """验证所有模型的便捷函数"""
    return get_weights_manager().validate_models()