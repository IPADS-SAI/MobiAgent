import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class TaskDescriptionExtractor:
    """任务描述提取器"""
    
    def __init__(self):
        self.supported_files = ['actions.json']
    
    def extract_from_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        从指定目录及其子目录中提取所有任务描述
        
        Args:
            directory_path: 目标目录路径
            
        Returns:
            包含任务描述信息的字典列表
        """
        task_descriptions = []
        directory = Path(directory_path)
        
        if not directory.exists():
            logger.error(f"目录不存在: {directory_path}")
            return task_descriptions
        
        # 遍历目录及子目录
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file in self.supported_files:
                    file_path = Path(root) / file
                    task_info = self._extract_from_file(file_path)
                    if task_info:
                        task_info['source_path'] = str(file_path)
                        task_info['source_dir'] = str(Path(root))
                        task_descriptions.append(task_info)
        
        logger.info(f"从 {directory_path} 提取到 {len(task_descriptions)} 个任务描述")
        return task_descriptions
    
    def _extract_from_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        从单个文件中提取任务描述
        
        Args:
            file_path: 文件路径
            
        Returns:
            任务信息字典或None
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 提取关键信息
            task_info = {
                'task_description': data.get('task_description', ''),
                'old_task_description': data.get('old_task_description', ''),
                'app_name': data.get('app_name', ''),
                'task_type': data.get('task_type'),
                'action_count': data.get('action_count', 0),
                'actions': data.get('actions', [])
            }
            
            # 验证必要字段
            if not task_info['task_description']:
                logger.warning(f"文件 {file_path} 中缺少task_description字段")
                return None
            
            return task_info
            
        except json.JSONDecodeError as e:
            logger.error(f"解析JSON文件失败 {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"读取文件失败 {file_path}: {e}")
            return None
    
    def group_by_app(self, task_descriptions: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        按应用名称分组任务描述
        
        Args:
            task_descriptions: 任务描述列表
            
        Returns:
            按app_name分组的字典
        """
        grouped = {}
        for task in task_descriptions:
            app_name = task.get('app_name', 'unknown')
            if app_name not in grouped:
                grouped[app_name] = []
            grouped[app_name].append(task)
        
        return grouped
    
    def group_by_task_type(self, task_descriptions: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        按任务类型分组任务描述
        
        Args:
            task_descriptions: 任务描述列表
            
        Returns:
            按task_type分组的字典
        """
        grouped = {}
        for task in task_descriptions:
            task_type = task.get('task_type', 'general')
            if task_type not in grouped:
                grouped[task_type] = []
            grouped[task_type].append(task)
        
        return grouped


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    extractor = TaskDescriptionExtractor()
    
    # 测试提取
    test_dir = "./data"
    task_descriptions = extractor.extract_from_directory(test_dir)
    
    print(f"提取到 {len(task_descriptions)} 个任务描述:")
    for task in task_descriptions:
        print(f"- {task['app_name']}: {task['task_description']}")
    
    # 测试分组
    grouped_by_app = extractor.group_by_app(task_descriptions)
    print(f"\n按应用分组:")
    for app, tasks in grouped_by_app.items():
        print(f"- {app}: {len(tasks)} 个任务")
