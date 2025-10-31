"""
用户偏好模板配置
从JSON文件加载不同任务类型的偏好提取方面，用于指导LLM进行更精准的偏好提取
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

# 全局变量，用于缓存加载的模板
_PREFERENCE_TEMPLATES = None

def load_preference_templates() -> Dict:
    """
    从JSON文件加载偏好模板配置
    
    Returns:
        偏好模板配置字典
    """
    global _PREFERENCE_TEMPLATES
    
    if _PREFERENCE_TEMPLATES is not None:
        return _PREFERENCE_TEMPLATES
    
    try:
        # 获取当前文件所在目录
        current_dir = Path(__file__).parent
        json_file = current_dir / "preference_templates.json"
        
        if not json_file.exists():
            logging.error(f"偏好模板配置文件不存在: {json_file}")
            return {}
        
        with open(json_file, 'r', encoding='utf-8') as f:
            _PREFERENCE_TEMPLATES = json.load(f)
        
        logging.info(f"成功加载偏好模板配置: {len(_PREFERENCE_TEMPLATES)} 个任务类型")
        return _PREFERENCE_TEMPLATES
        
    except Exception as e:
        logging.error(f"加载偏好模板配置失败: {e}")
        return {}

def get_preference_templates() -> Dict:
    """
    获取偏好模板配置
    
    Returns:
        偏好模板配置字典
    """
    return load_preference_templates()

def identify_task_type(task_description: str) -> str:
    """
    根据任务描述识别任务类型
    
    Args:
        task_description: 任务描述
        
    Returns:
        任务类型
    """
    templates = get_preference_templates()
    task_description_lower = task_description.lower()
    
    for task_type, config in templates.items():
        if task_type == "general":
            continue
            
        keywords = config.get("keywords", [])
        for keyword in keywords:
            if keyword in task_description_lower:
                return task_type
    
    return "general"

def get_preference_aspects(task_type: str) -> List[str]:
    """
    获取指定任务类型的偏好提取方面
    
    Args:
        task_type: 任务类型
        
    Returns:
        偏好提取方面列表
    """
    templates = get_preference_templates()
    config = templates.get(task_type, templates.get("general", {}))
    return config.get("extraction_aspects", [])

def get_example_prompts(task_type: str) -> List[str]:
    """
    获取指定任务类型的示例查询问题
    
    Args:
        task_type: 任务类型
        
    Returns:
        示例查询问题列表
    """
    templates = get_preference_templates()
    config = templates.get(task_type, templates.get("general", {}))
    return config.get("example_prompts", [])

def generate_preference_extraction_prompt(task_data: dict, task_type: str) -> str:
    """
    生成偏好提取的prompt
    
    Args:
        task_data: 任务数据
        task_type: 任务类型
        
    Returns:
        生成的prompt
    """
    templates = get_preference_templates()
    config = templates.get(task_type, templates.get("general", {}))
    extraction_aspects = config.get("extraction_aspects", [])
    
    aspects_text = "\n".join([f"- {aspect}" for aspect in extraction_aspects])
    
    prompt = f"""
基于以下任务执行记录，请分析用户偏好：

任务描述：{task_data['task_description']}
执行记录：{task_data['actions']}
推理过程：{task_data['reacts']}

请重点关注以下方面的用户偏好：
{aspects_text}

请返回任务类型和用户偏好信息。

输出格式：
```json
{{
    "task_type": "{config.get('task_type', '通用')}",
    "preferences": {{
        "偏好1": "值1",
        "偏好2": "值2"
    }}
}}
```
"""
    
    return prompt

def get_all_task_types() -> List[str]:
    """
    获取所有支持的任务类型
    
    Returns:
        任务类型列表
    """
    templates = get_preference_templates()
    return list(templates.keys())

def get_task_type_config(task_type: str) -> Dict:
    """
    获取指定任务类型的完整配置
    
    Args:
        task_type: 任务类型
        
    Returns:
        任务类型配置
    """
    templates = get_preference_templates()
    return templates.get(task_type, templates.get("general", {}))

# 使用示例
if __name__ == "__main__":
    # 测试JSON配置加载
    print("=== JSON配置加载测试 ===")
    templates = get_preference_templates()
    print(f"加载的任务类型数量: {len(templates)}")
    print(f"支持的任务类型: {list(templates.keys())}")
    print()
    
    # 测试任务类型识别
    test_tasks = [
        "在携程上预订北京的一家商务酒店",
        "在淘宝上搜索飞利浦电动牙刷",
        "在12306上购买火车票",
        "在饿了么上订一份外卖",
        "在bilibili上看视频"
    ]
    
    print("=== 任务类型识别测试 ===")
    for task in test_tasks:
        task_type = identify_task_type(task)
        aspects = get_preference_aspects(task_type)
        print(f"任务: {task}")
        print(f"识别类型: {task_type}")
        print(f"提取方面数量: {len(aspects)}")
        print(f"前3个方面: {aspects[:3]}")
        print()
    
    # 测试prompt生成
    print("=== Prompt生成测试 ===")
    task_data = {
        'task_description': '在携程上预订北京的一家商务酒店',
        'actions': [{"type": "click", "position_x": 500, "position_y": 200}],
        'reacts': [{"reasoning": "用户要预订酒店"}]
    }
    
    task_type = identify_task_type(task_data['task_description'])
    prompt = generate_preference_extraction_prompt(task_data, task_type)
    print(f"生成的Prompt长度: {len(prompt)} 字符")
    print("Prompt预览:")
    print(prompt[:300] + "...")
