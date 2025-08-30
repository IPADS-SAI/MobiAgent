"""
LLM提示词模块
统一管理所有LLM相关的提示词内容
"""

import json
from typing import List


# 系统提示词
SYSTEM_PROMPT = """你是一个专业的任务验证配置生成专家，擅长根据用户任务描述和参考模版生成精确的YAML验证配置文件,用于检测判定任务的关键节点。

## 满足以下要求：

1. **优化OCR关键词**: 根据实际任务描述，为每个节点的OCR检查添加更准确和全面的关键词列表，关键词不和具体任务的具体内容相关，而是针对任务类型和常见操作的通用关键词。
2. **改进LLM提示词**: 让LLM验证提示更具体、更符合实际场景，且能准确判断任务节点是否完成。
3. **优化节点路径**: 根据任务模式调整节点之间的依赖关系和路径选择,节点和路径针对该类任务的共性进行优化。
4. **调整检查器类型**: 根据验证需求选择合适的escalate或juxtaposition类型（严格但耗时）。
5. **针对关键节点**: 配置文件应重点关注任务中的关键节点，确保这些节点的验证准确有效，最终完成状态能代表整个任务完成，不一定要覆盖所有步骤

## 配置规则约束：
- 只能使用type: escalate 或 type: juxtaposition
- params只能包含ocr和llm字段
- ocr可以使用any或all,使用[]格式
- llm必须包含prompt和expected_true字段
- 保持deps (AND语义) 和 next (OR语义) 的正确使用，且使用[,]格式

请直接返回优化后的完整YAML配置，不要包含其他解释文字。"""


# 配置要求模板
CONFIG_REQUIREMENTS = """## 配置要求：

1. **基本信息**:
   - 生成合适的task_id、app_id、task_type和description
   - 基于实际的应用名称和任务类型

2. **节点设计**:
   - 根据任务流程设计合理的节点序列
   - 使用deps (AND语义) 表示强制依赖
   - 使用next (OR语义) 表示可选路径
   - 确保节点间的逻辑关系正确

3. **条件检查**:
   - type只能是escalate或juxtaposition
   - params只能包含ocr和llm字段
   - ocr使用any表示任意匹配，all表示全部匹配
   - llm包含prompt和expected_true字段
   - 根据任务特点选择合适的关键词

4. **成功条件**:
   - 设置合理的success条件
   - 使用any_of或all_of"""


# 配置示例格式
CONFIG_EXAMPLE = """## 参考配置示例格式：

```yaml
task_id: example_task
app_id: com.example.app
task_type: demo
description: 示例任务描述

nodes:
  - id: step1
    name: 第一步
    next: [step2]
    condition:
      type: escalate
      params:
        ocr:
          any: ["关键词1", "关键词2"]
        llm:
          prompt: "该步是否完成了XXX操作？"
          expected_true: true
    

  - id: step2
    name: 第二步
    condition:
      type: juxtaposition
      params:
        ocr:
          all: ["确认", "完成"]
        llm:
          prompt: "该步是否显示了完成状态？"
          expected_true: true

success:
  any_of: [step2]
```"""


def generate_user_prompt(task_descriptions: List[str], template_yaml: str, app_name: str = "unknown") -> str:
    """
    生成用户提示词
    
    Args:
        task_descriptions: 任务描述列表
        template_yaml: 参考模版YAML内容
        app_name: 应用名称
        
    Returns:
        完整的用户提示词
    """
    prompt = f"""请根据以下任务描述和参考模版，生成一个完整的任务验证DAG配置：

## 任务描述列表
{json.dumps(task_descriptions, ensure_ascii=False, indent=2)}

## 参考模版配置
以下是参考模版的结构和格式：

```yaml
{template_yaml}
```

{CONFIG_REQUIREMENTS}

{CONFIG_EXAMPLE}

请直接返回完整的YAML配置，不要包含其他解释文字。"""
    
    return prompt.strip()
