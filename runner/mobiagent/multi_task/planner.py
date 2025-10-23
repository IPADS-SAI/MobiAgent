"""
Planner模块
包含任务分析、计划生成、artifact提取等逻辑
"""

import json
import logging
import re
from typing import Dict, Optional, Tuple
from pathlib import Path
from openai import OpenAI

from .models import Plan, Artifact
from .prompts import (
    PLANNER_TASK_ANALYSIS_PROMPT,
    PLANNER_PLAN_GENERATION_PROMPT,
    PLANNER_EXTRACT_ARTIFACT_PROMPT,
    PLANNER_NEXT_STEP_PROMPT
)

# 延迟导入，避免循环依赖
try:
    from utils.local_experience import PromptTemplateSearch
    from utils.load_md_prompt import load_prompt
except ImportError:
    PromptTemplateSearch = None
    load_prompt = None
    logging.warning("无法导入经验检索模块，get_app_package_name 功能将受限")

# 加载planner_oneshot.md prompt
planner_prompt_template = None
if load_prompt:
    try:
        planner_prompt_template = load_prompt("planner_oneshot.md")
    except Exception as e:
        logging.warning(f"无法加载planner_oneshot.md: {e}")

# 经验模板路径
current_file_path = Path(__file__).resolve()
current_dir = current_file_path.parent
default_template_path = current_dir.parent.parent.parent / "utils" / "experience" / "templates-new.json"

APP_LIST = {
    "携程": "ctrip.android.view",
    "同城": "com.tongcheng.android",
    "飞猪": "com.taobao.trip",
    "去哪儿": "com.Qunar",
    "华住会": "com.htinns",
    "饿了么": "me.ele",
    "支付宝": "com.eg.android.AlipayGphone",
    "淘宝": "com.taobao.taobao",
    "京东": "com.jingdong.app.mall",
    "美团": "com.sankuai.meituan",
    "滴滴出行": "com.sdu.didi.psnger",
    "微信": "com.tencent.mm",
    "微博": "com.sina.weibo",
    "华为商城": "com.vmall.client",
    "华为视频": "com.huawei.himovie",
    "华为音乐": "com.huawei.music",
    "华为应用市场": "com.huawei.appmarket",
    "拼多多": "com.xunmeng.pinduoduo",
    "小红书": "com.xingin.xhs",
    "QQ": "com.tencent.mobileqq",
    "bilibili": "tv.danmaku.bili",
    "爱奇艺": "com.qiyi.video",
    "腾讯视频": "com.tencent.qqlive",
    "优酷": "com.youku.phone",
    "知乎": "com.zhihu.android",
    "QQ音乐": "com.tencent.qqmusic",
    "网易云音乐": "com.netease.cloudmusic",
    "酷狗音乐": "com.kugou.android",
    "抖音": "com.ss.android.ugc.aweme",
    "高德地图": "com.autonavi.minimap",
    "咸鱼": "com.taobao.idlefish"
}

def get_app_package_name(
    task_description: str,
    planner_client: OpenAI = None,
    planner_model: str = "gemini-2.5-flash"
) -> Tuple[str, str, str]:
    """
    单阶段任务：本地检索经验，调用模型完成应用选择和任务描述生成
    
    Args:
        task_description: 原始任务描述
        planner_client: Planner客户端
        planner_model: Planner模型名称
    
    Returns:
        (app_name, package_name, final_desc): 应用名、包名、完善后的任务描述
    """
    if not planner_client:
        logging.error("planner_client 未提供，无法调用模型")
        raise ValueError("planner_client is required")
    
    if not PromptTemplateSearch or not planner_prompt_template:
        logging.error("经验检索模块未正确加载")
        raise ValueError("经验检索模块未正确加载")
    
    # 本地检索经验
    search_engine = PromptTemplateSearch()
    logging.info(f"Using template path: {default_template_path}")
    experience_content = search_engine.get_experience(task_description, str(default_template_path), 1)
    logging.info(f"检索到的相关经验:\n{experience_content}")

    # 构建Prompt
    prompt = planner_prompt_template.format(
        task_description=task_description,
        experience_content=experience_content
    )
    
    # 调用模型
    response_str = planner_client.chat.completions.create(
        model=planner_model,
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]
    ).choices[0].message.content
    
    logging.info(f"Planner 响应: \n{response_str}")
    
    # 解析模型响应，如果有</think>标签，则移除</think>标签及之前的内容
    if "</think>" in response_str:
        response_str = response_str.split("</think>")[-1].strip()
    
    response_json = parse_planner_response(response_str)
    if response_json is None:
        logging.error("无法解析模型响应为 JSON。")
        logging.error(f"原始响应内容: {response_str}")
        raise ValueError("无法解析模型响应为 JSON。")

    app_name = response_json.get("app_name")
    package_name = response_json.get("package_name")
    final_desc = response_json.get("final_task_description", task_description)
    
    return app_name, package_name, final_desc


def parse_planner_response(response_str: str) -> dict:
    """解析Planner响应"""
    try:
        return json.loads(response_str)
    except json.JSONDecodeError:
        try:
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_str, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            json_match = re.search(r'(\{.*?\})', response_str, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            raise ValueError("无法在响应中找到有效的JSON")
        except Exception as e:
            logging.error(f"JSON解析失败: {e}")
            logging.error(f"原始响应: {response_str}")
            raise ValueError(f"无法解析JSON响应: {e}")


def analyze_task(
    task_description: str,
    planner_client: OpenAI,
    planner_model: str,
    experience_results: str = "",
    temperature: float = 0.0
) -> Tuple[bool, Optional[str]]:
    """
    分析任务是单阶段还是多阶段
    
    Args:
        task_description: 任务描述
        planner_client: Planner客户端
        planner_model: Planner模型名称
        experience_results: 经验检索结果（可选）
        temperature: 采样温度
    
    Returns:
        (is_multi_stage, reason): (是否多阶段, 原因说明)
    """
    # 如果有经验检索模块，尝试检索相关经验
    if PromptTemplateSearch and default_template_path.exists():
        try:
            search_engine = PromptTemplateSearch()
            experience_results = search_engine.get_experience(
                task_description, 
                str(default_template_path), 
                2
            )
            logging.info(f"检索到的相关经验:\n{experience_results}")
        except Exception as e:
            logging.warning(f"经验检索失败: {e}")
            experience_results = ""
    
    prompt = PLANNER_TASK_ANALYSIS_PROMPT.format(
        task_description=task_description,
        experience_content=experience_results or "无相关经验"
    )
    
    logging.info(f"任务分析prompt: \n{prompt}")
    
    response = planner_client.chat.completions.create(
        model=planner_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    
    response_str = response.choices[0].message.content
    logging.info(f"任务分析响应: \n{response_str}")
    
    try:
        result = parse_planner_response(response_str)
        
        # 解析task_type字段
        task_type = result.get("task_type", "single")
        is_multi = (task_type == "multi")
        reason = result.get("reasoning", "")
        
        return is_multi, reason
    except Exception as e:
        logging.error(f"解析任务分析响应失败: {e}")
        # 默认返回单阶段
        return False, None


def generate_plan(
    task_description: str,
    planner_client: OpenAI,
    planner_model: str,
    experience_results: str = "",
    temperature: float = 0.0
) -> Optional[Plan]:
    """
    生成多阶段任务计划
    
    Args:
        task_description: 任务描述
        planner_client: Planner客户端
        planner_model: Planner模型名称
        experience_results: 经验检索结果（可选）
        temperature: 采样温度
    
    Returns:
        Plan对象，生成失败返回None
    """
    # 如果有经验检索模块，尝试检索相关经验
    if PromptTemplateSearch and default_template_path.exists():
        try:
            search_engine = PromptTemplateSearch()
            experience_results = search_engine.get_experience(
                task_description, 
                str(default_template_path), 
                2
            )
            logging.info(f"检索到的相关经验:\n{experience_results}")
        except Exception as e:
            logging.warning(f"经验检索失败: {e}")
            experience_results = ""
    
    prompt = PLANNER_PLAN_GENERATION_PROMPT.format(
        task_description=task_description,
        experience_content=experience_results or "无相关经验",
        app_list = APP_LIST
    )
    
    logging.info(f"计划生成prompt: \n{prompt}")
    
    response = planner_client.chat.completions.create(
        model=planner_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    
    response_str = response.choices[0].message.content
    logging.info(f"计划生成响应: \n{response_str}")
    
    try:
        plan_data = parse_planner_response(response_str)
        plan = Plan.model_validate(plan_data)
        logging.info(f"生成的计划: {plan}")
        return plan
    except Exception as e:
        logging.error(f"生成计划失败: {e}")
        return None


def extract_artifact(
    subtask_id: int,
    subtask_description: str,
    artifact_schema: Dict[str, str],
    subtask_output: Dict,
    planner_client: OpenAI,
    planner_model: str,
    temperature: float = 0.0
) -> Optional[Artifact]:
    """
    从子任务输出中提取artifact
    
    Args:
        subtask_id: 子任务ID
        subtask_description: 子任务描述
        artifact_schema: 需要提取的字段定义
        subtask_output: 子任务输出（包含actions、reacts等）
        planner_client: Planner客户端
        planner_model: Planner模型名称
        temperature: 采样温度
    
    Returns:
        Artifact对象，提取失败返回None
    """
    # 提取执行历史
    execution_history = ""
    if subtask_output.get("reacts"):
        history_items = []
        for i, react in enumerate(subtask_output["reacts"][-5:], 1):  # 只取最后5步
            reasoning = react.get("reasoning", "")
            function_name = react.get("function", {}).get("name", "")
            history_items.append(f"{i}. {reasoning} -> {function_name}")
        execution_history = "\n".join(history_items)
    
    # 构建artifact_schema描述
    schema_desc = "\n".join([f"- {key}: {desc}" for key, desc in artifact_schema.items()])
    if not schema_desc:
        schema_desc = "无特定字段要求"
    
    # 提取最后一次截图
    last_screenshot = ""
    if subtask_output.get("screenshots"):
        last_screenshot = subtask_output["screenshots"][-1]
    
    prompt = PLANNER_EXTRACT_ARTIFACT_PROMPT.format(
        subtask_description=subtask_description,
        artifact_schema=schema_desc,
        execution_history=execution_history or "无执行历史",
        subtask_id=subtask_id
    )
    
    logging.info(f"Artifact提取prompt: \n{prompt}")
    
    messages = [{"role": "user", "content": prompt}]
    
    # 如果有截图，添加到消息中
    if last_screenshot:
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{last_screenshot}"}}
            ]
        }]
    
    response = planner_client.chat.completions.create(
        model=planner_model,
        messages=messages,
        temperature=temperature
    )
    
    response_str = response.choices[0].message.content
    logging.info(f"Artifact提取响应: \n{response_str}")
    
    try:
        artifact_data = parse_planner_response(response_str)
        artifact = Artifact.model_validate(artifact_data)
        logging.info(f"提取的artifact: {artifact}")
        return artifact
        
    except Exception as e:
        logging.error(f"提取artifact失败: {e}")
        # 返回一个失败的artifact
        return Artifact(
            subtask_id=subtask_id,
            success=False,
            summary=f"提取失败: {str(e)}",
            data={}
        )


def refine_next_subtask_description(
    next_subtask_description: str,
    artifacts: Dict[str, Artifact],
    planner_client: OpenAI,
    planner_model: str,
    temperature: float = 0.0
) -> str:
    """
    根据已有artifacts重写下一个子任务描述
    
    Args:
        next_subtask_description: 下一个子任务描述
        artifacts: 已有的artifacts字典
        planner_client: Planner客户端
        planner_model: Planner模型名称
        temperature: 采样温度
    
    Returns:
        重写后的子任务描述
    """
    if not artifacts:
        return next_subtask_description
    
    # 构建artifacts描述
    artifacts_desc = "\n".join([
        f"- {key}: {artifact.summary}\n  数据: {artifact.data}"
        for key, artifact in artifacts.items()
        if artifact.success  # 只包含成功的artifact
    ])
    
    if not artifacts_desc:
        artifacts_desc = "暂无可用的前置任务结果"
    
    prompt = PLANNER_NEXT_STEP_PROMPT.format(
        next_subtask_description=next_subtask_description,
        artifacts_desc=artifacts_desc
    )
    
    logging.info(f"子任务描述重写prompt: \n{prompt}")
    
    response = planner_client.chat.completions.create(
        model=planner_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    
    response_str = response.choices[0].message.content
    logging.info(f"子任务描述重写响应: \n{response_str}")
    
    try:
        result = parse_planner_response(response_str)
        refined_description = result.get("refined_description", next_subtask_description)
        logging.info(f"重写后的子任务描述: {refined_description}")
        return refined_description
    except Exception as e:
        logging.error(f"重写子任务描述失败: {e}")
        return next_subtask_description
