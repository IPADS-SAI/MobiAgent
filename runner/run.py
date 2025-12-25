#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

"""
统一的任务执行入口
支持多种模型（MobiAgent, UI-TARS等）
"""

import os
import sys
import json
import logging
import argparse
import time
from datetime import datetime
from typing import Dict, List, Optional

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from task_manager import TaskManager
from device import create_device, AndroidDevice, HarmonyDevice


def setup_logging(log_level: str = "INFO"):
    """
    设置日志 - 配置根logger和所有模块logger
    
    Args:
        log_level: 日志级别 ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    
    # 清除已有的handlers以避免重复
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers[:]:
            root.removeHandler(handler)
    
    # 设置根logger级别
    root.setLevel(numeric_level)
    
    # 创建日志格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 添加标准输出处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)
    
    # 添加文件处理器
    try:
        file_handler = logging.FileHandler('runner.log', encoding='utf-8')
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Failed to setup file logging: {e}")
    
    # 配置所有已知模块的logger
    module_loggers = [
        'task_manager',
        'base_task',
        'device',
        'providers.mobiagent.mobile_task',
        'providers.mobiagent.load_md_prompt',
        'providers.qwen.qwen_task',
        'providers.qwen.utils',
        'providers.uitars.uitars_task',
        'providers.uitars.ui_tars_helper',
        'providers.autoglm.autoglm_task',
    ]
    
    for logger_name in module_loggers:
        module_logger = logging.getLogger(logger_name)
        module_logger.setLevel(numeric_level)
    
    sys.stdout.flush()


def load_tasks(task_file: str) -> List:
    """
    从文件加载任务列表
    
    Args:
        task_file: 任务文件路径
        
    Returns:
        任务列表
    """
    with open(task_file, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
    return tasks


# Provider 默认配置
PROVIDER_DEFAULTS = {
    'mobiagent': {
        'api_base': 'http://localhost:8000/v1',
        'model': 'MobiMind-1.5-4B',
        'temperature': 0.1,
    },
    'mobiagent_step': {
        'api_base': 'http://localhost:8000/v1',
        'model': 'MobiMind-1.5-4B',
        'temperature': 0.1,
    },
    'uitars': {
        'api_base': 'http://localhost:8000/v1',
        'model': 'UI-TARS-1.5-7B',
        'temperature': 0.0,
    },
    'qwen': {
        'api_base': 'http://localhost:8080/v1',
        'model': 'Qwen3-VL-30B-A3B-Instruct',
        'temperature': 0.0,
    },
    'autoglm': {
        'api_base': 'http://localhost:8000/v1',
        'model': 'autoglm-phone-9b',
        'temperature': 0.0,
    },
}


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='统一的GUI Agent任务执行器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用 MobiAgent 执行单任务
  python run.py --provider mobiagent --task "打开淘宝搜索手机" --api-base http://localhost:8000/v1
  
  # 使用 UI-TARS 执行任务
  python run.py --provider uitars --task "打开微信" --api-base http://localhost:8000/v1 --model UI-TARS-1.5-7B
  
  # 使用 Qwen VLM 执行任务
  python run.py --provider qwen --task "在微博查看新闻" --api-base http://localhost:8080/v1
"""
    )
    
    # ==================== 基础参数 ====================
    parser.add_argument('--provider', type=str, default='mobiagent_step',
                      choices=['mobiagent', 'mobiagent_step', 'uitars', 'qwen', 'autoglm'],
                      help='模型提供者 (默认: mobiagent_step, mobiagent是mobiagent_step的别名)')
    parser.add_argument('--device-type', type=str, default='Android',
                      choices=['Android', 'Harmony'],
                      help='设备类型 (默认: Android)')
    parser.add_argument('--device-id', type=str, default=None,
                      help='设备ID或IP地址')
    parser.add_argument('--max-steps', type=int, default=40,
                      help='最大步骤数 (默认: 40)')
    parser.add_argument('--log-level', type=str, default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                      help='日志级别 (默认: INFO)')
    
    # ==================== 任务相关 ====================
    parser.add_argument('--task-file', type=str, default=None,
                      help='任务文件路径 (task.json 或 task_mobiflow.json)')
    parser.add_argument('--task', type=str, default=None,
                      help='单个任务描述（直接指定任务）')
    parser.add_argument('--output-dir', type=str, default='results',
                      help='结果输出目录 (默认: results)')
    
    # ==================== 通用模型参数 ====================
    parser.add_argument('--api-base', type=str, default=None,
                      help='模型服务基础URL (通用参数, 默认按provider自动设置)')
    parser.add_argument('--api-key', type=str, default='',
                      help='API密钥 (通用参数, 无需验证时可留空)')
    parser.add_argument('--model', type=str, default=None,
                      help='模型名称 (通用参数, 默认按provider自动设置)')
    parser.add_argument('--temperature', type=float, default=None,
                      help='生成温度 (通用参数, 默认按provider自动设置)')
    
    # ==================== MobiAgent 专属参数 ====================
    mobiagent_group = parser.add_argument_group('MobiAgent 专属参数')
    mobiagent_group.add_argument('--service-ip', type=str, default='localhost',
                      help='MobiAgent服务IP (默认: localhost)')
    mobiagent_group.add_argument('--decider-port', type=int, default=8000,
                      help='Decider端口 (默认: 8000)')
    mobiagent_group.add_argument('--grounder-port', type=int, default=8001,
                      help='Grounder端口 (默认: 8001)')
    mobiagent_group.add_argument('--planner-port', type=int, default=8080,
                      help='Planner端口 (默认: 8080)')
    mobiagent_group.add_argument('--planner-model', type=str, default='Qwen3-VL-30B-A3B-Instruct',
                      help='Planner模型名称 (默认: Qwen3-VL-30B-A3B-Instruct)')
    mobiagent_group.add_argument('--enable-planning', action='store_true', default=False,
                      help='启用任务规划（自动分析APP和优化任务描述）')
    mobiagent_group.add_argument('--use-e2e', action='store_true', default=True,
                      help='使用端到端模式 (默认: True)')
    mobiagent_group.add_argument('--decider-model', type=str, default='MobiMind-1.5-4B',
                      help='Decider模型名称 (默认: MobiMind-1.5-4B)')
    mobiagent_group.add_argument('--grounder-model', type=str, default='MobiMind-1.5-4B',
                      help='Grounder模型名称 (默认: MobiMind-1.5-4B)')
    mobiagent_group.add_argument('--use-experience', action='store_true', default=False,
                      help='使用经验')
    
    # ==================== UI-TARS 专属参数 ====================
    uitars_group = parser.add_argument_group('UI-TARS 专属参数')
    uitars_group.add_argument('--step-delay', type=float, default=2.0,
                      help='步骤延迟秒数 (默认: 2.0)')
    
    # ==================== 向后兼容的旧参数 (deprecated) ====================
    compat_group = parser.add_argument_group('向后兼容参数 (deprecated, 建议使用通用参数)')
    compat_group.add_argument('--model-url', type=str, default=None,
                      help='[deprecated] 模型服务地址, 请使用 --api-base')
    compat_group.add_argument('--model-name', type=str, default=None,
                      help='[deprecated] 模型名称, 请使用 --model')
    compat_group.add_argument('--qwen-api-key', type=str, default=None,
                      help='[deprecated] Qwen API密钥, 请使用 --api-key')
    compat_group.add_argument('--qwen-api-base', type=str, default=None,
                      help='[deprecated] Qwen API地址, 请使用 --api-base')
    compat_group.add_argument('--qwen-model', type=str, default="Qwen3-VL-30B-A3B-Instruct",
                      help='[deprecated] Qwen模型名称, 请使用 --model')
    
    # ==================== 可视化参数 ====================
    parser.add_argument('--draw', action='store_true', default=False,
                      help='是否在截图上绘制操作可视化 (默认: False)')
    
    args = parser.parse_args()
    
    # 处理 provider 别名
    if args.provider == 'mobiagent':
        args.provider = 'mobiagent_step'
    
    # 获取 provider 默认值
    defaults = PROVIDER_DEFAULTS.get(args.provider, {})
    
    # 合并向后兼容参数到统一参数
    # api-base 优先级: --api-base > --model-url > --qwen-api-base > provider默认值
    if args.api_base is None:
        args.api_base = args.model_url or args.qwen_api_base or defaults.get('api_base')
    
    # model 优先级: --model > --model-name > --qwen-model > provider默认值
    if args.model is None:
        args.model = args.model_name or args.qwen_model or defaults.get('model')
    
    # api-key 优先级: --api-key > --qwen-api-key
    if not args.api_key and args.qwen_api_key:
        args.api_key = args.qwen_api_key
    
    # temperature 使用 provider 默认值
    if args.temperature is None:
        args.temperature = defaults.get('temperature', 0.0)
    
    return args


def create_device(device_type: str, device_id: Optional[str] = None):
    """
    创建设备对象（使用独立的device模块）
    
    Args:
        device_type: 设备类型
        device_id: 设备ID或IP
        
    Returns:
        设备对象
    """
    from device import create_device as factory_create_device
    device = factory_create_device(device_type, adb_endpoint=device_id)
    logging.info(f"已连接到 {device_type} 设备")
    return device


def execute_single_task(
    provider: str,
    task_description: str,
    device,
    output_dir: str,
    device_type: str,
    args,
    app_name: Optional[str] = None,
    task_type: Optional[str] = None
) -> Dict:
    """
    执行单个任务
    
    Args:
        provider: 模型提供者
        task_description: 任务描述
        device: 设备对象
        output_dir: 输出目录
        device_type: 设备类型
        args: 命令行参数
        app_name: APP名称 (可选)
        task_type: 任务类型 (可选)
        
    Returns:
        执行结果字典
    """
    # 创建任务特定的输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 清理任务描述中的特殊字符
    safe_task = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' 
                       for c in task_description)[:50]
    
    if app_name and task_type:
        task_dir = os.path.join(output_dir, provider, app_name, task_type, f"{timestamp}_{safe_task}")
    else:
        task_dir = os.path.join(output_dir, provider, f"{timestamp}_{safe_task}")
        
    os.makedirs(task_dir, exist_ok=True)
    
    logging.info(f"=" * 60)
    logging.info(f"开始执行任务: {task_description}")
    logging.info(f"Provider: {provider}")
    logging.info(f"输出目录: {task_dir}")
    if app_name and task_type:
        logging.info(f"App: {app_name}, Type: {task_type}")
    logging.info(f"=" * 60)
    
    # 准备kwargs参数 - 使用统一的参数命名
    kwargs = {
        # 通用参数
        "api_base": args.api_base,
        "api_key": args.api_key,
        "model": args.model,
        "temperature": args.temperature,
    }
    
    if provider == "mobiagent_step":
        kwargs.update({
            "service_ip": args.service_ip,
            "decider_port": args.decider_port,
            "grounder_port": args.grounder_port,
            "planner_port": args.planner_port,
            "enable_planning": args.enable_planning,
            "use_e2e": args.use_e2e,
            "decider_model": args.decider_model,
            "grounder_model": args.grounder_model,
            "planner_model": args.planner_model,
            "use_experience": args.use_experience,
        })
    elif provider == "uitars":
        kwargs.update({
            "step_delay": args.step_delay,
            "device_ip": args.device_id,
            # 向后兼容: 同时传递旧参数名
            "model_base_url": args.api_base,
            "model_name": args.model,
        })
    elif provider == "qwen":
        kwargs.update({
            # 向后兼容: 同时传递旧参数名
            "model_name": args.model,
        })
    
    # 创建任务管理器
    try:
        task_manager = TaskManager(
            provider=provider,
            task_description=task_description,
            device=device,
            data_dir=task_dir,
            device_type=device_type,
            max_steps=args.max_steps,
            draw=args.draw,
            **kwargs
        )
        
        # 执行任务
        start_time = time.time()
        result = task_manager.execute()
        elapsed_time = time.time() - start_time
        
        result["elapsed_time"] = elapsed_time
        result["task_description"] = task_description
        result["output_dir"] = task_dir
        
        logging.info(f"任务完成! 状态: {result.get('status', 'unknown')}")
        logging.info(f"耗时: {elapsed_time:.2f}秒")
        logging.info(f"步数: {result.get('step_count', 0)}")
        
        return result
        
    except Exception as e:
        logging.error(f"任务执行失败: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "task_description": task_description,
            "output_dir": task_dir
        }


def execute_batch_tasks(
    provider: str,
    tasks: List,
    device,
    output_dir: str,
    device_type: str,
    args
) -> Dict:
    """
    批量执行任务
    
    Args:
        provider: 模型提供者
        tasks: 任务列表
        device: 设备对象
        output_dir: 输出目录
        device_type: 设备类型
        args: 命令行参数
        
    Returns:
        批量执行结果字典
    """
    results = []
    success_count = 0
    fail_count = 0
    error_count = 0
    
    total_tasks = len(tasks) if isinstance(tasks, list) else sum(
        len(app_data.get("tasks", [])) for app_data in tasks
    )
    
    logging.info(f"开始批量执行 {total_tasks} 个任务")
    
    # 处理不同格式的任务文件
    # 判断是否为MobiFlow格式(多APP多任务)
    is_mobiflow = False
    if isinstance(tasks, list) and len(tasks) > 0 and isinstance(tasks[0], dict):
        if "tasks" in tasks[0] and isinstance(tasks[0]["tasks"], list):
            is_mobiflow = True
            
    if is_mobiflow:
        # MobiFlow格式任务
        task_list = []
        for app_data in tasks:
            app_name = app_data.get("app", "unknown")
            task_type = app_data.get("type", "unknown")
            for task_desc in app_data.get("tasks", []):
                task_list.append({
                    "app": app_name,
                    "type": task_type,
                    "description": task_desc
                })
    else:
        # 简单列表格式或其他格式
        if isinstance(tasks, dict):
             task_list = []
             logging.warning("未知的任务格式")
        else:
             task_list = tasks
    
    for idx, task_item in enumerate(task_list, 1):
        # 提取任务描述和元数据
        app_name = None
        task_type = None
        
        if isinstance(task_item, str):
            task_description = task_item
        elif isinstance(task_item, dict):
            task_description = task_item.get("description", 
                                            task_item.get("task", str(task_item)))
            app_name = task_item.get("app")
            task_type = task_item.get("type")
        else:
            task_description = str(task_item)
        
        logging.info(f"\n[{idx}/{total_tasks}] 执行任务: {task_description}")
        
        result = execute_single_task(
            provider=provider,
            task_description=task_description,
            device=device,
            output_dir=output_dir,
            device_type=device_type,
            args=args,
            app_name=app_name,
            task_type=task_type
        )
        
        # 统计结果
        status = result.get("status", "unknown")
        if status == "success" or result.get("success", False):
            success_count += 1
        elif status == "failed":
            fail_count += 1
        else:
            error_count += 1
        
        results.append(result)
        
        # 任务间休息
        if idx < total_tasks:
            logging.info("等待3秒后执行下一个任务...")
            time.sleep(3)
    
    # 生成汇总报告
    summary = {
        "total_tasks": total_tasks,
        "success_count": success_count,
        "fail_count": fail_count,
        "error_count": error_count,
        "success_rate": success_count / total_tasks if total_tasks > 0 else 0,
        "results": results
    }
    
    # 保存汇总报告
    summary_path = os.path.join(output_dir, provider, "summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    logging.info(f"\n{'='*60}")
    logging.info(f"批量任务执行完成!")
    logging.info(f"总任务数: {total_tasks}")
    logging.info(f"成功: {success_count} ({success_count/total_tasks*100:.1f}%)")
    logging.info(f"失败: {fail_count}")
    logging.info(f"错误: {error_count}")
    logging.info(f"汇总报告: {summary_path}")
    logging.info(f"{'='*60}")
    
    return summary


def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    
    logging.info("=" * 60)
    logging.info("统一GUI Agent任务执行器")
    logging.info(f"Provider: {args.provider}")
    logging.info(f"Device Type: {args.device_type}")
    logging.info("=" * 60)
    
    # 创建设备
    try:
        device = create_device(args.device_type, args.device_id)
        logging.info(f"设备连接成功: {args.device_type}")
    except Exception as e:
        logging.error(f"设备连接失败: {e}")
        return 1
    
    # 确定任务
    if args.task:
        # 单个任务
        result = execute_single_task(
            provider=args.provider,
            task_description=args.task,
            device=device,
            output_dir=args.output_dir,
            device_type=args.device_type,
            args=args
        )
        return 0 if result.get("status") != "error" else 1
        
    elif args.task_file:
        # 批量任务
        if not os.path.exists(args.task_file):
            logging.error(f"任务文件不存在: {args.task_file}")
            return 1
        
        tasks = load_tasks(args.task_file)
        summary = execute_batch_tasks(
            provider=args.provider,
            tasks=tasks,
            device=device,
            output_dir=args.output_dir,
            device_type=args.device_type,
            args=args
        )
        return 0 if summary["error_count"] == 0 else 1
        
    else:
        logging.error("请指定 --task 或 --task-file 参数")
        return 1


if __name__ == "__main__":
    sys.exit(main())
