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
    设置日志
    
    Args:
        log_level: 日志级别
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    
    # 清除已有的handlers以避免重复
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
            
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('runner.log', encoding='utf-8')
        ]
    )
    
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


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='统一的GUI Agent任务执行器')
    
    # 基础参数
    parser.add_argument('--provider', type=str, default='mobiagent_step',
                      choices=['mobiagent_step', 'uitars', 'qwen'],
                      help='模型提供者 (默认: mobiagent_step)')
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
    
    # 任务相关
    parser.add_argument('--task-file', type=str, default=None,
                      help='任务文件路径 (task.json 或 task_mobiflow.json)')
    parser.add_argument('--task', type=str, default=None,
                      help='单个任务描述（直接指定任务）')
    parser.add_argument('--output-dir', type=str, default='results',
                      help='结果输出目录 (默认: results)')
    
    # MobiAgent特定参数
    parser.add_argument('--service-ip', type=str, default='localhost',
                      help='MobiAgent服务IP (默认: localhost)')
    parser.add_argument('--decider-port', type=int, default=8000,
                      help='MobiAgent Decider端口 (默认: 8000)')
    parser.add_argument('--grounder-port', type=int, default=8001,
                      help='MobiAgent Grounder端口 (默认: 8001)')
    parser.add_argument('--planner-port', type=int, default=8080,
                      help='MobiAgent Planner端口 (默认: 8080)')
    parser.add_argument('--planner-model', type=str, default='Qwen3-VL-30B-A3B-Instruct',
                      help='Planner模型名称 (默认: Qwen3-VL-30B-A3B-Instruct)')
    parser.add_argument('--enable-planning', action='store_true', default=False,
                      help='启用任务规划（自动分析APP和优化任务描述）')
    parser.add_argument('--use-qwen3', action='store_true', default=True,
                      help='使用Qwen3模型')
    parser.add_argument('--use-e2e', action='store_true', default=False,
                      help='使用端到端模式')
    parser.add_argument('--decider-model', type=str, default='MobiMind-1.5-4B',
                      help='Decider模型名称 (默认: MobiMind-1.5-4B)')
    parser.add_argument('--grounder-model', type=str, default='MobiMind-1.5-4B',
                      help='Grounder模型名称 (默认: MobiMind-1.5-4B)')
    parser.add_argument('--use-experience', action='store_true', default=False,
                      help='使用经验')
    parser.add_argument('--use-graphrag', action='store_true', default=False,
                      help='使用GraphRAG')
    
    # UI-TARS特定参数
    parser.add_argument('--model-url', type=str, default='http://123.60.91.241:9003/v1',
                      help='UI-TARS模型服务地址')
    parser.add_argument('--model-name', type=str, default='UI-TARS-1.5-7B',
                      help='UI-TARS模型名称')
    parser.add_argument('--temperature', type=float, default=0.7,
                      help='温度参数 (默认: 0.7)')
    parser.add_argument('--step-delay', type=float, default=2.0,
                      help='步骤延迟秒数 (默认: 2.0)')
    
    # Qwen/VLLM SPECIFIC ARGS
    parser.add_argument('--qwen-api-key', type=str, default="",
                      help='API Key for Qwen/VLLM')
    parser.add_argument('--qwen-api-base', type=str, default="",
                      help='API Base URL for Qwen/VLLM')
    parser.add_argument('--qwen-model', type=str, default='Qwen3-VL-30B-A3B-Instruct',
                      help='Model name for Qwen/VLLM')
    
    # 可视化参数
    parser.add_argument('--draw', action='store_true', default=False,
                      help='是否在截图上绘制操作可视化 (默认: False)')
    
    return parser.parse_args()


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
    
    # 准备kwargs参数
    kwargs = {}
    
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
            "model_base_url": args.model_url,
            "model_name": args.model_name,
            "temperature": args.temperature,
            "step_delay": args.step_delay,
            "device_ip": args.device_id,
        })
    elif provider == "qwen":
        kwargs.update({
            "api_key": args.qwen_api_key,
            "api_base": args.qwen_api_base,
            "model_name": args.qwen_model,
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
    # 判断是否为MobiFlow格式
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
