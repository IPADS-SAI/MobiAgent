#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单任务测试器 - 用于测试单个任务执行
"""

import json
import os
import time
import logging
from datetime import datetime
from ui_tars_automation import UITarsAutomationFramework, ExecutionConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_single_task():
    """测试单个任务"""
    print("UI-TARS 单任务测试器")
    print("=" * 50)
    
    # 应用包名映射
    app_packages = {
        "bilibili": "tv.danmaku.bili",
        "淘宝": "com.taobao.taobao",
        "携程": "ctrip.android.view",
        "网易云音乐": "com.netease.cloudmusic",
        "小红书": "com.xingin.xhs",
        "高德": "com.autonavi.minimap",
        "饿了么": "me.ele"
    }
    
    # 获取模型服务地址
    model_url = input("请输入模型服务地址 (默认: http://192.168.12.152:8000/v1): ").strip()
    if not model_url:
        model_url = "http://192.168.12.152:8000/v1"
    
    # 选择要测试的任务
    print("\n可用应用:")
    for i, app in enumerate(app_packages.keys(), 1):
        print(f"{i}. {app}")
    
    app_choice = input("\n请选择应用 (输入数字): ").strip()
    try:
        app_names = list(app_packages.keys())
        app_name = app_names[int(app_choice) - 1]
        package_name = app_packages[app_name]
    except (ValueError, IndexError):
        print("无效选择")
        return
    
    # 输入任务描述
    task_description = input(f"\n请输入{app_name}的任务描述: ").strip()
    if not task_description:
        print("任务描述不能为空")
        return
    
    print(f"\n将执行任务: {task_description}")
    print(f"目标应用: {app_name} ({package_name})")
    
    confirm = input("是否继续？(y/N): ").strip().lower()
    if confirm != 'y':
        print("已取消执行")
        return
    
    try:
        # 配置
        config = ExecutionConfig(
            model_base_url=model_url,
            model_name="UI-TARS-7B-SFT",
            max_steps=30,
            step_delay=2.0,
            language="Chinese",
            save_data=True,
            data_base_dir="test_data"
        )
        
        # 创建框架实例
        framework = UITarsAutomationFramework(config)
        print("✅ 框架初始化成功!")
        
        # 启动应用
        print(f"启动应用: {package_name}")
        framework.device.app_start(package_name, stop=True)
        time.sleep(3)  # 等待应用启动
        
        # 执行任务
        print(f"\n开始执行任务: {task_description}")
        print("-" * 50)
        
        success = framework.execute_task(task_description)
        summary = framework.get_execution_summary()
        
        print("\n" + "="*60)
        print("执行结果:")
        print(f"任务: {summary['task_description']}")
        print(f"总步数: {summary['total_steps']}")
        print(f"成功: {'✅ 是' if summary['success'] else '❌ 否'}")
        print(f"数据保存位置: {summary.get('task_directory', 'N/A')}")
        print("="*60)
        
        if summary['action_history']:
            print("\n最近几步操作:")
            for i, action in enumerate(summary['action_history'][-3:], len(summary['action_history'])-2):
                print(f"  步骤{i}: {action['thought'][:50]}...")
        
    except Exception as e:
        print(f"❌ 任务执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_task()
