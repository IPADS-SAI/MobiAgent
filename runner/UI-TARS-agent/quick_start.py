#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UI-TARS 自动化框架快速启动脚本
"""

from ui_tars_automation import UITarsAutomationFramework, ExecutionConfig

def quick_start():
    """快速启动示例 - 淘宝购物任务"""
    print("UI-TARS 快速启动 - 淘宝购物任务")
    print("=" * 50)
    
    # 获取模型服务地址
    model_url = input("请输入模型服务地址 (默认: http://192.168.12.152:8000/v1): ").strip()
    if not model_url:
        model_url = "http://192.168.12.152:8000/v1"
    
    # 配置
    config = ExecutionConfig(
        model_base_url=model_url,
        model_name="UI-TARS-7B-SFT",
        max_steps=25,
        step_delay=2.0,
        language="Chinese"
    )
    
    # 创建框架实例
    try:
        framework = UITarsAutomationFramework(config)
        print("✅ 框架初始化成功!")
    except Exception as e:
        print(f"❌ 框架初始化失败: {e}")
        return
    
    # 执行淘宝购物任务
    task_description = "饿了么中帮我点蜜雪冰城的芋圆葡萄，要少冰，5分糖，加珍珠"
    
    print(f"\n开始执行任务: {task_description}")
    print("-" * 50)
    
    try:
        success = framework.execute_task(task_description)
        summary = framework.get_execution_summary()
        
        print("\n" + "="*60)
        print("执行结果:")
        print(f"任务: {summary['task_description']}")
        print(f"总步数: {summary['total_steps']}")
        print(f"成功: {'✅ 是' if summary['success'] else '❌ 否'}")
        print("="*60)
        
        if summary['action_history']:
            print("\n最近几步操作:")
            for i, action in enumerate(summary['action_history'][-3:], len(summary['action_history'])-2):
                print(f"  步骤{i}: {action['thought'][:50]}...")
        
    except Exception as e:
        print(f"❌ 任务执行失败: {e}")

if __name__ == "__main__":
    quick_start()
