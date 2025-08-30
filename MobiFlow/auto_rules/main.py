#!/usr/bin/env python3
"""
Auto Rules 主程序
基于 LLM 的任务配置自动生成系统

使用方法:
    python main.py [目标目录] [API密钥] [选项]

示例:
    # 分析目录并生成配置
    python main.py /path/to/taobao_test YOUR_API_KEY --output-file config.yaml
    
    # 仅提取任务描述不调用LLM
    python main.py /path/to/taobao_test dummy --dry-run
"""

import argparse
import logging
import sys
import json
import yaml
from pathlib import Path

from task_extractor import TaskDescriptionExtractor
from llm_generator import LLMConfigGenerator
from prompts import generate_user_prompt

# 添加上级目录到路径以导入llmconfig
sys.path.append(str(Path(__file__).parent.parent))
import llmconfig


def format_yaml_with_brackets(config: dict) -> str:
    """
    格式化YAML输出，确保特定数组字段使用方括号格式
    
    Args:
        config: 配置字典
        
    Returns:
        格式化后的YAML字符串
    """
    import re
    
    # 首先使用标准YAML格式化
    yaml_str = yaml.dump(config, default_flow_style=False, 
                        allow_unicode=True, sort_keys=False, indent=2)
    
    # 逐行处理，修复数组格式
    lines = yaml_str.split('\n')
    result_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # 检查是否是需要格式化的数组字段
        if re.match(r'(\s+)(deps|next|any_of):\s*$', line):
            # 处理单项数组
            indent = re.match(r'(\s+)', line).group(1) if re.match(r'(\s+)', line) else ''
            field_name = re.search(r'(deps|next|any_of):', line).group(1)
            
            # 查找下一行的数组项
            if i + 1 < len(lines) and re.match(r'\s*-\s*(.+)', lines[i + 1]):
                item = re.match(r'\s*-\s*(.+)', lines[i + 1]).group(1)
                result_lines.append(f'{indent}{field_name}: [{item}]')
                i += 2  # 跳过数组项行
                continue
        
        elif re.match(r'(\s+)(all|any):\s*$', line):
            # 处理多项数组
            indent = re.match(r'(\s+)', line).group(1) if re.match(r'(\s+)', line) else ''
            field_name = re.search(r'(all|any):', line).group(1)
            
            # 收集所有数组项
            items = []
            j = i + 1
            while j < len(lines) and re.match(r'\s*-\s*(.+)', lines[j]):
                item = re.match(r'\s*-\s*(.+)', lines[j]).group(1)
                # 确保引号正确
                if not (item.startswith('"') and item.endswith('"')):
                    item = f'"{item}"'
                items.append(item)
                j += 1
            
            if items:
                formatted_items = ', '.join(items)
                result_lines.append(f'{indent}{field_name}: [{formatted_items}]')
                i = j  # 跳过所有数组项行
                continue
        
        # 普通行直接添加
        result_lines.append(line)
        i += 1
    
    return '\n'.join(result_lines)


def load_template_yaml(template_type: str) -> str:
    """加载模版 YAML 文件内容"""
    base_dir = Path(__file__).parent.parent / "task_rules"
    
    if template_type == "orders":
        template_file = base_dir / "example_checker_ordes.yaml"
    elif template_type == "modes":
        template_file = base_dir / "example_checker_modes.yaml"
    else:
        raise ValueError(f"不支持的模版类型: {template_type}")
    
    if not template_file.exists():
        raise FileNotFoundError(f"模版文件不存在: {template_file}")
    
    with open(template_file, 'r', encoding='utf-8') as f:
        return f.read()


def generate_llm_prompt(task_descriptions: list, template_yaml: str, app_name: str = "unknown") -> str:
    """生成发送给 LLM 的提示词（使用共享的提示词模块）"""
    return generate_user_prompt(task_descriptions, template_yaml, app_name)


def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(
        description="基于 LLM 的任务配置自动生成系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # 必需参数
    parser.add_argument("target_dir", 
                       help="包含actions.json文件的目标目录路径")
    
    # parser.add_argument("api_key",
    #                    help="OpenAI API密钥")
    
    # 可选参数
    parser.add_argument("--output-file", "-o",
                       help="输出配置文件路径（默认：基于目录名生成）")
    
    parser.add_argument("--template-type", "-t",
                       choices=["orders", "modes"],
                       default="orders",
                       help="使用的模版类型 (默认: orders)")
    
    parser.add_argument("--app-name",
                       help="应用名称（默认从任务描述中提取）")
    
    parser.add_argument("--dry-run",
                       action="store_true",
                       help="仅提取和显示任务描述，不调用LLM")
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    
    try:
        # 验证参数
        target_dir = Path(args.target_dir)
        if not target_dir.exists():
            logger.error(f"目标目录不存在: {args.target_dir}")
            sys.exit(1)
        
        # 步骤1: 提取任务描述
        logger.info(f"从 {args.target_dir} 提取任务描述...")
        extractor = TaskDescriptionExtractor()
        task_data = extractor.extract_from_directory(args.target_dir)
        
        if not task_data:
            logger.warning("未找到任何有效的actions.json文件")
            sys.exit(0)
        
        # 提取task_description列表
        task_descriptions = [task['task_description'] for task in task_data if task['task_description']]
        
        if not task_descriptions:
            logger.warning("未找到任何有效的task_description")
            sys.exit(0)
        
        logger.info(f"成功提取 {len(task_descriptions)} 个任务描述")
        
        # 显示提取的任务描述
        print(f"\n提取到的任务描述:")
        for i, desc in enumerate(task_descriptions, 1):
            print(f"{i}. {desc}")
        
        # 确定应用名称
        app_name = args.app_name
        if not app_name:
            app_names = set(task.get('app_name', '') for task in task_data if task.get('app_name'))
            if app_names:
                app_name = list(app_names)[0]
            else:
                app_name = target_dir.name
        
        logger.info(f"使用应用名称: {app_name}")
        
        if args.dry_run:
            logger.info("试运行模式，不调用LLM生成配置")
            return
        
        # 步骤2: 加载模版
        logger.info(f"加载 {args.template_type} 模版...")
        template_yaml = load_template_yaml(args.template_type)
        
        # 步骤3: 生成LLM提示词
        logger.info("生成LLM提示词...")
        prompt = generate_llm_prompt(task_descriptions, template_yaml, app_name)
        
        # 步骤4: 调用LLM生成配置
        logger.info("调用LLM生成配置...")

        api_key = llmconfig.API_KEY
        base_url = llmconfig.BASE_URL
        model   =   llmconfig.MODEL
        llm_generator = LLMConfigGenerator(api_key, base_url,model)
        
        generated_config = llm_generator.generate_config_from_prompt(prompt)
        
        if not generated_config:
            logger.error("LLM生成配置失败")
            sys.exit(1)
        
        # 步骤5: 保存配置文件
        if args.output_file:
            output_path = Path(args.output_file)
        else:
            # 基于目录名生成输出文件名
            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in target_dir.name)
            output_path = Path(f"{safe_name}_{args.template_type}_config.yaml")
        
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"保存配置到: {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            if isinstance(generated_config, dict):
                # 使用自定义格式化保存YAML，确保数组使用方括号格式
                yaml_content = format_yaml_with_brackets(generated_config)
                f.write(yaml_content)
            else:
                f.write(generated_config)
        
        logger.info("任务配置生成完成!")
        print(f"\n✓ 配置文件已生成: {output_path}")
        
    except KeyboardInterrupt:
        logger.info("用户中断程序")
        sys.exit(0)
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        sys.exit(1)


def print_analysis_summary(analysis, group_name):
    """打印分析结果摘要"""
    print(f"\n=== {group_name} 分析结果 ===")
    print(f"总任务数: {analysis['total_tasks']}")
    print(f"应用名称: {', '.join(analysis['app_names'])}")
    
    if analysis['common_actions']:
        print("常见动作:")
        for action in analysis['common_actions'][:5]:  # 显示前5个
            print(f"  - {action['action']}: {action['count']}次 ({action['frequency']:.1%})")
    
    if analysis['task_patterns']:
        print("识别的任务模式:")
        for pattern in analysis['task_patterns']:
            print(f"  - {pattern['name']}: 置信度 {pattern['confidence']:.1%}")
    
    complexity = analysis.get('complexity_analysis', {})
    if complexity:
        level = complexity.get('complexity_level', 'unknown')
        avg_actions = complexity.get('avg_actions', 0)
        print(f"复杂度: {level} (平均 {avg_actions:.1f} 步操作)")
    
    print()


if __name__ == "__main__":
    main()
