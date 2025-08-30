#!/usr/bin/env python3
"""
通用任务测试执行入口
支持通过 task.json 配置文件灵活配置不同任务的测试

使用方法：
- python universal_test_runner.py task_configs/taobao.json              # 测试所有类型
- python universal_test_runner.py task_configs/taobao.json type3        # 测试指定类型
- python universal_test_runner.py task_configs/taobao.json type3:150    # 测试指定类型的指定trace
- python universal_test_runner.py task_configs/taobao.json 150,151,152  # 测试指定的trace编号
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Union, Any, Optional
from dataclasses import dataclass, asdict

try:
    import llmconfig
except ImportError:
    print("警告: 无法导入 llmconfig，将使用默认配置")
    class MockLLMConfig:
        API_KEY = "your key"
        BASE_URL = "your url"  
        MODEL = "gemini-2.5-pro-preview-06-05"
    llmconfig = MockLLMConfig()

from avdag.verifier import verify_task_folder, VerifierOptions, make_llm_options
from avdag.ocr_processor import create_standard_ocr_functions
from avdag.logger import configure_logging

@dataclass
class TestResult:
    """测试结果数据结构"""
    trace_id: Union[str, int]
    task_type: str
    task_name: str
    success: bool
    score: float
    matched_nodes: List[str]
    reason: str
    manual_review_needed: bool
    execution_time: float
    error_message: str = ""

@dataclass 
class TestSummary:
    """测试汇总数据结构"""
    task_name: str
    total_tests: int
    success_count: int
    success_rate: float
    total_execution_time: float
    results_by_type: Dict[str, Dict[str, Any]]
    all_results: List[TestResult]

class UniversalTestRunner:
    """通用测试执行器"""
    
    def __init__(self, config_file: str):
        """初始化测试执行器
        
        Args:
            config_file: 任务配置文件路径
        """
        self.config_file = config_file
        self.config = self._load_config()
        # 使用当前工作目录作为基础目录，而不是配置文件所在目录
        self.base_dir = os.getcwd()
        self.start_time = datetime.now()
        
        # 生成带时间戳的文件名
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        task_name = self.config['task_name']
        
        # 配置日志
        log_config = self.config.get('logging', {})
        log_file = log_config.get('output_file', 'test-{task_name}-{timestamp}.log')
        log_file = log_file.format(task_name=task_name, timestamp=timestamp)
        
        configure_logging(
            level=log_config.get('level', 'DEBUG'),
            use_colors=log_config.get('use_colors', True),
            show_time=log_config.get('show_time', True),
            show_module=log_config.get('show_module', True),
            output_file=log_file
        )
        
        # 配置输出文件
        output_config = self.config.get('output', {})
        self.summary_file = output_config.get('summary_file', 'test-{task_name}-summary-{timestamp}.txt')
        self.summary_file = self.summary_file.format(task_name=task_name, timestamp=timestamp)
        
        self.detailed_file = output_config.get('detailed_results_file', 'test-{task_name}-detailed-{timestamp}.json')
        self.detailed_file = self.detailed_file.format(task_name=task_name, timestamp=timestamp)
        
        # 创建验证选项
        self.opts = self._create_verifier_options()
        
        print(f"=== 通用任务测试执行器 ===")
        print(f"任务名称: {self.config['task_name']}")
        print(f"任务描述: {self.config.get('description', 'N/A')}")
        print(f"日志文件: {log_file}")
        print(f"汇总文件: {self.summary_file}")
        print(f"详细结果: {self.detailed_file}")
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise RuntimeError(f"无法加载配置文件 {self.config_file}: {str(e)}")
    
    def _create_verifier_options(self) -> VerifierOptions:
        """创建验证选项"""
        test_opts = self.config.get('test_options', {})
        
        # 获取LLM配置
        api_key = getattr(llmconfig, 'API_KEY', None)
        base_url = getattr(llmconfig, 'BASE_URL', None) 
        model = getattr(llmconfig, 'MODEL', None)
        
        print("=== LLM 配置 ===")
        print(f"API_KEY: {api_key}")
        print(f"BASE_URL: {base_url}")
        print(f"MODEL: {model}")
        
        if test_opts.get('enable_ocr', True):
            # 创建标准OCR函数
            ocr_func, texts_func = create_standard_ocr_functions()
        else:
            ocr_func = None
        
        if test_opts.get('enable_llm', True) and api_key and base_url:
            # 创建带LLM的选项
            opts = make_llm_options(
                api_key=api_key, 
                base_url=base_url, 
                model=model, 
                force_llm=test_opts.get('force_llm', False)
            )
            if ocr_func:
                opts.ocr = ocr_func
            print("[验证] 已启用OCR+LLM验证模式")
        else:
            # 仅OCR模式
            opts = VerifierOptions(ocr=ocr_func)
            print("[验证] 已启用纯OCR验证模式")
            if not api_key or not base_url:
                print("[警告] LLM_API_KEY/LLM_BASE_URL 未设置，已退化到 OCR-only 验证")
        
        # 应用其他选项
        if test_opts.get('ocr_frame_exclusive'):
            opts.ocr_frame_exclusive = True
        if test_opts.get('llm_frame_exclusive'):
            opts.llm_frame_exclusive = True  
        if test_opts.get('prevent_frame_backtrack'):
            opts.prevent_frame_backtrack = True
            
        return opts
    
    def _get_rule_file_path(self, task_type: str) -> str:
        """获取规则文件完整路径"""
        rules_base = self.config['rules_base_dir']
        rule_file = self.config['task_types'][task_type]['rule_file']
        return os.path.join(self.base_dir, rules_base, rule_file)
    
    def _get_data_path(self, task_type: str, trace_id: Union[str, int]) -> str:
        """获取数据路径"""
        data_base = self.config['data_base_dir']
        
        # 如果trace_id包含路径分隔符，说明是层级结构（如 "type2/11"）
        if isinstance(trace_id, str) and '/' in trace_id:
            return os.path.join(self.base_dir, data_base, trace_id)
        # 如果trace_id是字符串(如"type2")，直接使用作为文件夹名
        elif isinstance(trace_id, str):
            return os.path.join(self.base_dir, data_base, trace_id)
        else:
            # 如果是数字，需要在task_type目录下查找
            return os.path.join(self.base_dir, data_base, task_type, str(trace_id))
    
    def _auto_discover_traces(self, task_type: str) -> List[Union[str, int]]:
        """自动发现指定类型的所有trace"""
        data_base = self.config['data_base_dir']
        base_path = os.path.join(self.base_dir, data_base)
        
        discovered_traces = []
        
        # 首先尝试查找 type* 目录格式 (如 type2, type3)
        type_dir = os.path.join(base_path, task_type)
        if os.path.exists(type_dir) and os.path.isdir(type_dir):
            print(f"[自动发现] 找到 type 目录: {type_dir}")
            
            # 扫描 type 目录下的子目录
            sub_traces = []
            for item in os.listdir(type_dir):
                item_path = os.path.join(type_dir, item)
                if os.path.isdir(item_path):
                    try:
                        # 尝试转换为数字
                        trace_num = int(item)
                        sub_traces.append(f"{task_type}/{trace_num}")
                    except ValueError:
                        # 如果不是数字，也包含字符串格式的目录
                        sub_traces.append(f"{task_type}/{item}")
            
            if sub_traces:
                # 对数字进行排序
                numeric_traces = [t for t in sub_traces if t.split('/')[-1].isdigit()]
                string_traces = [t for t in sub_traces if not t.split('/')[-1].isdigit()]
                
                # 按最后一部分（trace编号）排序
                numeric_traces.sort(key=lambda x: int(x.split('/')[-1]))
                string_traces.sort()
                
                discovered_traces = numeric_traces + string_traces
                print(f"[自动发现] 在 {task_type} 目录下发现 {len(discovered_traces)} 个子目录: {[t.split('/')[-1] for t in discovered_traces]}")
                return discovered_traces
            else:
                # 如果 type 目录下没有子目录，则将 type 目录本身作为 trace
                discovered_traces.append(task_type)
                print(f"[自动发现] {task_type} 目录下无子目录，使用目录本身作为 trace")
                return discovered_traces
        
        # 然后尝试查找数字目录格式 (如 150, 151, 152...)
        if os.path.exists(base_path):
            for item in os.listdir(base_path):
                item_path = os.path.join(base_path, item)
                if os.path.isdir(item_path):
                    try:
                        # 尝试转换为数字
                        trace_num = int(item)
                        discovered_traces.append(trace_num)
                    except ValueError:
                        # 如果不是数字，也包含字符串格式的目录
                        if item.startswith('type') or item == task_type:
                            discovered_traces.append(item)
        
        # 排序结果
        numeric_traces = [t for t in discovered_traces if isinstance(t, int)]
        string_traces = [t for t in discovered_traces if isinstance(t, str)]
        numeric_traces.sort()
        string_traces.sort()
        
        final_traces = string_traces + numeric_traces
        
        if final_traces:
            print(f"[自动发现] 类型 {task_type} 发现 {len(final_traces)} 个 traces: {final_traces}")
        else:
            print(f"[自动发现] 类型 {task_type} 未发现任何可用的 traces")
        
        return final_traces

    def _get_traces_for_type(self, task_type: str) -> List[Union[str, int]]:
        """获取指定类型的所有trace"""
        task_config = self.config['task_types'][task_type]
        data_traces = task_config.get('data_traces')
        
        # 如果没有配置 data_traces 或配置为空，则自动发现
        if not data_traces:
            print(f"[配置] 类型 {task_type} 未配置 data_traces，启用自动发现模式")
            return self._auto_discover_traces(task_type)
        
        # 如果配置了 data_traces，则优先使用配置的值
        print(f"[配置] 类型 {task_type} 使用配置的 data_traces: {data_traces}")
        
        if isinstance(data_traces, str):
            # 如果是字符串，检查对应目录是否存在
            data_path = self._get_data_path(task_type, data_traces)
            if os.path.exists(data_path):
                # 检查是否是目录，如果是目录且有子目录，则扫描子目录
                if os.path.isdir(data_path):
                    sub_traces = []
                    for item in os.listdir(data_path):
                        item_path = os.path.join(data_path, item)
                        if os.path.isdir(item_path):
                            try:
                                # 尝试转换为数字
                                trace_num = int(item)
                                sub_traces.append(f"{data_traces}/{trace_num}")
                            except ValueError:
                                # 如果不是数字，也包含字符串格式的目录
                                sub_traces.append(f"{data_traces}/{item}")
                    
                    if sub_traces:
                        # 对数字进行排序
                        numeric_traces = [t for t in sub_traces if t.split('/')[-1].isdigit()]
                        string_traces = [t for t in sub_traces if not t.split('/')[-1].isdigit()]
                        
                        # 按最后一部分（trace编号）排序
                        numeric_traces.sort(key=lambda x: int(x.split('/')[-1]))
                        string_traces.sort()
                        
                        discovered_traces = numeric_traces + string_traces
                        print(f"[配置] 类型 {task_type} 在目录 {data_traces} 下发现 {len(discovered_traces)} 个子目录: {[t.split('/')[-1] for t in discovered_traces]}")
                        return discovered_traces
                    else:
                        # 如果没有子目录，则使用目录本身
                        return [data_traces]
                else:
                    return [data_traces]
            else:
                print(f"警告: 配置的数据路径不存在: {data_path}，尝试自动发现")
                return self._auto_discover_traces(task_type)
        elif isinstance(data_traces, list):
            # 如果是列表，返回所有存在的trace
            valid_traces = []
            for trace in data_traces:
                data_path = self._get_data_path(task_type, trace)
                if os.path.exists(data_path):
                    valid_traces.append(trace)
                else:
                    print(f"警告: trace {trace} 数据路径不存在: {data_path}")
            
            # 如果配置的traces都不存在，则尝试自动发现
            if not valid_traces:
                print(f"警告: 配置的所有 traces 都不存在，尝试自动发现")
                return self._auto_discover_traces(task_type)
            
            return valid_traces
        else:
            print(f"警告: 不支持的数据traces格式: {type(data_traces)}，尝试自动发现")
            return self._auto_discover_traces(task_type)
    
    def test_single_trace(self, task_type: str, trace_id: Union[str, int]) -> TestResult:
        """测试单个trace"""
        task_config = self.config['task_types'][task_type]
        task_name = task_config['name']
        
        start_time = time.time()
        
        try:
            # 获取文件路径
            rule_file = self._get_rule_file_path(task_type)
            data_path = self._get_data_path(task_type, trace_id)
            
            # 检查文件是否存在
            if not os.path.exists(rule_file):
                error_msg = f"规则文件不存在: {rule_file}"
                print(f"❌ {trace_id}: {error_msg}")
                return TestResult(
                    trace_id=trace_id, task_type=task_type, task_name=task_name,
                    success=False, score=0.0, matched_nodes=[], reason=error_msg,
                    manual_review_needed=False, execution_time=time.time() - start_time,
                    error_message=error_msg
                )
            
            if not os.path.exists(data_path):
                error_msg = f"数据路径不存在: {data_path}" 
                print(f"❌ {trace_id}: {error_msg}")
                return TestResult(
                    trace_id=trace_id, task_type=task_type, task_name=task_name,
                    success=False, score=0.0, matched_nodes=[], reason=error_msg,
                    manual_review_needed=False, execution_time=time.time() - start_time,
                    error_message=error_msg
                )
            
            print(f"\n--- 测试 {trace_id} [{task_name}] ---")
            print(f"规则文件: {os.path.basename(rule_file)}")
            print(f"数据路径: {data_path}")
            
            # 执行验证
            res = verify_task_folder(rule_file, data_path, self.opts)
            
            # 构建结果
            matched_nodes = [m.node_id for m in res.matched] if res.matched else []
            execution_time = time.time() - start_time
            
            result = TestResult(
                trace_id=trace_id,
                task_type=task_type, 
                task_name=task_name,
                success=res.ok,
                score=res.total_score,
                matched_nodes=matched_nodes,
                reason=res.reason or "",
                manual_review_needed=res.manual_review_needed,
                execution_time=execution_time
            )
            
            # 输出结果
            status = "✓ 成功" if res.ok else "✗ 失败"
            print(f"验证结果: {status}")
            print(f"匹配节点: {matched_nodes}")
            print(f"任务得分: {res.total_score}分")
            print(f"执行时间: {execution_time:.2f}秒")
            
            if res.reason:
                print(f"详细原因: {res.reason}")
            if res.manual_review_needed:
                print("⚠️  需要人工复核")
            
            return result
            
        except Exception as e:
            error_msg = f"测试执行异常: {str(e)}"
            print(f"❌ {trace_id}: {error_msg}")
            
            return TestResult(
                trace_id=trace_id, task_type=task_type, task_name=task_name,
                success=False, score=0.0, matched_nodes=[], reason="",
                manual_review_needed=False, execution_time=time.time() - start_time,
                error_message=error_msg
            )
    
    def test_by_type(self, task_type: str, specific_trace: Optional[Union[str, int]] = None) -> List[TestResult]:
        """按类型测试"""
        if task_type not in self.config['task_types']:
            print(f"错误: 任务类型 '{task_type}' 未在配置中定义")
            return []
        
        task_config = self.config['task_types'][task_type] 
        task_name = task_config['name']
        
        if specific_trace is not None:
            # 测试指定的trace
            traces = [specific_trace]
        else:
            # 获取该类型的所有trace
            traces = self._get_traces_for_type(task_type)
        
        if not traces:
            print(f"错误: 类型 '{task_type}' 没有可用的trace数据")
            return []
        
        print(f"\n{'='*60}")
        print(f"测试任务类型 {task_type} - {task_name}")
        print(f"trace数量: {len(traces)}")
        print(f"{'='*60}")
        
        results = []
        for trace_id in traces:
            result = self.test_single_trace(task_type, trace_id)
            results.append(result)
        
        return results
    
    def test_all_types(self) -> TestSummary:
        """测试所有类型"""
        print(f"\n{'='*80}")
        print(f"开始测试任务: {self.config['task_name']}")
        print(f"任务描述: {self.config.get('description', 'N/A')}")
        print(f"{'='*80}")
        
        all_results = []
        results_by_type = {}
        
        for task_type in self.config['task_types']:
            type_results = self.test_by_type(task_type)
            all_results.extend(type_results)
            
            # 汇总该类型的结果
            success_count = sum(1 for r in type_results if r.success)
            total_count = len(type_results)
            success_rate = (success_count / total_count * 100) if total_count > 0 else 0
            
            results_by_type[task_type] = {
                'task_name': self.config['task_types'][task_type]['name'],
                'total_tests': total_count,
                'success_count': success_count,
                'success_rate': success_rate,
                'results': type_results
            }
            
            print(f"\n--- 类型 {task_type} 汇总 ---")
            for result in type_results:
                status = "✓" if result.success else "✗"
                print(f"trace {result.trace_id}: {status} | score: {result.score} | nodes: {result.matched_nodes} | reason: {result.reason}")
            print(f"成功率: {success_count}/{total_count} ({success_rate:.1f}%)")
        
        # 生成总汇总
        total_tests = len(all_results)
        total_success = sum(1 for r in all_results if r.success)
        total_success_rate = (total_success / total_tests * 100) if total_tests > 0 else 0
        total_execution_time = time.time() - self.start_time.timestamp()
        
        summary = TestSummary(
            task_name=self.config['task_name'],
            total_tests=total_tests,
            success_count=total_success,
            success_rate=total_success_rate,
            total_execution_time=total_execution_time,
            results_by_type=results_by_type,
            all_results=all_results
        )
        
        return summary
    
    def save_results(self, summary: TestSummary):
        """保存测试结果"""
        # 保存详细结果(JSON格式)
        detailed_data = {
            'summary': asdict(summary),
            'timestamp': self.start_time.isoformat(),
            'config_file': self.config_file,
            'config': self.config
        }
        
        with open(self.detailed_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_data, f, ensure_ascii=False, indent=2, default=str)
        
        # 保存汇总结果(文本格式)
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            f.write(f"任务测试汇总报告\n")
            f.write(f"{'='*60}\n")
            f.write(f"任务名称: {summary.task_name}\n")
            f.write(f"测试时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"配置文件: {self.config_file}\n")
            f.write(f"总测试数: {summary.total_tests}\n")
            f.write(f"总成功数: {summary.success_count}\n")
            f.write(f"总成功率: {summary.success_rate:.1f}%\n")
            f.write(f"总执行时间: {summary.total_execution_time:.2f}秒\n")
            f.write(f"\n")
            
            f.write(f"分类型结果:\n")
            f.write(f"{'-'*40}\n")
            for task_type, type_data in summary.results_by_type.items():
                f.write(f"类型 {task_type} ({type_data['task_name']}):\n")
                f.write(f"  测试数: {type_data['total_tests']}\n")
                f.write(f"  成功数: {type_data['success_count']}\n") 
                f.write(f"  成功率: {type_data['success_rate']:.1f}%\n")
                f.write(f"\n")
            
            f.write(f"详细结果:\n")
            f.write(f"{'-'*40}\n")
            for result in summary.all_results:
                status = "✓" if result.success else "✗"
                f.write(f"trace {result.trace_id} [{result.task_type}]: {status}\n")
                f.write(f"  得分: {result.score}\n")
                f.write(f"  匹配节点: {result.matched_nodes}\n")
                f.write(f"  原因: {result.reason}\n")
                f.write(f"  执行时间: {result.execution_time:.2f}秒\n")
                if result.error_message:
                    f.write(f"  错误: {result.error_message}\n")
                f.write(f"\n")
        
        print(f"\n=== 结果已保存 ===")
        print(f"汇总文件: {self.summary_file}")
        print(f"详细文件: {self.detailed_file}")
    
    def print_final_summary(self, summary: TestSummary):
        """打印最终汇总"""
        print(f"\n{'='*80}")
        print(f"任务测试完成: {summary.task_name}")
        print(f"{'='*80}")
        print(f"总测试数: {summary.total_tests}")
        print(f"总成功数: {summary.success_count}")
        print(f"总成功率: {summary.success_rate:.1f}%")
        print(f"总执行时间: {summary.total_execution_time:.2f}秒")
        print(f"")
        
        print("分类型结果:")
        for task_type, type_data in summary.results_by_type.items():
            print(f"  类型 {task_type} ({type_data['task_name']}): {type_data['success_count']}/{type_data['total_tests']} ({type_data['success_rate']:.1f}%)")
        
        print(f"\n详细结果文件: {self.detailed_file}")
        print(f"汇总结果文件: {self.summary_file}")

def show_usage(config_dir: str = "task_configs"):
    """显示使用说明"""
    print(f"""
通用任务测试执行器使用说明:

1. 测试所有类型:
   python universal_test_runner.py {config_dir}/taobao.json

2. 测试指定类型:
   python universal_test_runner.py {config_dir}/taobao.json type3
   python universal_test_runner.py {config_dir}/xiaohongshu.json type2

3. 测试指定类型的指定trace:
   python universal_test_runner.py {config_dir}/taobao.json type3:150

4. 测试指定trace编号:
   python universal_test_runner.py {config_dir}/taobao.json 150,151,152

可用的配置文件:""")
    
    # 查找可用的配置文件
    if os.path.exists(config_dir):
        for file in os.listdir(config_dir):
            if file.endswith('.json'):
                print(f"  - {config_dir}/{file}")
    
    print(f"""
配置文件说明:
- 每个配置文件定义一个任务的测试参数
- 包含规则文件目录、数据目录、任务类型映射等
- 可通过修改配置文件来调整测试范围和参数
""")

def main():
    """主函数"""
    if len(sys.argv) < 2:
        show_usage()
        return
    
    config_file = sys.argv[1]
    
    if not os.path.exists(config_file):
        print(f"错误: 配置文件不存在: {config_file}")
        show_usage()
        return
    
    # 创建测试运行器
    runner = UniversalTestRunner(config_file)
    
    if len(sys.argv) == 2:
        # 测试所有类型
        summary = runner.test_all_types()
    else:
        # 解析参数
        arg = sys.argv[2]
        
        if ':' in arg:
            # 格式: type3:150 或 3:150
            task_type, trace_id = arg.split(':', 1)
            # 只有当任务类型不在配置中时，才尝试去掉 type 前缀
            if task_type not in runner.config['task_types'] and task_type.startswith('type'):
                numeric_type = task_type.replace('type', '')
                if numeric_type in runner.config['task_types']:
                    task_type = numeric_type
            
            try:
                trace_id = int(trace_id)
            except ValueError:
                pass  # 保持字符串格式
            results = runner.test_by_type(task_type, trace_id)
            
            # 创建简单汇总
            total_success = sum(1 for r in results if r.success)
            summary = TestSummary(
                task_name=runner.config['task_name'],
                total_tests=len(results),
                success_count=total_success,
                success_rate=(total_success / len(results) * 100) if results else 0,
                total_execution_time=time.time() - runner.start_time.timestamp(),
                results_by_type={task_type: {
                    'task_name': runner.config['task_types'].get(task_type, {}).get('name', f'任务{task_type}'),
                    'total_tests': len(results),
                    'success_count': total_success,
                    'success_rate': (total_success / len(results) * 100) if results else 0,
                    'results': results
                }},
                all_results=results
            )
            
        elif arg.startswith('type') or arg.isdigit():
            # 测试指定类型，支持 type3 或 3 的格式
            task_type = arg
            # 只有当任务类型不在配置中时，才尝试去掉 type 前缀
            if task_type not in runner.config['task_types'] and task_type.startswith('type'):
                numeric_type = task_type.replace('type', '')
                if numeric_type in runner.config['task_types']:
                    task_type = numeric_type
            
            results = runner.test_by_type(task_type)
            
            # 创建简单汇总
            total_success = sum(1 for r in results if r.success)
            summary = TestSummary(
                task_name=runner.config['task_name'],
                total_tests=len(results),
                success_count=total_success,
                success_rate=(total_success / len(results) * 100) if results else 0,
                total_execution_time=time.time() - runner.start_time.timestamp(),
                results_by_type={task_type: {
                    'task_name': runner.config['task_types'].get(task_type, {}).get('name', f'任务{task_type}'),
                    'total_tests': len(results),
                    'success_count': total_success,
                    'success_rate': (total_success / len(results) * 100) if results else 0,
                    'results': results
                }},
                all_results=results
            )
            
        else:
            # 按trace编号测试
            try:
                trace_nums = [int(x.strip()) for x in arg.split(",")]
                all_results = []
                
                # 找到每个trace对应的类型
                for trace_num in trace_nums:
                    found = False
                    for task_type, task_config in runner.config['task_types'].items():
                        data_traces = task_config['data_traces']
                        if isinstance(data_traces, list) and trace_num in data_traces:
                            result = runner.test_single_trace(task_type, trace_num)
                            all_results.append(result)
                            found = True
                            break
                    
                    if not found:
                        print(f"警告: trace编号{trace_num}未在配置中找到")
                
                # 创建汇总
                total_success = sum(1 for r in all_results if r.success)
                summary = TestSummary(
                    task_name=runner.config['task_name'],
                    total_tests=len(all_results),
                    success_count=total_success,
                    success_rate=(total_success / len(all_results) * 100) if all_results else 0,
                    total_execution_time=time.time() - runner.start_time.timestamp(),
                    results_by_type={},
                    all_results=all_results
                )
                
            except ValueError:
                print(f"错误: 无效的trace编号格式: {arg}")
                return
    
    # 保存和打印结果
    runner.save_results(summary)
    runner.print_final_summary(summary)

if __name__ == "__main__":
    main()
