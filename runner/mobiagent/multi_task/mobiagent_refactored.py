"""
MobiAgent多阶段任务执行系统 - 主入口

本模块提供统一的任务执行入口,自动分析任务类型并选择合适的执行策略
"""

import json
import logging
import time
import os
import argparse
import tempfile
from typing import Optional
from abc import ABC, abstractmethod
import uiautomator2 as u2
import base64
from openai import OpenAI

# 导入自定义模块
try:
    # 尝试相对导入（当作为模块运行时）
    from .models import Plan, Artifact, State, Subtask, ExtractArtifactConfig
    from .executor import task_in_app
    from .planner import (
        analyze_task,
        generate_plan,
        extract_artifact,
        refine_next_subtask_description,
        get_app_package_name
    )
except ImportError:
    # 回退到绝对导入（当直接运行时）
    from models import Plan, Artifact, State, Subtask, ExtractArtifactConfig
    from executor import task_in_app
    from planner import (
        analyze_task,
        generate_plan,
        extract_artifact,
        refine_next_subtask_description,
        get_app_package_name
    )

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ==================== 设备抽象层 ====================

class Device(ABC):
    """设备操作抽象类"""
    
    @abstractmethod
    def start_app(self, app):
        pass

    @abstractmethod
    def screenshot(self, path):
        pass

    @abstractmethod
    def click(self, x, y):
        pass

    @abstractmethod
    def input(self, text):
        pass

    @abstractmethod
    def swipe(self, direction):
        pass

    @abstractmethod
    def keyevent(self, key):
        pass

    @abstractmethod
    def dump_hierarchy(self):
        pass


class AndroidDevice(Device):
    """Android设备实现"""
    
    def __init__(self, adb_endpoint=None):
        super().__init__()
        if adb_endpoint:
            self.d = u2.connect(adb_endpoint)
        else:
            self.d = u2.connect()
        
        self.app_package_names = {
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
            "bilibili": "tv.danmaku.bili",
            "腾讯视频": "com.tencent.qqlive"
        }

    def start_app(self, app):
        package_name = self.app_package_names.get(app)
        if not package_name:
            raise ValueError(f"App '{app}' is not registered with a package name.")
        self.d.app_start(package_name, stop=True)
        time.sleep(0.5)
        if not self.d.app_wait(package_name, timeout=10):
            raise RuntimeError(f"Failed to start app '{app}' with package '{package_name}'")
    
    def app_start(self, package_name):
        self.d.app_start(package_name, stop=True)
        time.sleep(0.5)
        if not self.d.app_wait(package_name, timeout=10):
            raise RuntimeError(f"Failed to start package '{package_name}'")
        
    def screenshot(self, path):
        self.d.screenshot(path)

    def click(self, x, y):
        self.d.click(x, y)
        time.sleep(0.5)

    def input(self, text):
        current_ime = self.d.current_ime()
        self.d.shell(['settings', 'put', 'secure', 'default_input_method', 'com.android.adbkeyboard/.AdbIME'])
        time.sleep(0.5)
        charsb64 = base64.b64encode(text.encode('utf-8')).decode('utf-8')
        self.d.shell(['am', 'broadcast', '-a', 'ADB_INPUT_B64', '--es', 'msg', charsb64])
        time.sleep(0.5)
        self.d.shell(['settings', 'put', 'secure', 'default_input_method', current_ime])
        time.sleep(0.5)

    def swipe(self, direction, scale=0.5):
        self.d.swipe_ext(direction=direction, scale=scale)

    def keyevent(self, key):
        self.d.keyevent(key)

    def dump_hierarchy(self):
        return self.d.dump_hierarchy()


# ==================== 全局变量 ====================

decider_client = None
grounder_client = None
planner_client = None
planner_model = ""
decider_model = ""
grounder_model = ""


def init(service_ip, decider_port, grounder_port, planner_port):
    """初始化客户端连接"""
    global decider_client, grounder_client, planner_client
    
    decider_client = OpenAI(
        api_key="0",
        base_url=f"http://{service_ip}:{decider_port}/v1",
    )
    grounder_client = OpenAI(
        api_key="0",
        base_url=f"http://{service_ip}:{grounder_port}/v1",
    )
    planner_client = OpenAI(
        api_key="0",
        base_url=f"http://{service_ip}:{grounder_port}/v1",
    )


# ==================== 状态管理 ====================

def save_state(state: State, state_dir: str):
    """保存State到文件"""
    state_path = os.path.join(state_dir, "state.json")
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state.model_dump(), f, ensure_ascii=False, indent=4)
    logging.info(f"State saved to {state_path}")


def load_state(state_dir: str) -> Optional[State]:
    """从文件加载State"""
    state_path = os.path.join(state_dir, "state.json")
    if not os.path.exists(state_path):
        return None
    with open(state_path, "r", encoding="utf-8") as f:
        state_dict = json.load(f)
    state = State(**state_dict)
    logging.info(f"State loaded from {state_path}")
    return state


# ==================== 任务执行 ====================

def execute_task(task_description: str, device: Device, base_data_dir: str, extract_config: Optional[ExtractArtifactConfig] = None, use_qwen3=False, use_experience=False) -> dict:
    """
    统一的任务执行入口
    自动分析任务类型并选择合适的执行策略
    
    Args:
        task_description: 用户的原始任务描述
        device: 设备对象
        base_data_dir: 数据存储基础目录
        extract_config: Artifact提取配置
        use_qwen3: 是否使用Qwen3模型
        use_experience: 是否使用经验进行任务改写
    
    Returns:
        dict: 执行结果
    """
    logging.info(f"\n{'='*60}")
    logging.info(f"开始执行任务: {task_description}")
    logging.info(f"{'='*60}\n")
    
    # Step 1: 分析任务类型
    logging.info("Step 1: 分析任务类型...")
    is_multi_stage, reason = analyze_task(
        task_description,
        planner_client,
        planner_model
    )
    
    logging.info(f"任务类型: {'多阶段' if is_multi_stage else '单阶段'}")
    logging.info(f"分析理由: {reason}")
    
    # 创建任务目录
    existing_dirs = [d for d in os.listdir(base_data_dir) 
                     if os.path.isdir(os.path.join(base_data_dir, d)) and d.isdigit()]
    task_index = max([int(d) for d in existing_dirs], default=0) + 1
    task_dir = os.path.join(base_data_dir, str(task_index))
    os.makedirs(task_dir)
    
    # 保存任务分析结果
    analysis_path = os.path.join(task_dir, "task_analysis.json")
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump({
            "is_multi_stage": is_multi_stage,
            "reason": reason
        }, f, ensure_ascii=False, indent=4)
    
    # Step 2: 根据任务类型选择执行策略
    if is_multi_stage:
        logging.info("\n执行策略: 多阶段任务模式")
        logging.info(f"{'='*60}\n")
        logging.info(f"使用Qwen3: {use_qwen3}")
        
        result = execute_multi_stage_task_internal(task_description, device, task_dir, extract_config, use_qwen3=use_qwen3)
        
        return {
            "task_description": task_description,
            "task_type": "multi",
            "analysis": {"is_multi_stage": is_multi_stage, "reason": reason},
            "result": result
        }
    
    else:
        logging.info("\n执行策略: 单阶段任务模式")
        logging.info(f"{'='*60}\n")
        logging.info(f"使用经验: {use_experience}")
        logging.info(f"使用Qwen3: {use_qwen3}")
        
        # 获取APP信息（调用Planner进行经验检索和任务重写）
        try:
            app_name, package_name, refined_description = get_app_package_name(
                task_description,
                planner_client,
                planner_model
            )
        except Exception as e:
            logging.error(f"获取应用信息失败: {e}")
            return {
                "task_description": task_description,
                "task_type": "single",
                "success": False,
                "error": f"获取应用信息失败: {e}"
            }
        
        if not app_name or not package_name:
            logging.error("无法确定目标应用")
            return {
                "task_description": task_description,
                "task_type": "single",
                "success": False,
                "error": "无法确定目标应用"
            }
        
        logging.info(f"应用: {app_name} ({package_name})")
        logging.info(f"重写后的任务描述: {refined_description}")
        
        # 根据 use_experience 参数决定是否使用 planner 改写的任务描述
        if use_experience:
            logging.info("使用经验: 使用Planner重写后的任务描述")
            final_task_description = refined_description
        else:
            logging.info("不使用经验: 使用原始任务描述")
            final_task_description = task_description
        
        logging.info(f"最终任务描述: {final_task_description}")
        
        # 启动APP
        try:
            device.app_start(package_name)
            time.sleep(2)
        except Exception as e:
            logging.error(f"启动APP失败: {e}")
            return {
                "task_description": task_description,
                "task_type": "single",
                "success": False,
                "error": str(e)
            }
        
        # 执行任务
        try:
            execution_result = task_in_app(
                app=app_name,
                old_task=task_description,
                task=final_task_description,  # 使用根据use_experience选择的任务描述
                device=device,
                data_dir=task_dir,
                decider_client=decider_client,
                grounder_client=grounder_client,
                decider_model=decider_model,
                grounder_model=grounder_model,
                bbox_flag=True,
                use_qwen3=use_qwen3  # 传递 use_qwen3 参数
            )
            
            return {
                "task_description": task_description,
                "task_type": "single",
                "app_name": app_name,
                "package_name": package_name,
                "refined_description": refined_description,
                "success": execution_result.get("success", False),
                "action_count": len(execution_result.get("actions", []))
            }
        
        except Exception as e:
            logging.error(f"执行任务失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                "task_description": task_description,
                "task_type": "single",
                "success": False,
                "error": str(e)
            }


def execute_multi_stage_task_internal(task_description: str, device: Device, task_dir: str, extract_config: Optional[ExtractArtifactConfig] = None, use_qwen3=False) -> dict:
    """
    内部函数：执行多阶段任务（不创建新的任务目录）
    
    Args:
        task_description: 任务描述
        device: 设备对象
        task_dir: 任务目录（已创建）
        extract_config: Artifact提取配置
        use_qwen3: 是否使用Qwen3模型
    
    Returns:
        dict: 执行结果
    """
    # 生成Plan
    logging.info("生成任务规划...")
    plan = generate_plan(
        task_description,
        planner_client,
        planner_model
    )
    
    if not plan:
        logging.error("生成计划失败")
        return {
            "success": False,
            "error": "生成计划失败"
        }
    
    logging.info(f"生成的Plan包含 {len(plan.subtasks)} 个子任务")
    
    # 创建State
    state = State(
        task_description=task_description,
        plan=plan,
        current_subtask_index=0,
        artifacts={},
        completed_subtasks=[],
        extract_config=extract_config if extract_config else ExtractArtifactConfig()
    )
    
    # 保存Plan
    plan_path = os.path.join(task_dir, "plan.json")
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan.model_dump(), f, ensure_ascii=False, indent=4)
    
    # 依次执行子任务
    while state.current_subtask_index < len(plan.subtasks):
        subtask_index = state.current_subtask_index
        subtask = plan.subtasks[subtask_index]
        
        logging.info(f"\n{'='*60}")
        logging.info(f"执行子任务 {subtask.subtask_id}/{len(plan.subtasks)}")
        logging.info(f"应用: {subtask.app_name}")
        logging.info(f"描述: {subtask.description}")
        logging.info(f"{'='*60}\n")
        
        # 检查依赖
        for dep_id in subtask.depends_on:
            if dep_id not in state.completed_subtasks:
                logging.error(f"子任务 {subtask.subtask_id} 依赖的子任务 {dep_id} 尚未完成")
                state.current_subtask_index += 1
                continue
        
        # 完善任务描述
        refined_description = subtask.description
        if state.artifacts:
            refined_description = refine_next_subtask_description(
                subtask.description,
                state.artifacts,
                planner_client,
                planner_model
            )
            logging.info(f"完善后的任务描述: {refined_description}")
        
        # 创建子任务数据目录
        subtask_dir = os.path.join(task_dir, f"subtask_{subtask.subtask_id}")
        os.makedirs(subtask_dir, exist_ok=True)
        
        # 启动APP
        try:
            device.app_start(subtask.package_name)
            time.sleep(2)
        except Exception as e:
            logging.error(f"启动APP失败: {e}")
            artifact = Artifact(
                subtask_id=subtask.subtask_id,
                success=False,
                summary=f"启动APP失败: {e}",
                data={}
            )
            state.artifacts[f"subtask_{subtask.subtask_id}"] = artifact
            state.current_subtask_index += 1
            save_state(state, task_dir)
            continue
        
        # 执行子任务
        try:
            execution_result = task_in_app(
                app=subtask.app_name,
                old_task=task_description,
                task=refined_description,
                device=device,
                data_dir=subtask_dir,
                decider_client=decider_client,
                grounder_client=grounder_client,
                decider_model=decider_model,
                grounder_model=grounder_model,
                bbox_flag=True,
                use_qwen3=use_qwen3  # 传递 use_qwen3 参数
            )
            
            # 从结果中提取artifact
            artifact = extract_artifact(
                subtask.subtask_id,
                subtask.description,
                subtask.artifact_schema,
                execution_result,
                planner_client,
                planner_model,
                state.extract_config  # 传递配置
            )
            
            if artifact and artifact.success:
                state.artifacts[f"subtask_{subtask.subtask_id}"] = artifact
                logging.info(f"提取的artifact: {artifact}")
            
            state.completed_subtasks.append(subtask.subtask_id)
            state.current_subtask_index += 1
            
            # 保存State
            save_state(state, task_dir)
            
        except Exception as e:
            logging.error(f"执行子任务失败: {e}")
            import traceback
            traceback.print_exc()
            
            artifact = Artifact(
                subtask_id=subtask.subtask_id,
                success=False,
                summary=f"执行失败: {e}",
                data={}
            )
            state.artifacts[f"subtask_{subtask.subtask_id}"] = artifact
            state.current_subtask_index += 1
            save_state(state, task_dir)
            continue
    
    # 返回最终结果
    return {
        "success": len(state.completed_subtasks) == len(plan.subtasks),
        "total_subtasks": len(plan.subtasks),
        "completed_subtasks": len(state.completed_subtasks),
        "artifacts": {k: v.model_dump() for k, v in state.artifacts.items()}
    }


# ==================== 主程序 ====================

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="MobiAgent - 智能任务执行系统")
    parser.add_argument("--service_ip", type=str, default="localhost", 
                       help="Ip for the services (default: localhost)")
    parser.add_argument("--decider_port", type=int, default=8000, 
                       help="Port for decider service (default: 8000)")
    parser.add_argument("--grounder_port", type=int, default=8001, 
                       help="Port for grounder service (default: 8001)")
    parser.add_argument("--planner_port", type=int, default=8002, 
                       help="Port for planner service (default: 8002)")
    parser.add_argument("--task", type=str, default=None, 
                       help="直接指定任务描述")
    parser.add_argument("--task_file", type=str, default="task.json", 
                       help="任务列表文件路径 (default: task.json)")
    parser.add_argument("--use_ocr", action="store_true", default=True,
                       help="是否使用OCR提取页面文本 (default: True)")
    parser.add_argument("--num_screenshots", type=int, default=1,
                       help="使用最后几张截图进行artifact提取 (default: 1)")
    parser.add_argument("--use_text_with_image", action="store_true", default=True,
                       help="是否将文本和截图一起发送给模型 (default: True)")
    parser.add_argument("--enable_hybrid_ocr", action="store_true", default=True,
                       help="是否启用混合OCR识别 (default: True)")
    parser.add_argument("--use_qwen3", action="store_true", default=True,
                       help="是否使用Qwen3模型 (default: True)")
    parser.add_argument("--use_experience", action="store_true", default=False,
                       help="是否使用经验进行任务改写 (default: False)")
    
    args = parser.parse_args()

    # 使用命令行参数初始化
    init(args.service_ip, args.decider_port, args.grounder_port, args.planner_port)
    
    # 创建Artifact提取配置
    extract_config = ExtractArtifactConfig(
        use_ocr=args.use_ocr,
        num_screenshots=args.num_screenshots,
        use_text_with_image=args.use_text_with_image,
        enable_hybrid_ocr=args.enable_hybrid_ocr
    )
    
    device = AndroidDevice()
    logging.info("已连接到设备")

    data_base_dir = os.path.join(os.path.dirname(__file__), 'data')
    if not os.path.exists(data_base_dir):
        os.makedirs(data_base_dir)

    logging.info("=" * 60)
    logging.info("MobiAgent 智能任务执行系统")
    logging.info("每个任务将由Planner自动分析并选择最优执行策略")
    logging.info("=" * 60)
    
    # 记录执行配置
    logging.info(f"\n执行配置:")
    logging.info(f"  - 使用Qwen3模型: {args.use_qwen3}")
    logging.info(f"  - 使用经验进行任务改写: {args.use_experience}")
    logging.info(f"  - 使用OCR: {args.use_ocr}")
    logging.info(f"  - 启用混合OCR: {args.enable_hybrid_ocr}\n")
    
    task_list = []
    
    if args.task:
        # 命令行指定单个任务
        task_list = [args.task]
        logging.info(f"执行单个任务: {args.task}")
    else:
        # 从文件读取任务列表
        task_json_path = os.path.join(os.path.dirname(__file__), args.task_file)
        if os.path.exists(task_json_path):
            with open(task_json_path, "r", encoding="utf-8") as f:
                task_list = json.load(f)
            logging.info(f"从 {args.task_file} 读取到 {len(task_list)} 个任务")
        else:
            logging.error(f"未找到任务文件: {task_json_path}")
            logging.info("请使用 --task 参数指定任务，或创建task.json文件")
            exit(1)
    
    # 执行所有任务
    results = []
    for i, task_description in enumerate(task_list, 1):
        logging.info(f"\n\n{'#'*60}")
        logging.info(f"任务 {i}/{len(task_list)}: {task_description}")
        logging.info(f"{'#'*60}\n")
        
        try:
            # 使用统一入口执行任务，传递 use_qwen3 和 use_experience 参数
            result = execute_task(
                task_description, 
                device, 
                data_base_dir, 
                extract_config,
                use_qwen3=args.use_qwen3,
                use_experience=args.use_experience
            )
            results.append(result)
            
            # 输出执行结果摘要
            logging.info(f"\n{'='*60}")
            logging.info(f"任务 {i} 执行完成")
            logging.info(f"类型: {result.get('task_type', 'unknown')}")
            
            if result.get('task_type') == 'single':
                logging.info(f"成功: {result.get('success', False)}")
                logging.info(f"应用: {result.get('app_name', 'N/A')}")
                logging.info(f"操作数: {result.get('action_count', 0)}")
            elif result.get('task_type') == 'multi':
                multi_result = result.get('result', {})
                logging.info(f"成功: {multi_result.get('success', False)}")
                logging.info(f"子任务数: {multi_result.get('total_subtasks', 0)}")
                logging.info(f"完成数: {multi_result.get('completed_subtasks', 0)}")
            
            logging.info(f"{'='*60}\n")
            
        except Exception as e:
            logging.error(f"任务执行失败: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "task_description": task_description,
                "success": False,
                "error": str(e)
            })
    
    # 输出总体统计
    logging.info(f"\n\n{'#'*60}")
    logging.info("所有任务执行完成")
    logging.info(f"{'#'*60}")
    logging.info(f"总任务数: {len(task_list)}")
    
    single_tasks = [r for r in results if r.get('task_type') == 'single']
    multi_tasks = [r for r in results if r.get('task_type') == 'multi']
    
    logging.info(f"单阶段任务: {len(single_tasks)} 个")
    logging.info(f"多阶段任务: {len(multi_tasks)} 个")
    
    successful_single = sum(1 for r in single_tasks if r.get('success'))
    successful_multi = sum(1 for r in multi_tasks if r.get('result', {}).get('success'))
    
    logging.info(f"单阶段任务成功: {successful_single}/{len(single_tasks)}")
    logging.info(f"多阶段任务成功: {successful_multi}/{len(multi_tasks)}")
    
    logging.info(f"{'#'*60}\n")
