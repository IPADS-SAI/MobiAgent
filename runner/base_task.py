# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

import os
import os
import math
import shutil
import json
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import re
import textwrap
from PIL import Image, ImageDraw, ImageFont

# 使用模块级别的logger（级别由 setup_logging() 统一配置）
logger = logging.getLogger(__name__)

# 绘图常量
RED_DOT_COLOR = (255, 0, 0, 200)
SCROLL_COLOR = (0, 122, 255, 210)
SCROLL_WIDTH = 12
STRIP_BG_COLOR = "white"
STRIP_ACCENT_COLOR = "#0062ff"

ACTION_LABELS = {
    "open_app": "Open App",
    "click": "Tap",
    "longclick": "Long Press",
    "doubleclick": "Double Tap",
    "scroll": "Scroll",
    "input_text": "Input Text",
    "back": "Back",
    "home": "Home",
    "retry": "Retry",
    "wait": "Wait",
}

# 统一动作类型映射表：将各模型特定的动作名称映射为统一格式
# 格式: 原始名称 -> 规范名称
ACTION_TYPE_ALIASES = {
    # ==================== 点击相关 ====================
    'click': 'click',
    'tap': 'click',
    'left_single': 'click',
    'right_single': 'click',
    'hover': 'click',
    
    # ==================== 长按相关 ====================
    'longclick': 'longclick',
    'long_press': 'longclick',
    'long_click': 'longclick',
    
    # ==================== 双击相关 ====================
    'doubleclick': 'doubleclick',
    'double_tap': 'doubleclick',
    'double_click': 'doubleclick',
    'left_double': 'doubleclick',
    
    # ==================== 输入相关 ====================
    'input_text': 'input_text',
    'type': 'input_text',
    'input': 'input_text',
    'set_text': 'input_text',
    
    # ==================== 滑动相关 ====================
    'scroll': 'scroll',
    'swipe': 'scroll',
    'drag': 'scroll',
    
    # ==================== 启动应用 ====================
    'open_app': 'open_app',
    'open': 'open_app',
    'start_app': 'open_app',
    'launch': 'open_app',
    
    # ==================== 系统按键 ====================
    'home': 'home',
    'press_home': 'home',
    'key_home': 'home',
    'back': 'back',
    'press_back': 'back',
    'key_back': 'back',
    
    # ==================== 等待 ====================
    'wait': 'wait',
    
    # ==================== 任务终止 ====================
    'done': 'done',
    'finished': 'done',
    'complete': 'done',
    
    # ==================== 重试/失败 ====================
    'retry': 'retry',
    'fail': 'retry',
    'error': 'retry',
    
    # ==================== AutoGLM 特有动作 ====================
    'take_over': 'wait',   # 用户接管，暂时映射为等待
    'note': 'wait',        # 记录页面，暂不实现
    'call_api': 'wait',    # 调用API，暂不实现
    'interact': 'wait',    # 用户交互，暂时映射为等待
}

class BaseTask(ABC):
    """
    基础任务类，定义统一的任务执行接口
    所有模型适配器都需要继承此类并实现抽象方法
    
    支持两种执行模式：
    1. 步骤循环模式：框架控制循环，子类实现execute_step返回单步动作
    2. 一次性执行模式：子类实现_execute_task完成整个任务（用于已有循环逻辑的模型）
    """
    
    def __init__(
        self,
        task_description: str,
        device,
        data_dir: str,
        device_type: str = "Android",
        max_steps: int = 40,
        max_retries: int = 3,
        use_step_loop: bool = False,
        draw: bool = False,
        enable_planning: bool = False,
        **kwargs
    ):
        """
        初始化基础任务
        
        Args:
            task_description: 任务描述
            device: 设备对象（AndroidDevice或HarmonyDevice）
            data_dir: 数据保存目录
            device_type: 设备类型 ("Android" 或 "Harmony")
            max_steps: 最大步骤数
            max_retries: 最大重试次数
            use_step_loop: 是否使用步骤循环模式
            draw: 是否绘制操作可视化
            enable_planning: 是否启用任务规划（可分析APP、结合experience和profile等）
            **kwargs: 其他参数
        """
        self.task_description = task_description
        self.original_task_description = task_description  # 保存原始任务描述
        self.device = device
        self.data_dir = data_dir
        self.device_type = device_type
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.use_step_loop = use_step_loop
        self.draw = draw
        self.enable_planning = enable_planning
        
        # 确保数据目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.actions = []
        self.reacts = []
        self.step_count = 0
        self.retry_count = 0
        self.total_time = 0.0
        
        # planning 相关属性
        self.app_name = None
        self.package_name = None
        self.planning_info = {}  # 存储 planning 返回的其他信息
        
        logger.info(f"初始化任务: {task_description}")
        logger.info(f"数据目录: {data_dir}")
        logger.info(f"设备类型: {device_type}")
        logger.info(f"使用步骤循环: {use_step_loop}")
        logger.info(f"绘制可视化: {draw}")
        logger.info(f"启用规划: {enable_planning}")
    
    def execute(self) -> Dict:
        """
        执行任务的主流程
        
        Returns:
            包含任务执行结果的字典
        """
        logger.info(f"开始执行任务: {self.task_description}")
        start_time = time.time()
        
        try:
            # 如果启用 planning，先进行任务规划
            if self.enable_planning:
                logger.info("执行 planning...")
                try:
                    planning_result = self._plan_task()
                    if planning_result:
                        # 更新任务描述（如果 planning 返回了优化后的描述）
                        if 'task_description' in planning_result:
                            self.task_description = planning_result['task_description']
                            logger.info(f"任务描述已优化: {self.task_description}")
                        
                        # 更新 APP 信息
                        if 'app_name' in planning_result:
                            self.app_name = planning_result['app_name']
                            logger.info(f"目标 APP: {self.app_name}")
                        
                        if 'package_name' in planning_result:
                            self.package_name = planning_result['package_name']
                            logger.info(f"包名: {self.package_name}")
                        
                        # 保存其他 planning 信息
                        self.planning_info = planning_result
                        
                        # 如果需要启动应用
                        if self.package_name:
                            logger.info(f"启动应用: {self.package_name}")
                            self.device.app_start(self.package_name)
                except Exception as e:
                    logger.warning(f"planning 失败，继续使用原始任务: {e}")
            
            logger.info(60*"=")
            logger.info(f"step_count: {self.step_count}")
            logger.info(60*"=")
            if self.use_step_loop:
                # 步骤循环模式：框架控制循环
                result = self._execute_with_step_loop()
                
            else:
                # 一次性执行模式：子类自己控制循环
                result = self._execute_task()
            logger.info(60*"=")

            # 如果启动了应用，执行完成后停止应用
            if self.enable_planning and self.package_name:
                try:
                    logger.info(f"停止应用: {self.package_name}")
                    self.device.app_stop(self.package_name)
                except Exception as e:
                    logger.warning(f"停止应用失败: {e}")
            
            self.total_time = time.time() - start_time
            
            # 保存结果
            self._save_results(result)
            
            logger.info(f"任务已完成，耗时{self.total_time:.2f}秒")
            return result
            
        except Exception as e:
            logger.error(f"任务执行失败: {e}", exc_info=True)
            self.total_time = time.time() - start_time
            result = {
                "status": "error",
                "error": str(e),
                "step_count": self.step_count,
                "total_time": self.total_time
            }
            self._save_results(result)
            return result
    
    def _execute_with_step_loop(self) -> Dict:
        """
        使用步骤循环模式执行任务
        
        Returns:
            任务执行结果字典
        """
        logger.info("使用步骤循环模式")
        
        while True:
            if self.step_count >= self.max_steps:
                logger.warning(f"达到最大步数: {self.max_steps}")
                return {
                    "status": "max_steps_reached",
                    "step_count": self.step_count,
                    "message": f"任务达到最大步数 ({self.max_steps})"
                }
            
            self.step_count += 1
            logger.info(f"\n{'='*50}")
            logger.info(f"Step {self.step_count}/{self.max_steps}")
            logger.info(f"{'='*50}")
            
            try:
                screenshot_path = self._save_screenshot(self.step_count)
                hierarchy_path = self._save_hierarchy(self.step_count)
            except Exception as e:
                logger.error(f"保存状态失败: {e}")
            
            try:
                step_start_time = time.time()
                action_seq = self.execute_step(self.step_count)
                step_elapsed_time = time.time() - step_start_time
                
                logger.info(f"Step {self.step_count} 动作: {json.dumps(action_seq, ensure_ascii=False)}")
                
                # 统一规范化动作类型
                action_seq = [self._normalize_action(a) for a in action_seq]
                logger.debug(f"规范化后动作: {json.dumps(action_seq, ensure_ascii=False)}")
                
                should_continue = self._execute_action_seq(action_seq)

                if self.draw:
                    try:
                        self._draw_actions_on_image(self.step_count, action_seq)
                    except Exception as e:
                        logger.warning(f"绘制失败: {e}")
                
                if not should_continue:
                    status = "completed"
                    for action in action_seq:
                        if action.get("type") == "done":
                            status = action.get("params", {}).get("status", "success")
                            break
                        elif action.get("type") == "retry" and self.retry_count >= self.max_retries:
                            status = "failed"
                            break
                    
                    return {
                        "status": status,
                        "step_count": self.step_count,
                        "message": f"Step {self.step_count} 任务完成"
                    }
                
                time.sleep(2)
                
                if hasattr(self, 'reflect_action') and callable(getattr(self, 'reflect_action', None)):            
                    try:
                        reflect_start_time = time.time()
                        self.reflect_action(self.step_count)
                        step_elapsed_time += time.time() - reflect_start_time
                    except Exception as e:
                        logger.warning(f"反思失败: {e}")
                
            except Exception as e:
                logger.error(f"Step {self.step_count} 失败: {e}", exc_info=True)
                return {
                    "status": "error",
                    "step_count": self.step_count,
                    "error": str(e)
                }
    
    # ... (rest of methods unchanged until _save_results needed access to PIL?)
    # 绘制方法将在类末尾实现

    def _execute_task(self) -> Dict:
        """
        子类一次性任务执行逻辑（用于已有循环逻辑的模型）
        
        Returns:
            包含任务执行结果的字典
        """
        raise NotImplementedError(
            "Either implement _execute_task() for one-time execution "
            "or implement execute_step() with use_step_loop=True"
        )
    
    def execute_step(self, step_index: int) -> List[Dict]:
        """
        子类单步执行逻辑（用于步骤循环模式）
        
        Args:
            step_index: 当前步骤索引（从1开始）
            
        Returns:
            动作序列列表，每个动作是一个字典，包含：
            - type: 动作类型（click, type, swipe, done等）
            - params: 动作参数
        """
        raise NotImplementedError(
            "Either implement execute_step() with use_step_loop=True "
            "or implement _execute_task() for one-time execution"
        )
    
    def _normalize_action(self, action: Dict) -> Dict:
        """
        规范化动作字典，将各模型特定的动作名称映射为统一格式
        
        Args:
            action: 原始动作字典，包含 type 和 params
            
        Returns:
            规范化后的动作字典
        """
        if not isinstance(action, dict):
            return action
            
        action_type = action.get('type', '').lower()
        params = action.get('params', {})
        
        # 映射动作类型
        normalized_type = ACTION_TYPE_ALIASES.get(action_type, action_type)
        
        # 规范化参数名称
        normalized_params = self._normalize_params(normalized_type, params)
        
        return {'type': normalized_type, 'params': normalized_params}
    
    def _normalize_params(self, action_type: str, params: Dict) -> Dict:
        """
        规范化参数格式，统一坐标参数名称
        
        Args:
            action_type: 规范化后的动作类型
            params: 原始参数字典
            
        Returns:
            规范化后的参数字典
        """
        if not isinstance(params, dict):
            return params
            
        result = dict(params)
        
        # 点击/长按/双击动作的坐标参数统一
        if action_type in ['click', 'longclick', 'doubleclick']:
            # 将 'points' 格式转换为 'coordinate' 格式
            if 'points' in result and 'coordinate' not in result:
                points = result['points']
                if isinstance(points, list) and len(points) >= 2:
                    result['coordinate'] = [points[0], points[1]]
            
            # 将 'position_x'/'position_y' 格式转换为 'coordinate' 格式
            if 'position_x' in result and 'position_y' in result and 'coordinate' not in result:
                result['coordinate'] = [result['position_x'], result['position_y']]
        
        # 滑动动作的坐标参数统一
        if action_type == 'scroll':
            # 将 'start'/'end' 格式转换为 'start_coordinate'/'end_coordinate'
            if 'start' in result and 'start_coordinate' not in result:
                result['start_coordinate'] = result.pop('start')
            if 'end' in result and 'end_coordinate' not in result:
                result['end_coordinate'] = result.pop('end')
        
        # 输入动作的文本参数统一
        if action_type == 'input_text':
            # 将 'content' 格式转换为 'text' 格式
            if 'content' in result and 'text' not in result:
                result['text'] = result.pop('content')
        
        return result
    
    def _execute_action_seq(self, action_seq: List[Dict]) -> bool:
        """
        执行动作序列
        
        Args:
            action_seq: 动作序列列表
            
        Returns:
            是否继续执行（True 继续，False 停止）
        """
        for action in action_seq:
            action_type = action.get('type')
            params = action.get('params', {})
            
            if action_type == 'retry':
                self.retry_count += 1
                logger.warning(f"重试次数: {self.retry_count}/{self.max_retries}")
                if self.retry_count >= self.max_retries:
                    logger.error(f"达到最大重试次数 ({self.max_retries})")
                    return False
                continue
            else:
                self.retry_count = 0
            
            if action_type == 'done':
                status = params.get('status', 'completed')
                logger.info(f"任务完成，状态: {status}")
                return False
            
            try:
                self._perform_action(action_type, params)
                self._add_action(
                    action_type=action_type,
                    step_index=self.step_count,
                    **params
                )
            except Exception as e:
                logger.error(f"执行动作 {action_type} 失败: {e}")
                raise
        
        return True
    
    def _perform_action(self, action_type: str, params: Dict):
        """
        执行具体动作
        
        Args:
            action_type: 动作类型
            params: 动作参数
        """
        logger.info(f"执行 {action_type}：{params}")
        
        action_type = action_type.lower()
        
        if action_type in ['click', 'tap']:
            if 'coordinate' in params:
                x, y = params['coordinate']
            elif 'position_x' in params and 'position_y' in params:
                x, y = params['position_x'], params['position_y']
            elif 'points' in params:
                x, y = params['points']
            else:
                raise ValueError("Click 动作缺少坐标信息")
            self.device.click(x, y)
            
        elif action_type in ['long_press', 'long_click', 'longclick']:
            if 'coordinate' in params:
                x, y = params['coordinate']
            elif 'position_x' in params and 'position_y' in params:
                x, y = params['position_x'], params['position_y']
            elif 'points' in params:
                x, y = params['points']
            else:
                raise ValueError("Long press action missing coordinate information")
            self.device.long_click(x, y)
            
        elif action_type in ['double_tap', 'double_click', 'doubleclick']:
            if 'coordinate' in params:
                x, y = params['coordinate']
            elif 'position_x' in params and 'position_y' in params:
                x, y = params['position_x'], params['position_y']
            elif 'points' in params:
                x, y = params['points']
            else:
                raise ValueError("Double tap action missing coordinate information")
            self.device.double_click(x, y)
            
        elif action_type in ['type', 'input', 'set_text', 'input_text']:
            text = params.get('text', '')
            self.device.input(text)
            # 尝试在输入后直接回车确定、搜索
            time.sleep(0.2)
            self.device.keyevent('ENTER')
            
        elif action_type in ['swipe', 'scroll']:
            if 'direction' in params:
                direction = params['direction']
                scale = params.get('scale', 0.5)
                self.device.swipe(direction, scale)
            elif 'start_coordinate' in params and 'end_coordinate' in params:
                start = params['start_coordinate']
                end = params['end_coordinate']
                self.device.swipe_with_coords(start[0], start[1], end[0], end[1])
            elif 'points' in params and len(params['points']) == 4:
                x1, y1, x2, y2 = params['points']
                self.device.swipe_with_coords(x1, y1, x2, y2)
            else:
                raise ValueError("Swipe action missing coordinate information")
            
        elif action_type == 'back':
            self.device.keyevent('BACK')
            
        elif action_type == 'home':
            self.device.keyevent('HOME')
            
        elif action_type in ['open', 'start_app', 'launch', 'open_app']:
            app_name = params.get('app_name')
            package_name = params.get('package_name')
            if app_name:
                self.device.start_app(app_name)
            elif package_name:
                self.device.app_start(package_name)
                
        elif action_type == 'wait':
            seconds = params.get('seconds', 1)
            time.sleep(seconds)
            
        else:
            logger.warning(f"未知的动作类型: {action_type}")
    
    def reflect_action(self, step_index: int):
        """
        可选的反思方法，子类可以实现以改进后续步骤
        
        Args:
            step_index: 当前步骤索引
        """
        pass
    
    def _plan_task(self) -> Optional[Dict]:
        """
        任务规划方法，子类可以实现以进行任务分析和优化
        
        此方法在任务执行前调用，可以用于：
        - 分析任务描述，确定需要使用的 APP
        - 结合历史经验 (experience) 优化任务描述
        - 使用用户画像 (profile) 个性化任务
        - 返回任务执行所需的额外信息
        
        Returns:
            包含规划结果的字典，可能包含：
            - task_description: 优化后的任务描述
            - app_name: 目标应用名称
            - package_name: 应用包名
            - 其他任务执行所需的信息
            
            如果不需要 planning 或 planning 失败，返回 None
        
        Example:
            {
                "task_description": "在淘宝搜索苹果手机并查看价格",
                "app_name": "淘宝",
                "package_name": "com.taobao.taobao",
                "experience_used": True
            }
        """
        # 默认实现：不进行 planning
        # 子类可以覆盖此方法实现具体的 planning 逻辑
        logger.debug("Default _plan_task implementation: no planning performed")
        return None
    
    def _save_screenshot(self, step_index: int) -> str:
        """保存截图"""
        screenshot_path = os.path.join(self.data_dir, f"{step_index}.jpg")
        self.device.screenshot(screenshot_path)
        return screenshot_path
    
    def _save_hierarchy(self, step_index: int) -> str:
        """保存 UI 层级结构"""
        try:
            hierarchy = self.device.dump_hierarchy()
            
            if self.device_type == "Android":
                hierarchy_path = os.path.join(self.data_dir, f"{step_index}.xml")
                with open(hierarchy_path, "w", encoding="utf-8") as f:
                    f.write(hierarchy)
            else:
                hierarchy_path = os.path.join(self.data_dir, f"{step_index}.json")
                if isinstance(hierarchy, str):
                    try:
                        hierarchy_json = json.loads(hierarchy)
                    except json.JSONDecodeError:
                        hierarchy_json = {"raw": hierarchy}
                else:
                    hierarchy_json = hierarchy
                    
                with open(hierarchy_path, "w", encoding="utf-8") as f:
                    json.dump(hierarchy_json, f, ensure_ascii=False, indent=2)
                    
            return hierarchy_path
            
        except Exception as e:
            logger.error(f"保存 Step {step_index} 的 UI 层级失败: {e}")
            return ""
    
    def _save_results(self, result: Dict):
        """保存任务执行结果"""
        actions_data = {
            "original_task_description": self.original_task_description,
            "task_description": self.task_description,
            "device_type": self.device_type,
            "action_count": len(self.actions),
            "actions": self.actions,
            "total_time": self.total_time,
            "status": result.get("status", "unknown")
        }
        
        if self.enable_planning:
            actions_data["planning_enabled"] = True
            if self.app_name:
                actions_data["app_name"] = self.app_name
            if self.package_name:
                actions_data["package_name"] = self.package_name
            if self.planning_info:
                actions_data["planning_info"] = self.planning_info
        
        actions_path = os.path.join(self.data_dir, "actions.json")
        with open(actions_path, "w", encoding="utf-8") as f:
            json.dump(actions_data, f, ensure_ascii=False, indent=2)
        
        react_path = os.path.join(self.data_dir, "react.json")
        with open(react_path, "w", encoding="utf-8") as f:
            json.dump(self.reacts, f, ensure_ascii=False, indent=2)
        
        logger.info(f"结果已保存到 {self.data_dir}")
    
    def _add_action(self, action_type: str, step_index: int, **kwargs):
        """添加动作记录"""
        action = {
            "type": action_type,
            "action_index": step_index,
            **kwargs
        }
        self.actions.append(action)
    
    def _add_react(self, reasoning: str, action: str, parameters: Dict, step_index: int):
        """添加推理记录"""
        react = {
            "reasoning": reasoning,
            "function": {
                "name": action,
                "parameters": parameters
            },
            "action_index": step_index
        }
        self.reacts.append(react)

    # ----------------------------
    # 绘制方法
    # ----------------------------

    def _draw_actions_on_image(self, step_index: int, action_seq: List[Dict]):
        """绘制动作到截图上"""
        src_path = os.path.join(self.data_dir, f"{step_index}.jpg")
        dst_path = os.path.join(self.data_dir, f"{step_index}_draw.jpg")
        
        if not os.path.exists(src_path):
            logger.warning(f"源截图不存在: {src_path}")
            return

        try:
            image = Image.open(src_path)
            # 使用原始图像大小进行绘制
            width, height = image.size
            modified = False
            
            # 因为 action_seq 可能包含多个动作，我们将绘制所有可视化动作
            # 并为描述条使用最后一个动作（或聚合）
            
            last_action_desc = None
            
            for action in action_seq:
                action_type = action.get('type', '').lower()
                params = action.get('params', {})
                
                # 规范化类型
                if action_type == 'tap': action_type = 'click'
                if action_type == 'long_press': action_type = 'longclick'
                if action_type == 'double_tap': action_type = 'doubleclick'
                if action_type == 'input': action_type = 'input_text'
                if action_type in ['launch', 'start_app', 'open']: action_type = 'open_app'
                
                # 提取坐标
                tap_point = None
                gesture = None
                
                if 'points' in params:
                    pts = params['points']
                    if len(pts) == 2:
                        tap_point = (pts[0], pts[1])
                    elif len(pts) == 4:
                        gesture = ((pts[0], pts[1]), (pts[2], pts[3]))
                elif 'coordinate' in params:
                    tap_point = (params['coordinate'][0], params['coordinate'][1])
                elif 'start_coordinate' in params and 'end_coordinate' in params:
                    gesture = (
                        (params['start_coordinate'][0], params['start_coordinate'][1]),
                        (params['end_coordinate'][0], params['end_coordinate'][1])
                    )
                elif 'direction' in params:
                    w, h = width, height
                    cx, cy = w // 2, h // 2
                    d = 300
                    direction = params['direction'].lower()
                    if direction == 'up': gesture = ((cx, cy+d//2), (cx, cy-d//2))
                    elif direction == 'down': gesture = ((cx, cy-d//2), (cx, cy+d//2))
                    elif direction == 'left': gesture = ((cx+d//2, cy), (cx-d//2, cy))
                    elif direction == 'right': gesture = ((cx-d//2, cy), (cx+d//2, cy))
                
                # 绘制动作
                if action_type in ['click', 'longclick', 'doubleclick'] and tap_point:
                    image = self._draw_tap_overlay(image, tap_point)
                    modified = True
                    
                elif action_type in ['scroll', 'swipe'] and gesture:
                    image = self._draw_scroll_overlay(image, gesture)
                    modified = True
                    
                # 准备动作描述
                payload = params.get('text') or params.get('app_name') or str(tap_point or gesture or "")
                
                detail = f"Action payload: {payload}"
                action_label = ACTION_LABELS.get(action_type, action_type.replace("_", " ").title())
                
                if action_type in ['click', 'longclick', "doubleclick"]:
                    verb = action_label
                    detail = f"{verb} at {tap_point}" if tap_point else f"{verb}"
                elif action_type == 'open_app':
                    detail = f"Open app: {payload or 'Unknown'}"
                elif action_type in ['scroll', 'swipe']:
                    if gesture:
                         detail = f"Scroll from {gesture[0]} to {gesture[1]}"
                    else:
                         detail = f"Scroll {params.get('direction', 'unknown')}"
                elif action_type == 'input_text':
                    detail = f"Input text: {payload}"
                elif action_type == 'back':
                    detail = "Go back to previous screen"
                elif action_type == 'home':
                    detail = "Return to the home screen"
                elif action_type == 'wait':
                    detail = f"Wait for {params.get('seconds', 1)} seconds"
                    
                last_action_desc = {
                   "label": action_label,
                   "detail": detail
                }
            
            # 如果有动作记录，则保存
            if last_action_desc:
                # 添加描述条
                image = self._add_description_strip(image, last_action_desc)
                modified = True
            
            if modified:
                image.convert("RGB").save(dst_path)
                logger.info(f"保存可视化图像：{dst_path}")
            else:
                shutil.copy(src_path, dst_path)

        except Exception as e:
            logger.warning(f"绘制动作失败: {e}")

    def _draw_tap_overlay(self, image: Image.Image, coordinates: tuple, radius: int = 20) -> Image.Image:
        base = image.convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        x, y = coordinates
        left_up = (x - radius, y - radius)
        right_down = (x + radius, y + radius)
        draw.ellipse([left_up, right_down], fill=RED_DOT_COLOR)
        return Image.alpha_composite(base, overlay)

    def _draw_scroll_overlay(self, image: Image.Image, gesture: tuple) -> Image.Image:
        base = image.convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        start, end = gesture
        draw.line([start, end], fill=SCROLL_COLOR, width=SCROLL_WIDTH)
        self._draw_arrow_head(draw, start, end, SCROLL_COLOR)
        draw.ellipse(
            [
                (start[0] - 15, start[1] - 15),
                (start[0] + 15, start[1] + 15)
            ],
            outline=SCROLL_COLOR,
            width=6
        )
        return Image.alpha_composite(base, overlay)

    def _draw_arrow_head(self, draw, start, end, color, size=40):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length == 0:
            return
        ux, uy = dx / length, dy / length
        left = (end[0] - ux * size - uy * size / 2, end[1] - uy * size + ux * size / 2)
        right = (end[0] - ux * size + uy * size / 2, end[1] - uy * size - ux * size / 2)
        draw.polygon([end, left, right], fill=color)

    def _load_font(self, font_size: int) -> ImageFont.ImageFont:
        """加载字体，优先使用 msyh.ttf，失败则使用默认字体"""
        font_path = os.path.join(os.path.dirname(__file__),"..", "msyh.ttf")
        possible_paths = [
             font_path ,
             "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
             "/System/Library/Fonts/Helvetica.ttc"
        ]
        
        for p in possible_paths:
            if os.path.exists(p):
                try:
                    return ImageFont.truetype(p, font_size)
                except:
                    continue
        
        # 降级使用默认字体
        try:
             return ImageFont.load_default()
        except:
             return None

    def _calculate_characters_per_line(self, image_width: int, font: ImageFont.ImageFont) -> int:
        """计算一行能放入的字符数"""
        if not font: return 50
        sample_text = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        try:
            total_width = 0
            for char in sample_text:
                bbox = font.getbbox(char)
                total_width += bbox[2] - bbox[0]
            average_char_width = max(1, total_width // len(sample_text))
            return max(1, image_width // average_char_width)
        except:
             # 默认字体降级处理
             return image_width // 15

    def _add_description_strip(self, image: Image.Image, action_desc: Dict, font_size: int = 40, line_spacing: int = 10) -> Image.Image:
        """在图像下方添加动作描述条"""
        image = image.convert("RGB")
        width, height = image.size
        
        font = self._load_font(font_size)
        characters_per_line = self._calculate_characters_per_line(width, font)
        
        segments = [
            {"text": f"Action: {action_desc['label']}", "color": "red"},
            {"text": f"Detail: {action_desc['detail']}", "color": "black"}
        ]

        wrapped_lines = []
        wrapper = textwrap.TextWrapper(width=characters_per_line, break_long_words=True, break_on_hyphens=False)
        
        for segment in segments:
            lines = segment["text"].splitlines() or [segment["text"]]
            for line in lines:
                wrapped = wrapper.wrap(line) or [line]
                for wrapped_line in wrapped:
                    wrapped_lines.append((wrapped_line, segment["color"]))
        
        # 计算文本高度
        text_height = 0
        dummy_draw = ImageDraw.Draw(image)
        
        for line, _ in wrapped_lines:
            if font:
                 bbox = dummy_draw.textbbox((0, 0), line, font=font)
                 h_line = bbox[3] - bbox[1]
            else:
                 h_line = 20
            
            text_height += h_line + line_spacing

        strip_height = text_height + 20
        accent_height = 10
        
        new_image = Image.new("RGB", (width, height + accent_height + strip_height), STRIP_BG_COLOR)
        new_image.paste(image, (0, 0))
        draw = ImageDraw.Draw(new_image)
        draw.rectangle([(0, height), (width, height + accent_height)], fill=STRIP_ACCENT_COLOR)

        y = height + accent_height + 10
        for line, color in wrapped_lines:
            if font:
                 bbox = draw.textbbox((0, 0), line, font=font)
                 text_width = bbox[2] - bbox[0]
                 h_line = bbox[3] - bbox[1]
            else:
                 text_width = len(line) * 8
                 h_line = 20
                 
            x = (width - text_width) // 2
            
            if font:
                draw.text((x, y), line, font=font, fill=color)
            else:
                draw.text((x, y), line, fill=color)
            
            y += h_line + line_spacing

        return new_image

