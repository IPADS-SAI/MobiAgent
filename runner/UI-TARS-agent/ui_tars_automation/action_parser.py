#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UI-TARS 动作解析器 - 修复坐标转换问题
"""

import re
import ast
import logging
from typing import Dict, Tuple, Optional

# 直接使用UI-TARS官方库
import ui_tars.action_parser as ui_tars_parser

logger = logging.getLogger(__name__)


class ActionParser:
    """动作解析器 - 完全使用UI-TARS官方解析功能"""
    
    @staticmethod
    def parse_response(response: str, image_height: Optional[int] = None, 
                      image_width: Optional[int] = None, model_type: str = "qwen25vl") -> Tuple[str, str, Dict]:
        """
        解析模型响应，使用UI-TARS官方完整流程：
        1. parse_action_to_structure_output - 解析响应为结构化数据
        2. parsing_response_to_pyautogui_code - 转换为PyAutoGUI代码
        3. 从PyAutoGUI代码中提取最终坐标
        
        Args:
            response: 模型原始响应
            image_height: 原始图片高度
            image_width: 原始图片宽度
            model_type: 模型类型
            
        Returns:
            (thought, raw_action, parsed_action)
        """
        if not image_height or not image_width:
            raise ValueError("需要提供image_height和image_width进行坐标转换")
        
        try:
            # 步骤1: 计算smart resize尺寸
            smart_height, smart_width = ui_tars_parser.smart_resize(
                image_height, image_width, 
                factor=ui_tars_parser.IMAGE_FACTOR
            )
            
            logger.debug(f"原始图像尺寸: {image_width}x{image_height}")
            logger.debug(f"Smart resize尺寸: {smart_width}x{smart_height}")
            logger.debug(f"模型响应: {response}")
            
            # 步骤2: 使用官方parse_action_to_structure_output解析
            actions = ui_tars_parser.parse_action_to_structure_output(
                response,
                factor=ui_tars_parser.IMAGE_FACTOR,
                origin_resized_height=smart_height,
                origin_resized_width=smart_width,
                model_type=model_type
            )
            
            if not actions:
                raise ValueError("UI-TARS解析未返回有效动作")
            
            # 取第一个动作
            action = actions[0]
            
            # 提取thought和raw_action
            thought = action.get('thought', '')
            raw_action = ActionParser._extract_raw_action(response)
            
            # 步骤3: 使用官方parsing_response_to_pyautogui_code生成代码
            pyautogui_code = ui_tars_parser.parsing_response_to_pyautogui_code(
                action, image_height, image_width
            )
            
            # 调试信息
            logger.debug(f"Action结构化输出: {action}")
            logger.info(f"PyAutoGUI代码: {pyautogui_code}")
            
            # 步骤4: 从PyAutoGUI代码中提取坐标并转换为框架格式
            parsed_action = ActionParser._convert_pyautogui_to_internal(
                action, pyautogui_code
            )
            
            logger.info(f"UI-TARS官方解析成功: {parsed_action['action_type']}")
            logger.info(f"最终解析结果: {parsed_action}")
            
            # 检查是否需要将相对坐标转换为绝对坐标
            if parsed_action['action_type'] in ['click', 'double_click', 'right_click', 'hover']:
                x = parsed_action['action_params'].get('x', 0)
                y = parsed_action['action_params'].get('y', 0)
                
                # 如果x,y的值在0-1区间，则判定为相对位置
                if 0 <= x <= 1 and 0 <= y <= 1:
                    # 转换为绝对坐标
                    parsed_action['action_params']['x'] = int(x * image_width)
                    parsed_action['action_params']['y'] = int(y * image_height)
                    logger.info(f"转换相对坐标({x:.3f}, {y:.3f})为绝对坐标: {parsed_action['action_params']}")
                else:
                    # 确保坐标是整数
                    parsed_action['action_params']['x'] = int(x)
                    parsed_action['action_params']['y'] = int(y)
            
            elif parsed_action['action_type'] == 'scroll' and 'x' in parsed_action['action_params']:
                x = parsed_action['action_params'].get('x', 0)
                y = parsed_action['action_params'].get('y', 0)
                
                if 0 <= x <= 1 and 0 <= y <= 1:
                    parsed_action['action_params']['x'] = int(x * image_width)
                    parsed_action['action_params']['y'] = int(y * image_height)
                    logger.info(f"转换滚动相对坐标为绝对坐标: {parsed_action['action_params']}")
                else:
                    parsed_action['action_params']['x'] = int(x)
                    parsed_action['action_params']['y'] = int(y)
            
            elif parsed_action['action_type'] == 'drag':
                # 处理拖拽坐标
                start_x = parsed_action['action_params'].get('start_x', 0)
                start_y = parsed_action['action_params'].get('start_y', 0)
                end_x = parsed_action['action_params'].get('end_x', 0)
                end_y = parsed_action['action_params'].get('end_y', 0)
                
                if (0 <= start_x <= 1 and 0 <= start_y <= 1 and 
                    0 <= end_x <= 1 and 0 <= end_y <= 1):
                    parsed_action['action_params']['start_x'] = int(start_x * image_width)
                    parsed_action['action_params']['start_y'] = int(start_y * image_height)
                    parsed_action['action_params']['end_x'] = int(end_x * image_width)
                    parsed_action['action_params']['end_y'] = int(end_y * image_height)
                    logger.info(f"转换拖拽相对坐标为绝对坐标: {parsed_action['action_params']}")
                else:
                    parsed_action['action_params']['start_x'] = int(start_x)
                    parsed_action['action_params']['start_y'] = int(start_y)
                    parsed_action['action_params']['end_x'] = int(end_x)
                    parsed_action['action_params']['end_y'] = int(end_y)
            
            return thought, raw_action, parsed_action
            
        except Exception as e:
            logger.error(f"UI-TARS官方解析失败: {e}")
            # 使用简单备用解析
            return ActionParser._parse_fallback(response, image_height, image_width)
    
    @staticmethod
    def _convert_pyautogui_to_internal(action: Dict, pyautogui_code: str) -> Dict:
        """
        将UI-TARS官方生成的PyAutoGUI代码转换为框架内部格式
        """
        action_type = action.get('action_type', '')
        
        # 处理DONE情况
        if pyautogui_code.strip() == "DONE":
            return {'action_type': 'finished', 'action_params': {}}
        
        if action_type in ["click", "left_single", "left_double", "right_single", "hover"]:
            # 从PyAutoGUI代码中提取点击坐标
            click_match = re.search(r'pyautogui\.click\((\d+(?:\.\d+)?), (\d+(?:\.\d+)?)', pyautogui_code)
            double_click_match = re.search(r'pyautogui\.doubleClick\((\d+(?:\.\d+)?), (\d+(?:\.\d+)?)', pyautogui_code)
            move_match = re.search(r'pyautogui\.moveTo\((\d+(?:\.\d+)?), (\d+(?:\.\d+)?)', pyautogui_code)
            
            match = click_match or double_click_match or move_match
            if match:
                x = float(match.group(1))
                y = float(match.group(2))
                
                # 映射动作类型
                if action_type == "left_double" or "doubleClick" in pyautogui_code:
                    device_action_type = "double_click"
                elif action_type == "right_single" or "button='right'" in pyautogui_code:
                    device_action_type = "right_click"
                elif action_type == "hover" or "moveTo" in pyautogui_code:
                    device_action_type = "hover"
                else:
                    device_action_type = "click"
                    
                return {'action_type': device_action_type, 'action_params': {'x': x, 'y': y}}
        
        elif action_type == "scroll":
            # 从PyAutoGUI代码中提取滚动信息
            scroll_match = re.search(r'pyautogui\.scroll\((-?\d+)', pyautogui_code)
            if scroll_match:
                scroll_value = int(scroll_match.group(1))
                direction = 'up' if scroll_value > 0 else 'down'
                
                # 检查是否有坐标
                coord_match = re.search(r'x=(\d+(?:\.\d+)?), y=(\d+(?:\.\d+)?)', pyautogui_code)
                params = {'direction': direction}
                if coord_match:
                    params['x'] = float(coord_match.group(1))
                    params['y'] = float(coord_match.group(2))
                
                return {'action_type': 'scroll', 'action_params': params}
        
        elif action_type in ["drag", "select"]:
            # 从PyAutoGUI代码中提取拖拽坐标
            move_match = re.search(r'pyautogui\.moveTo\((\d+(?:\.\d+)?), (\d+(?:\.\d+)?)', pyautogui_code)
            drag_match = re.search(r'pyautogui\.dragTo\((\d+(?:\.\d+)?), (\d+(?:\.\d+)?)', pyautogui_code)
            
            if move_match and drag_match:
                start_x = float(move_match.group(1))
                start_y = float(move_match.group(2))
                end_x = float(drag_match.group(1))
                end_y = float(drag_match.group(2))
                
                return {
                    'action_type': 'drag',
                    'action_params': {
                        'start_x': start_x, 'start_y': start_y,
                        'end_x': end_x, 'end_y': end_y
                    }
                }
        
        elif action_type == "type":
            # 从PyAutoGUI代码中提取文本
            write_match = re.search(r"pyautogui\.write\('([^']*)'", pyautogui_code)
            copy_match = re.search(r"pyperclip\.copy\('([^']*)'", pyautogui_code)
            
            if write_match:
                text = write_match.group(1)
            elif copy_match:
                text = copy_match.group(1)
            else:
                text = action.get('action_inputs', {}).get('content', '')
            
            return {'action_type': 'type', 'action_params': {'text': text}}
        
        elif action_type == "hotkey":
            # 处理热键
            hotkey_match = re.search(r"pyautogui\.hotkey\(([^)]+)\)", pyautogui_code)
            keydown_match = re.search(r"pyautogui\.keyDown\(([^)]+)\)", pyautogui_code)
            press_match = re.search(r"pyautogui\.press\(([^)]+)\)", pyautogui_code)
            
            match = hotkey_match or keydown_match or press_match
            if match:
                keys_str = match.group(1).replace("'", "").replace('"', '')
                
                # 映射常见热键到我们的操作
                if 'home' in keys_str.lower():
                    return {'action_type': 'press_home', 'action_params': {}}
                elif 'back' in keys_str.lower() or 'escape' in keys_str.lower():
                    return {'action_type': 'press_back', 'action_params': {}}
                else:
                    return {'action_type': 'hotkey', 'action_params': {'keys': keys_str}}
            
            # 如果没找到匹配，使用原始输入
            hotkey_input = action.get('action_inputs', {})
            keys = hotkey_input.get('key', '') or hotkey_input.get('hotkey', '')
            return {'action_type': 'hotkey', 'action_params': {'keys': keys}}
        
        elif action_type == "finished":
            return {'action_type': 'finished', 'action_params': {}}
        
        # 默认返回原始信息
        logger.warning(f"未处理的动作类型: {action_type}, PyAutoGUI代码: {pyautogui_code}")
        return {'action_type': action_type, 'action_params': action.get('action_inputs', {})}
    
    @staticmethod
    def _extract_raw_action(response: str) -> str:
        """从响应中提取原始动作字符串"""
        try:
            if "Action:" in response:
                return response.split("Action:")[-1].strip().split('\n')[0]
            return response.strip()
        except:
            return response.strip()
    
    @staticmethod
    def _parse_fallback(response: str, image_height: int, image_width: int) -> Tuple[str, str, Dict]:
        """简单备用解析方法"""
        try:
            # 提取Thought和Action
            thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|\Z)", response, re.DOTALL)
            action_match = re.search(r"Action:\s*(.*?)(?=\n\n|\Z)", response, re.DOTALL)
            
            thought = thought_match.group(1).strip() if thought_match else ""
            action_str = action_match.group(1).strip() if action_match else ""
            
            if not action_str:
                raise ValueError("未找到有效的Action")
            
            # 简单解析
            if 'click(' in action_str:
                coord_match = re.search(r"[\(,\s](\d+(?:\.\d+)?)[,\s]+(\d+(?:\.\d+)?)[\),\s]", action_str)
                if coord_match:
                    x, y = float(coord_match.group(1)), float(coord_match.group(2))
                    
                    # 简单的坐标标准化
                    if 0 <= x <= 1 and 0 <= y <= 1:
                        x *= image_width
                        y *= image_height
                    
                    return thought, action_str, {
                        'action_type': 'click',
                        'action_params': {'x': int(x), 'y': int(y)}
                    }
            
            # 其他简单情况
            if 'scroll(' in action_str:
                direction_match = re.search(r'direction[=:]\s*["\']?(\w+)["\']?', action_str)
                direction = direction_match.group(1) if direction_match else 'down'
                return thought, action_str, {
                    'action_type': 'scroll',
                    'action_params': {'direction': direction}
                }
            
            raise ValueError(f"备用解析无法处理: {action_str}")
            
        except Exception as e:
            logger.error(f"备用解析失败: {e}")
            raise
