#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UI-TARS 自动化执行框架 - 主框架模块
"""

import base64
import time
import logging
from typing import List, Dict, Any
from openai import OpenAI
import uiautomator2 as u2
from PIL import Image

from .config import ExecutionConfig, ActionResult, APP_PACKAGES, MOBILE_PROMPT_TEMPLATE
from .action_parser import ActionParser
from .data_manager import DataManager
from .coordinate_processor import CoordinateProcessor

logger = logging.getLogger(__name__)

class UITarsAutomationFramework:
    """UI-TARS自动化框架"""
    
    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.device = None
        self.client = None
        self.action_history = []
        self.step_count = 0
        self.task_description = ""
        self.data_manager = None
        
        self._initialize_client()
        self._initialize_device()
    
    def _initialize_client(self):
        """初始化OpenAI客户端"""
        try:
            self.client = OpenAI(
                base_url=self.config.model_base_url,
                api_key="EMPTY"
            )
            logger.info(f"已连接到模型服务: {self.config.model_base_url}")
        except Exception as e:
            logger.error(f"模型客户端初始化失败: {e}")
            raise
    
    def _initialize_device(self):
        """初始化设备连接"""
        try:
            if self.config.device_ip:
                self.device = u2.connect(self.config.device_ip)
            else:
                self.device = u2.connect()
            
            # 获取设备信息
            device_info = self.device.info
            logger.info(f"已连接设备: {device_info.get('productName', 'Unknown')} "
                       f"- {device_info.get('version', 'Unknown')}")
        except Exception as e:
            logger.error(f"设备连接失败: {e}")
            raise
    
    def _capture_screenshot_and_data(self, step_number: int) -> str:
        """截图并保存相关数据"""
        try:
            # 保存截图和XML
            screenshot_path = self.data_manager.save_screenshot(self.device, step_number)
            xml_path = self.data_manager.save_xml(self.device, step_number)
            
            # 读取截图并转换为base64
            if screenshot_path:
                with open(screenshot_path, "rb") as image_file:
                    image_data = base64.b64encode(image_file.read()).decode('utf-8')
                image_data_url = f"data:image/jpeg;base64,{image_data}"
            else:
                # 临时截图用于模型调用
                temp_path = f"temp_screenshot_{step_number}.jpg"
                self.device.screenshot(temp_path)
                with open(temp_path, "rb") as image_file:
                    image_data = base64.b64encode(image_file.read()).decode('utf-8')
                image_data_url = f"data:image/jpeg;base64,{image_data}"
                import os
                os.remove(temp_path)  # 删除临时文件
            
            logger.info(f"步骤 {step_number} 数据已保存")
            return image_data_url, screenshot_path, xml_path
            
        except Exception as e:
            logger.error(f"截图和数据保存失败: {e}")
            raise
    
    def _build_messages(self, image_data: str) -> List[Dict]:
        """构建发送给模型的消息"""
        # 构建系统提示
        system_prompt = MOBILE_PROMPT_TEMPLATE.format(
            language=self.config.language,
            instruction=self.task_description
        )
        
        messages = [
            {
                "role": "user", 
                "content": system_prompt
            }
        ]
        
        # 添加历史操作记录
        for action in self.action_history:
            if action.get('thought') and action.get('raw_action'):
                messages.append({
                    "role": "assistant",
                    "content": f"Thought: {action['thought']}\nAction: {action['raw_action']}"
                })
        
        # 添加当前截图
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_data}
                }
            ]
        })
        
        return messages
    
    def _call_model(self, messages: List[Dict]) -> str:
        """调用模型获取响应"""
        try:
            logger.info("正在调用模型...")
            chat_completion = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=True
            )
            
            response = ""
            for chunk in chat_completion:
                if chunk.choices[0].delta.content:
                    response += chunk.choices[0].delta.content
            
            logger.info(f"模型响应: {response}")
            return response
        except Exception as e:
            logger.error(f"模型调用失败: {e}")
            raise
    
    def _execute_action(self, action: Dict, screenshot_path: str = None) -> ActionResult:
        """执行具体的操作"""
        try:
            action_type = action['action_type']
            params = action['action_params']
            
            logger.info(f"执行操作: {action_type} - {params}")
            
            if action_type == 'click':
                x, y = params['x'], params['y']
                
                # 创建可视化图像（如果有截图路径）
                if screenshot_path:
                    try:
                        vis_path = screenshot_path.replace('.jpg', '_visualization.jpg')
                        CoordinateProcessor.create_visualization_image(
                            screenshot_path, x, y, vis_path
                        )
                        logger.info(f"坐标可视化已保存: {vis_path}")
                    except Exception as e:
                        logger.warning(f"坐标可视化失败: {e}")
                
                # 执行点击操作（坐标已经是绝对坐标）
                logger.info(f"设备点击坐标: ({x}, {y})")
                
                # 确保坐标为整数
                x, y = round(x), round(y)
                
                # 执行点击
                self.device.click(x, y)
                
                # 等待操作完成
                time.sleep(0.5)
                
                return ActionResult(True, f"点击 ({x}, {y})")
            
            elif action_type == 'long_press':
                x, y = params['x'], params['y']
                
                # 创建可视化图像（如果有截图路径）
                if screenshot_path:
                    try:
                        vis_path = screenshot_path.replace('.jpg', '_longpress_visualization.jpg')
                        CoordinateProcessor.create_visualization_image(
                            screenshot_path, x, y, vis_path
                        )
                        logger.info(f"长按坐标可视化已保存: {vis_path}")
                    except Exception as e:
                        logger.warning(f"坐标可视化失败: {e}")
                
                logger.info(f"设备长按坐标: ({x}, {y})")
                x, y = round(x), round(y)
                self.device.long_click(x, y)
                time.sleep(0.5)
                
                return ActionResult(True, f"长按 ({x}, {y})")
            
            elif action_type == 'type':
                # 使用ADB键盘进行输入
                text = params['text']
                logger.info(f"输入文本: {text}")
                
                try:
                    # 获取当前输入法
                    current_ime = self.device.current_ime()
                    
                    # 切换到ADB键盘
                    self.device.shell(['settings', 'put', 'secure', 'default_input_method', 
                                     'com.android.adbkeyboard/.AdbIME'])
                    time.sleep(0.5)
                    
                    # 发送文本
                    charsb64 = base64.b64encode(text.encode('utf-8')).decode('utf-8')
                    self.device.shell(['am', 'broadcast', '-a', 'ADB_INPUT_B64', '--es', 'msg', charsb64])
                    time.sleep(0.5)
                    
                    # 恢复原输入法
                    self.device.shell(['settings', 'put', 'secure', 'default_input_method', current_ime])
                    time.sleep(0.5)
                    
                    return ActionResult(True, f"输入文本: {text}")
                    
                except Exception as e:
                    logger.error(f"文本输入失败: {e}")
                    return ActionResult(False, f"文本输入失败: {e}")
            
            elif action_type == 'scroll':
                direction = params['direction'].lower()
                
                # 获取坐标，如果没有提供则使用屏幕中心
                if 'x' in params and 'y' in params:
                    x, y = params['x'], params['y']
                else:
                    # 使用设备屏幕中心作为滚动起点
                    device_info = self.device.info
                    x = device_info['displayWidth'] // 2
                    y = device_info['displayHeight'] // 2
                
                # 坐标转换（仅当有原始坐标时）
                if screenshot_path and 'x' in params and 'y' in params:
                    try:
                        img = Image.open(screenshot_path)
                        width, height = img.size
                        actual_x, actual_y = CoordinateProcessor.convert_model_coords_to_actual(
                            params['x'], params['y'], width, height
                        )
                        x, y = actual_x, actual_y
                    except Exception as e:
                        logger.warning(f"坐标转换失败，使用原始坐标: {e}")
                
                logger.info(f"滚动操作: {direction} at ({x}, {y})")
                x, y = round(x), round(y)
                
                # 参考滚动实现，使用较短的duration
                if direction == 'down':
                    self.device.swipe(x, y, x, y - 300, duration=0.1)
                elif direction == 'up':
                    self.device.swipe(x, y, x, y + 300, duration=0.1)
                elif direction == 'left':
                    self.device.swipe(x, y, x + 300, y, duration=0.1)
                elif direction == 'right':
                    self.device.swipe(x, y, x - 300, y, duration=0.1)
                
                time.sleep(0.5)
                return ActionResult(True, f"滚动 {direction}")
            
            elif action_type == 'drag':
                start_x, start_y = params['start_x'], params['start_y']
                end_x, end_y = params['end_x'], params['end_y']
                
                # 坐标转换
                if screenshot_path:
                    try:
                        img = Image.open(screenshot_path)
                        width, height = img.size
                        actual_start_x, actual_start_y = CoordinateProcessor.convert_model_coords_to_actual(
                            start_x, start_y, width, height
                        )
                        actual_end_x, actual_end_y = CoordinateProcessor.convert_model_coords_to_actual(
                            end_x, end_y, width, height
                        )
                        start_x, start_y = actual_start_x, actual_start_y
                        end_x, end_y = actual_end_x, actual_end_y
                    except Exception as e:
                        logger.warning(f"坐标转换失败，使用原始坐标: {e}")
                
                logger.info(f"拖拽操作: ({start_x}, {start_y}) -> ({end_x}, {end_y})")
                start_x, start_y = round(start_x), round(start_y)
                end_x, end_y = round(end_x), round(end_y)
                
                # 参考拖拽实现
                self.device.swipe(start_x, start_y, end_x, end_y, duration=0.1)
                time.sleep(0.5)
                
                return ActionResult(True, f"拖拽 ({start_x}, {start_y}) → ({end_x}, {end_y})")
            
            elif action_type == 'press_home':
                logger.info("按下Home键")
                self.device.press("home")
                time.sleep(0.5)
                return ActionResult(True, "按下Home键")
            
            elif action_type == 'press_back':
                logger.info("按下返回键")
                self.device.press("back") 
                time.sleep(0.5)
                return ActionResult(True, "按下返回键")
            
            elif action_type == 'open_app':
                app_name = params.get('app_name', '')
                logger.info(f"尝试打开应用: {app_name}")
                
                # 从映射表中获取包名
                package_name = APP_PACKAGES.get(app_name)
                if package_name:
                    try:
                        # 使用device.app_start启动应用
                        self.device.app_start(package_name, stop=True)
                        time.sleep(2.0)  # 等待应用启动
                        logger.info(f"成功启动应用: {app_name} ({package_name})")
                        return ActionResult(True, f"已打开应用: {app_name}")
                    except Exception as e:
                        logger.error(f"启动应用失败: {e}")
                        return ActionResult(False, f"启动应用失败: {app_name} - {e}")
                else:
                    # 如果没有找到包名，尝试通过图标点击
                    logger.warning(f"未找到应用包名: {app_name}，请手动点击应用图标")
                    return ActionResult(False, f"未找到应用包名: {app_name}，请使用click操作点击应用图标")
            
            elif action_type == 'finished':
                return ActionResult(True, "任务完成", error="FINISHED")

            elif action_type == 'wait':
                # 等待指定秒数
                seconds = params.get('seconds', 1)
                try:
                    seconds = float(seconds)
                except Exception:
                    seconds = 1
                logger.info(f"等待 {seconds} 秒")
                time.sleep(seconds)
                return ActionResult(True, f"等待 {seconds} 秒")
            else:
                return ActionResult(False, f"不支持的操作类型: {action_type}")
                
        except Exception as e:
            logger.error(f"操作执行失败: {e}")
            return ActionResult(False, f"操作执行失败: {e}", error=str(e))
    
    def execute_task(self, task_description: str) -> bool:
        """执行完整的任务"""
        self.task_description = task_description
        self.step_count = 0
        self.action_history = []
        
        # 初始化数据管理器
        self.data_manager = DataManager(self.config, task_description)
        
        logger.info(f"开始执行任务: {task_description}")
        
        try:
            while self.step_count < self.config.max_steps:
                self.step_count += 1
                logger.info(f"\n{'='*50}")
                logger.info(f"第 {self.step_count} 步")
                logger.info(f"{'='*50}")
                
                # 开始新步骤
                self.data_manager.start_new_step(self.step_count)
                
                # 1. 截图并保存数据
                image_data, screenshot_path, xml_path = self._capture_screenshot_and_data(self.step_count)
                
                # 2. 构建消息
                messages = self._build_messages(image_data)
                
                # 3. 调用模型
                response = self._call_model(messages)
                
                # 4. 解析响应（传递图片尺寸信息用于坐标转换）
                image_height, image_width = None, None
                if screenshot_path:
                    try:
                        from PIL import Image
                        with Image.open(screenshot_path) as img:
                            image_width, image_height = img.size
                    except Exception as e:
                        logger.warning(f"无法获取图片尺寸: {e}")
                
                thought, raw_action, parsed_action = ActionParser.parse_response(
                    response, image_height, image_width
                )
                
                logger.info(f"思考: {thought}")
                logger.info(f"操作: {raw_action}")
                
                # 5. 执行操作
                result = self._execute_action(parsed_action, screenshot_path)
                result.screenshot_path = screenshot_path
                result.xml_path = xml_path
                
                # 6. 记录历史
                action_record = {
                    'step': self.step_count,
                    'thought': thought,
                    'raw_action': raw_action,
                    'parsed_action': parsed_action,
                    'result': result,
                    'screenshot_path': screenshot_path,
                    'xml_path': xml_path
                }
                self.action_history.append(action_record)
                
                # 7. 保存步骤数据
                step_data = {
                    'step': self.step_count,
                    'thought': thought,
                    'raw_action': raw_action,
                    'parsed_action': parsed_action,
                    'result': {
                        'success': result.success,
                        'message': result.message,
                        'error': result.error
                    },
                    'screenshot_path': screenshot_path,
                    'xml_path': xml_path,
                    'timestamp': time.time()
                }
                self.data_manager.save_step_data(self.step_count, step_data)
                
                # 8. 检查是否完成
                if not result.success:
                    logger.error(f"操作失败: {result.message}")
                    break
                
                if result.error == "FINISHED":
                    logger.info("任务执行完成!")
                    break
                
                # 9. 等待
                time.sleep(self.config.step_delay)
            
            # 保存执行总结
            execution_summary = self.get_execution_summary()
            self.data_manager.save_execution_summary(execution_summary)
            
            success = execution_summary.get('success', False)
            if not success and self.step_count >= self.config.max_steps:
                logger.warning(f"达到最大步数限制 ({self.config.max_steps})")
            
            return success
            
        except Exception as e:
            logger.error(f"任务执行失败: {e}")
            return False
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """获取执行摘要"""
        return {
            'task_description': self.task_description,
            'total_steps': self.step_count,
            'action_history': self.action_history,
            'success': len(self.action_history) > 0 and self.action_history[-1]['result'].error == "FINISHED",
            'task_directory': self.data_manager.get_task_directory() if self.data_manager else None
        }
