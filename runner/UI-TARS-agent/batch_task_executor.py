#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量任务执行器 - 读取auto_task-3.json中的任务并自动执行
"""

import json
import os
import time
import logging
import traceback
import shutil
import base64
import re
from datetime import datetime
from openai import OpenAI
from ui_tars_automation import UITarsAutomationFramework, ExecutionConfig
from ui_tars_automation.action_parser import ActionParser

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_execution.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class BatchTaskExecutor:
    """批量任务执行器"""
    
    def __init__(self, model_url: str, data_base_dir: str = "data_uitars-test"):
        """初始化批量执行器
        
        Args:
            model_url: 模型服务地址
            data_base_dir: 数据保存基目录
        """
        self.model_url = model_url
        self.data_base_dir = data_base_dir
        
        # 初始化模型客户端用于智能应用选择（使用专门的模型）
        try:
            self.app_selection_client = OpenAI(
                api_key="sk-rfCIGhxrzcdsMV4jC17e406bE56c47CbA5416068A62318D3",
                base_url="http://ipads.chat.gpt:3006/v1"
            )
            self.app_selection_model = "gemini-2.5-pro-preview-06-05"
            logger.info(f"已连接到应用选择模型服务: http://ipads.chat.gpt:3006/v1")
        except Exception as e:
            logger.warning(f"应用选择模型客户端初始化失败，将只使用预定义包名映射: {e}")
            self.app_selection_client = None
            self.app_selection_model = None
        
        # 应用包名映射（从open_app.py参考）
        self.app_packages = {
            "微信": "com.tencent.mm",
            "QQ": "com.tencent.mobileqq",
            "新浪微博": "com.sina.weibo",
            "饿了么": "me.ele",
            "美团": "com.sankuai.meituan",
            "bilibili": "tv.danmaku.bili",
            "爱奇艺": "com.qiyi.video",
            "腾讯视频": "com.tencent.qqlive",
            "优酷": "com.youku.phone",
            "淘宝": "com.taobao.taobao",
            "京东": "com.jingdong.app.mall",
            "携程": "ctrip.android.view",
            "同城": "com.tongcheng.android",
            "飞猪": "com.taobao.trip",
            "去哪儿": "com.Qunar",
            "华住会": "com.htinns",
            "知乎": "com.zhihu.android",
            "小红书": "com.xingin.xhs",
            "QQ音乐": "com.tencent.qqmusic",
            "网易云音乐": "com.netease.cloudmusic",
            "酷狗音乐": "com.kugou.android",
            "高德": "com.autonavi.minimap"
        }
    
    def parse_json_response(self, response_str: str) -> dict:
        """解析JSON响应
        
        Args:
            response_str: 模型返回的响应字符串
            
        Returns:
            解析后的JSON对象
        """
        try:
            # 尝试直接解析JSON
            return json.loads(response_str)
        except json.JSONDecodeError:
            # 如果直接解析失败，尝试提取JSON部分
            try:
                # 查找JSON代码块
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_str, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1))
                
                # 查找花括号包围的JSON
                json_match = re.search(r'(\{.*?\})', response_str, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1))
                
                raise ValueError("无法在响应中找到有效的JSON")
            except Exception as e:
                logger.error(f"JSON解析失败: {e}")
                logger.error(f"原始响应: {response_str}")
                raise ValueError(f"无法解析JSON响应: {e}")
    

    def get_app_package_name_by_ai(self, task_description: str) -> str:
        """使用AI根据任务描述获取应用包名
        
        Args:
            task_description: 任务描述
            
        Returns:
            应用包名
        """
        if not self.app_selection_client:
            logger.error("应用选择模型客户端未初始化")
            return None
        
        # 从open_app.py复制的提示模板
        app_selection_prompt_template = """
## 角色定义
你是一个智能手机应用选择助手，需要根据用户的任务描述选择最合适的应用。

## 任务描述
用户想要完成的任务是："{task_description}"

## 可用应用列表
以下是可用的应用及其包名：
- 微信: com.tencent.mm
- QQ: com.tencent.mobileqq
- 新浪微博: com.sina.weibo
- 饿了么: me.ele
- 美团: com.sankuai.meituan
- bilibili: tv.danmaku.bili
- 爱奇艺: com.qiyi.video
- 腾讯视频: com.tencent.qqlive
- 优酷: com.youku.phone
- 淘宝: com.taobao.taobao
- 京东: com.jingdong.app.mall
- 携程: ctrip.android.view
- 同城: com.tongcheng.android
- 飞猪: com.taobao.trip
- 去哪儿: com.Qunar
- 华住会: com.htinns
- 知乎: com.zhihu.android
- 小红书: com.xingin.xhs
- QQ音乐: com.tencent.qqmusic
- 网易云音乐: com.netease.cloudmusic
- 酷狗音乐: com.kugou.android
- 高德: com.autonavi.minimap

## 任务要求
请分析任务描述，选择最合适的应用来完成该任务。

## 输出格式
请严格按照以下JSON格式输出：
```json
{{
    "reasoning": "分析任务内容，说明为什么选择这个应用最合适",
    "package_name": "选择的应用包名"
}}
```

## 重要规则
1. 只能从上述可用应用列表中选择
2. 必须选择最符合任务需求的应用
3. 如果任务涉及多个可能的应用，选择最主要和最常用的那个
4. 包名必须完全匹配列表中的包名，不能修改
"""
        
        app_selection_prompt = app_selection_prompt_template.format(task_description=task_description)
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response_str = self.app_selection_client.chat.completions.create(
                    model=self.app_selection_model,
                    messages=[
                        {
                            "role": "user",
                            "content": app_selection_prompt
                        }
                    ]
                ).choices[0].message.content
                
                logger.info(f"应用选择响应 (尝试 {attempt + 1}): \n{response_str}")
                
                # 解析响应
                response = self.parse_json_response(response_str)
                package_name = response.get("package_name")
                reasoning = response.get("reasoning")
                
                if package_name:
                    logger.info(f"AI选择应用原因: {reasoning}")
                    logger.info(f"AI选择的包名: {package_name}")
                    return package_name
                else:
                    logger.warning(f"AI响应中没有包名信息: {response}")
                    
            except Exception as e:
                logger.error(f"AI应用选择失败 (尝试 {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    logger.error("AI应用选择最终失败")
                    
        return None
    

    def load_tasks(self, task_file: str) -> list:
        """加载任务文件
        
        Args:
            task_file: 任务文件路径
            
        Returns:
            任务列表
        """
        try:
            with open(task_file, 'r', encoding='utf-8') as f:
                tasks = json.load(f)
            logger.info(f"成功加载任务文件: {task_file}")
            return tasks
        except Exception as e:
            logger.error(f"加载任务文件失败: {e}")
            raise
    
    def get_package_name(self, app_name: str, task_description: str = None) -> str:
        """根据应用名称获取包名，如果失败则使用AI分析任务描述
        
        Args:
            app_name: 应用名称
            task_description: 任务描述（可选，用于AI分析）
            
        Returns:
            应用包名
        """
        # 首先尝试从预定义映射中查找
        package_name = self.app_packages.get(app_name)
        
        if package_name:
            logger.info(f"从预定义映射找到应用包名: {app_name} -> {package_name}")
            return package_name
        
        # 如果预定义映射中没有找到，且提供了任务描述，则使用AI分析
        if task_description and self.app_selection_client:
            logger.info(f"预定义映射中未找到应用 '{app_name}'，尝试使用AI分析任务描述")
            ai_package_name = self.get_app_package_name_by_ai(task_description)
            
            if ai_package_name:
                logger.info(f"AI成功识别应用包名: {ai_package_name}")
                # 将AI识别的结果添加到缓存中，避免重复查询
                self.app_packages[app_name] = ai_package_name
                return ai_package_name
            else:
                logger.warning(f"AI也无法识别应用 '{app_name}' 的包名")
        
        # 所有方法都失败了
        logger.error(f"无法获取应用 '{app_name}' 的包名")
        return None
    
    def create_task_directory(self, app_name: str, task_type: str, task_index: int) -> str:
        """创建任务目录
        
        Args:
            app_name: 应用名称
            task_type: 任务类型
            task_index: 任务索引
            
        Returns:
            任务目录路径
        """
        task_dir = os.path.join(self.data_base_dir, app_name, task_type, str(task_index))
        os.makedirs(task_dir, exist_ok=True)
        return task_dir
    
    def save_task_data(self, task_dir: str, task_info: dict, execution_result: dict):
        """保存任务数据到task_data.json和actions.json
        
        Args:
            task_dir: 任务目录
            task_info: 任务信息
            execution_result: 执行结果
        """
        try:
            # 构建task_data.json（保持原有格式）
            task_data = {
                "task_description": task_info["task_description"],
                "app_name": task_info["app_name"],
                "task_type": task_info["task_type"],
                "task_index": task_info["task_index"],
                "package_name": task_info["package_name"],
                "execution_time": datetime.now().isoformat(),
                "action_count": execution_result.get("total_steps", 0),
                "actions": [],
                "success": execution_result.get("success", False)
            }
            
            # 构建actions.json（参考淘宝格式）
            actions_data = {
                "app_name": task_info["app_name"],
                "task_type": task_info["task_type"],
                "task_description": task_info["task_description"],
                "action_count": execution_result.get("total_steps", 0),
                "actions": []
            }
            
            # 转换action历史为标准格式
            for i, action in enumerate(execution_result.get("action_history", []), 1):
                # task_data.json格式（保持原有格式）
                action_record = {
                    "reasoning": action["thought"],
                    "function": {
                        "name": action["parsed_action"]["action_type"],
                        "parameters": action["parsed_action"]["action_params"]
                    }
                }
                task_data["actions"].append(action_record)
                
                # actions.json格式（参考淘宝格式）
                action_type = action["parsed_action"]["action_type"]
                action_params = action["parsed_action"]["action_params"]
                
                action_item = {
                    "type": action_type,
                    "action_index": i
                }
                
                # 根据不同操作类型添加相应参数
                if action_type == "click":
                    action_item.update({
                        "position_x": action_params.get("x"),
                        "position_y": action_params.get("y")
                    })
                    # 如果有bounding_box信息，添加bounds
                    if "bounding_box" in action_params:
                        action_item["bounds"] = action_params["bounding_box"]
                
                elif action_type == "type":
                    action_item["text"] = action_params.get("text", "")
                
                elif action_type == "scroll":
                    action_item.update({
                        "position_x": action_params.get("x"),
                        "position_y": action_params.get("y"),
                        "direction": action_params.get("direction", "down")
                    })
                
                elif action_type == "swipe" or action_type == "drag":
                    action_item.update({
                        "press_position_x": action_params.get("start_x"),
                        "press_position_y": action_params.get("start_y"),
                        "release_position_x": action_params.get("end_x"),
                        "release_position_y": action_params.get("end_y"),
                        "direction": action_params.get("direction", "")
                    })
                
                elif action_type == "long_press":
                    action_item.update({
                        "position_x": action_params.get("x"),
                        "position_y": action_params.get("y")
                    })
                
                elif action_type in ["finished", "done"]:
                    action_item["type"] = "done"
                
                actions_data["actions"].append(action_item)
            
            # 保存task_data.json
            task_data_path = os.path.join(task_dir, "task_data.json")
            with open(task_data_path, "w", encoding="utf-8") as f:
                json.dump(task_data, f, ensure_ascii=False, indent=4)
            
            # 保存actions.json（参考淘宝格式）
            actions_path = os.path.join(task_dir, "actions.json")
            with open(actions_path, "w", encoding="utf-8") as f:
                json.dump(actions_data, f, ensure_ascii=False, indent=4)

            # 生成 react.json（每步的 reasoning 和 function），并复制每步的截图与 xml 到 task_dir，命名为 1.jpg / 1.xml ...
            reacts = []
            for i, action in enumerate(execution_result.get("action_history", []), 1):
                reasoning = action.get('thought', '')
                parsed = action.get('parsed_action', {}) or {}
                func_name = parsed.get('action_type')
                func_params = parsed.get('action_params', {})

                # 映射内部类型到采集格式（例如 type -> input, finished/done -> done）
                out_func_name = func_name
                if func_name == 'type':
                    out_func_name = 'input'
                elif func_name in ['finished', 'done']:
                    out_func_name = 'done'

                reacts.append({
                    'reasoning': reasoning,
                    'function': {
                        'name': out_func_name,
                        'parameters': func_params
                    },
                    'action_index': i
                })

                # 复制截图和 xml（如果存在）。支持两种位置：
                # 1) action 中记录的绝对/相对路径；
                # 2) task_dir/<step>/screenshot_{i}.jpg 和 task_dir/<step>/hierarchy_{i}.xml（框架增强执行时的保存位置）。
                screenshot_src = action.get('screenshot_path')
                xml_src = action.get('xml_path')
                try:
                    copied = False
                    if screenshot_src and os.path.exists(screenshot_src):
                        dst_img = os.path.join(task_dir, f"{i}.jpg")
                        shutil.copyfile(screenshot_src, dst_img)
                        copied = True
                    else:
                        # 备选位置： task_dir/<i>/screenshot_{i}.jpg
                        alt_img = os.path.join(task_dir, str(i), f"screenshot_{i}.jpg")
                        if os.path.exists(alt_img):
                            dst_img = os.path.join(task_dir, f"{i}.jpg")
                            shutil.copyfile(alt_img, dst_img)
                            copied = True

                    if xml_src and os.path.exists(xml_src):
                        dst_xml = os.path.join(task_dir, f"{i}.xml")
                        shutil.copyfile(xml_src, dst_xml)
                        copied = True
                    else:
                        # 备选位置： task_dir/<i>/hierarchy_{i}.xml
                        alt_xml = os.path.join(task_dir, str(i), f"hierarchy_{i}.xml")
                        if os.path.exists(alt_xml):
                            dst_xml = os.path.join(task_dir, f"{i}.xml")
                            shutil.copyfile(alt_xml, dst_xml)
                            copied = True

                    if not copied:
                        logger.debug(f"未找到步骤 {i} 的截图或 xml，跳过复制")
                except Exception as e:
                    logger.warning(f"复制步骤资源失败 (step {i}): {e}")

            reacts_path = os.path.join(task_dir, "reacts.json")
            with open(reacts_path, 'w', encoding='utf-8') as f:
                json.dump(reacts, f, ensure_ascii=False, indent=4)

            # 如果 reacts 或 action_history 为空，仍然尝试从 step 子目录复制截图和 xml
            try:
                # 查找 task_dir 下的数字子目录，如 1, 2, ...
                step_dirs = []
                for name in os.listdir(task_dir):
                    subp = os.path.join(task_dir, name)
                    if os.path.isdir(subp) and name.isdigit():
                        step_dirs.append(int(name))
                step_dirs.sort()

                for i in step_dirs:
                    dst_img = os.path.join(task_dir, f"{i}.jpg")
                    dst_xml = os.path.join(task_dir, f"{i}.xml")

                    # 仅在根目录中不存在时复制
                    if not os.path.exists(dst_img):
                        alt_img = os.path.join(task_dir, str(i), f"screenshot_{i}.jpg")
                        if os.path.exists(alt_img):
                            try:
                                shutil.copyfile(alt_img, dst_img)
                                logger.info(f"已复制截图到根目录: {dst_img}")
                            except Exception as e:
                                logger.warning(f"复制截图失败 ({alt_img} -> {dst_img}): {e}")

                    if not os.path.exists(dst_xml):
                        alt_xml = os.path.join(task_dir, str(i), f"hierarchy_{i}.xml")
                        if os.path.exists(alt_xml):
                            try:
                                shutil.copyfile(alt_xml, dst_xml)
                                logger.info(f"已复制 xml 到根目录: {dst_xml}")
                            except Exception as e:
                                logger.warning(f"复制 xml 失败 ({alt_xml} -> {dst_xml}): {e}")
            except Exception as e:
                logger.debug(f"扫描 step 子目录并复制资源时出错: {e}")
            
            logger.info(f"任务数据已保存到: {task_dir}")
            logger.info(f"  - task_data.json: {task_data_path}")
            logger.info(f"  - actions.json: {actions_path}")
            
        except Exception as e:
            logger.error(f"保存任务数据失败: {e}")
    
    def execute_single_task(self, app_name: str, task_type: str, task_index: int, task_description: str) -> bool:
        """执行单个任务
        
        Args:
            app_name: 应用名称
            task_type: 任务类型 (type1, type2, ...)
            task_index: 任务索引 (1, 2, 3, ...)
            task_description: 任务描述
            
        Returns:
            是否执行成功
        """
        logger.info(f"开始执行任务: {app_name} - {task_type} - 任务{task_index}")
        logger.info(f"任务描述: {task_description}")
        
        # 获取包名
        package_name = self.get_package_name(app_name, task_description)
        if not package_name:
            logger.error(f"未找到应用 {app_name} 的包名")
            return False
        
        # 创建任务目录 - 直接使用目标格式：app名称/typex/任务序号
        task_dir = self.create_task_directory(app_name, task_type, task_index)
        
        try:
            # 配置 - 禁用框架自动数据保存，我们手动管理
            config = ExecutionConfig(
                model_base_url=self.model_url,
                model_name="UI-TARS-7B-SFT",
                max_steps=20,  # 设置为20步
                step_delay=2.0,
                language="Chinese",
                save_data=False,  # 禁用框架自动保存，我们手动保存到指定位置
                save_screenshots=False,
                save_xml=False
            )
            
            # 创建框架实例
            framework = UITarsAutomationFramework(config)
            
            # 启动应用（参考open_app.py）
            logger.info(f"启动应用: {package_name}")
            framework.device.app_start(package_name, stop=True)
            time.sleep(3)  # 等待应用启动
            
            # 手动保存数据到我们的目录结构
            original_execute = framework.execute_task
            
            def enhanced_execute_task(task_desc):
                """增强的执行方法，手动保存每步数据"""
                framework.task_description = task_desc
                framework.step_count = 0
                framework.action_history = []
                
                logger.info(f"开始执行任务: {task_desc}")
                
                try:
                    while framework.step_count < framework.config.max_steps:
                        framework.step_count += 1
                        logger.info(f"\n{'='*50}")
                        logger.info(f"第 {framework.step_count} 步")
                        logger.info(f"{'='*50}")
                        
                        # 创建步骤目录
                        step_dir = os.path.join(task_dir, str(framework.step_count))
                        os.makedirs(step_dir, exist_ok=True)
                        
                        # 保存截图和XML
                        screenshot_path = os.path.join(step_dir, f"screenshot_{framework.step_count}.jpg")
                        xml_path = os.path.join(step_dir, f"hierarchy_{framework.step_count}.xml")
                        
                        framework.device.screenshot(screenshot_path)
                        hierarchy = framework.device.dump_hierarchy()
                        with open(xml_path, "w", encoding="utf-8") as f:
                            f.write(hierarchy)
                        
                        # 获取图片数据用于模型调用
                        with open(screenshot_path, "rb") as image_file:
                            image_data = base64.b64encode(image_file.read()).decode('utf-8')
                            image_data_url = f"data:image/jpeg;base64,{image_data}"
                        
                        # 构建消息并调用模型
                        messages = framework._build_messages(image_data_url)
                        response = framework._call_model(messages)
                        
                        # 解析响应
                        from PIL import Image
                        
                        image_height, image_width = None, None
                        try:
                            with Image.open(screenshot_path) as img:
                                image_width, image_height = img.size
                        except Exception as e:
                            logger.warning(f"无法获取图片尺寸: {e}")
                        
                        thought, raw_action, parsed_action = ActionParser.parse_response(
                            response, image_height, image_width
                        )
                        
                        logger.info(f"思考: {thought}")
                        logger.info(f"操作: {raw_action}")
                        
                        # 执行操作
                        result = framework._execute_action(parsed_action, screenshot_path)
                        result.screenshot_path = screenshot_path
                        result.xml_path = xml_path
                        
                        # 记录历史
                        action_record = {
                            'step': framework.step_count,
                            'thought': thought,
                            'raw_action': raw_action,
                            'parsed_action': parsed_action,
                            'result': result,
                            'screenshot_path': screenshot_path,
                            'xml_path': xml_path
                        }
                        framework.action_history.append(action_record)
                        
                        # 检查是否完成
                        if not result.success:
                            logger.error(f"操作失败: {result.message}")
                            break
                        
                        if result.error == "FINISHED":
                            logger.info("任务执行完成!")
                            break
                        
                        # 等待
                        time.sleep(framework.config.step_delay)
                    
                    success = len(framework.action_history) > 0 and framework.action_history[-1]['result'].error == "FINISHED"
                    return success
                    
                except Exception as e:
                    logger.error(f"任务执行失败: {e}")
                    return False
            
            # 替换执行方法
            framework.execute_task = enhanced_execute_task
            
            # 执行任务
            success = framework.execute_task(task_description)
            execution_result = framework.get_execution_summary()
            
            # 保存任务数据
            task_info = {
                "task_description": task_description,
                "app_name": app_name,
                "task_type": task_type,
                "task_index": task_index,
                "package_name": package_name
            }
            self.save_task_data(task_dir, task_info, execution_result)
            
            logger.info(f"任务执行完成: {'成功' if success else '失败'}")
            return success
            
        except Exception as e:
            logger.error(f"任务执行异常: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def execute_all_tasks(self, task_file: str):
        """执行所有任务
        
        Args:
            task_file: 任务文件路径
        """
        logger.info("开始批量任务执行")
        
        # 加载任务
        all_tasks = self.load_tasks(task_file)
        
        total_tasks = 0
        successful_tasks = 0
        failed_tasks = 0
        
        # 统计总任务数
        for app_group in all_tasks:
            total_tasks += len(app_group["tasks"])
        
        logger.info(f"总共需要执行 {total_tasks} 个任务")
        
        # 逐个执行任务
        for app_group in all_tasks:
            app_name = app_group["app"]
            task_type = app_group["type"]
            tasks = app_group["tasks"]
            
            logger.info(f"开始执行应用 {app_name} 的 {task_type} 类型任务")
            
            for task_index, task_description in enumerate(tasks, 1):
                try:
                    success = self.execute_single_task(
                        app_name=app_name,
                        task_type=task_type,
                        task_index=task_index,
                        task_description=task_description
                    )
                    
                    if success:
                        successful_tasks += 1
                    else:
                        failed_tasks += 1
                    
                    # 任务间间隔
                    time.sleep(5)
                    
                except Exception as e:
                    logger.error(f"任务执行出错: {e}")
                    failed_tasks += 1
        
        # 执行总结
        logger.info("="*60)
        logger.info("批量任务执行完成")
        logger.info(f"总任务数: {total_tasks}")
        logger.info(f"成功: {successful_tasks}")
        logger.info(f"失败: {failed_tasks}")
        logger.info(f"成功率: {successful_tasks/total_tasks*100:.1f}%" if total_tasks > 0 else "N/A")
        logger.info("="*60)

def main():
    """主函数"""
    print("UI-TARS 批量任务执行器")
    print("=" * 50)
    
    # 获取模型服务地址
    model_url = input("请输入模型服务地址 (默认: http://192.168.12.152:8001/v1): ").strip()
    if not model_url:
        model_url = "http://192.168.12.152:8001/v1"
    
    # 任务文件路径
    task_file = "auto_task-3-sup.json"
    if not os.path.exists(task_file):
        print(f"错误: 任务文件 {task_file} 不存在")
        return
    
    # 创建批量执行器
    executor = BatchTaskExecutor(model_url)
    
    # 询问是否继续
    print(f"将读取 {task_file} 并批量执行任务")
    confirm = input("是否继续？(y/N): ").strip().lower()
    if confirm != 'y':
        print("已取消执行")
        return
    
    # 执行所有任务
    try:
        executor.execute_all_tasks(task_file)
    except KeyboardInterrupt:
        print("\n用户中断执行")
    except Exception as e:
        print(f"执行过程中出现错误: {e}")

if __name__ == "__main__":
    main()
