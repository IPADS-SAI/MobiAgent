#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MobiAgent Mobile Standalone Version
手机上独立可运行版本 - 最小化依赖
"""

from openai import OpenAI
import base64
from PIL import Image
import json
import io
import logging
import time
import re
import os
import argparse
import sys
from pathlib import Path

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger(__name__)

MAX_STEPS = 35
SCREENSHOT_FACTOR = 0.5  # 截图压缩比例

# 全局客户端
decider_client = None
grounder_client = None
planner_client = None


def init_clients(service_ip, decider_port, grounder_port, planner_port):
    """初始化LLM客户端"""
    global decider_client, grounder_client, planner_client
    
    try:
        decider_client = OpenAI(
            api_key="0",
            base_url=f"http://{service_ip}:{decider_port}/v1"
        )
        grounder_client = OpenAI(
            api_key="0",
            base_url=f"http://{service_ip}:{grounder_port}/v1"
        )
        planner_client = OpenAI(
            api_key="0",
            base_url=f"http://{service_ip}:{planner_port}/v1"
        )
        logger.info(f"[OK] 已连接到服务: {service_ip}")
    except Exception as e:
        logger.error(f"[ERROR] 客户端初始化失败: {e}")
        raise


class AndroidDevice:
    """Android设备控制（精简版）"""
    
    def __init__(self, adb_endpoint=None):
        """初始化Android设备连接"""
        try:
            import uiautomator2 as u2
            if adb_endpoint:
                self.d = u2.connect(adb_endpoint)
            else:
                self.d = u2.connect()
            logger.info("[OK] 已连接到Android设备")
        except Exception as e:
            logger.error(f"[ERROR] 无法连接到Android设备: {e}")
            raise
        
        self.app_package_names = {
            "携程": "ctrip.android.view",
            "支付宝": "com.eg.android.AlipayGphone",
            "微信": "com.tencent.mm",
            "淘宝": "com.taobao.taobao",
            "京东": "com.jingdong.app.mall",
            "美团": "com.sankuai.meituan",
            "滴滴": "com.sdu.didi.psnger",
            "微博": "com.sina.weibo",
        }

    def start_app(self, app_name):
        """启动应用"""
        package_name = self.app_package_names.get(app_name, app_name)
        try:
            self.d.app_start(package_name, stop=True)
            time.sleep(1)
            logger.info(f"[OK] 已启动应用: {app_name}")
        except Exception as e:
            logger.error(f"[ERROR] 启动应用失败: {e}")
            raise

    def stop_app(self, app_name):
        """停止应用"""
        package_name = self.app_package_names.get(app_name, app_name)
        try:
            self.d.app_stop(package_name)
            logger.info(f"[OK] 已停止应用: {app_name}")
        except Exception as e:
            logger.error(f"[WARNING] 停止应用失败: {e}")

    def screenshot(self, path=None):
        """获取截图"""
        try:
            if path is None:
                path = "screenshot.jpg"
            self.d.screenshot(path)
            return path
        except Exception as e:
            logger.error(f"[ERROR] 截图失败: {e}")
            raise

    def click(self, x, y):
        """点击屏幕"""
        try:
            self.d.click(x, y)
            time.sleep(0.5)
            logger.info(f"[OK] 已点击: ({x}, {y})")
        except Exception as e:
            logger.error(f"[ERROR] 点击失败: {e}")

    def input_text(self, text):
        """输入文本 - 多重方法确保输入成功"""
        try:
            logger.info(f"[INPUT] 开始输入文本: {text}")
            
            # 方法1：尝试 ADB IME Base64 输入（最可靠）
            try:
                logger.info(f"[INPUT-M1] 尝试 ADB IME Base64 输入")
                current_ime = self.d.current_ime()
                logger.info(f"[INPUT] 当前输入法: {current_ime}")
                
                # 切换到ADB IME
                self.d.shell(['ime', 'set', 'com.android.adbkeyboard/.AdbIME'])
                time.sleep(0.5)
                
                # 使用Base64编码输入（支持中文）
                charsb64 = base64.b64encode(text.encode('utf-8')).decode('utf-8')
                logger.info(f"[INPUT] Base64编码: {charsb64[:50]}...")
                
                self.d.shell(['am', 'broadcast', '-a', 'ADB_INPUT_B64', 
                             '--es', 'msg', charsb64])
                time.sleep(1)  # 增加等待时间
                
                # 恢复原输入法
                self.d.shell(['ime', 'set', current_ime])
                time.sleep(0.3)
                
                logger.info(f"[OK] 已输入文本(ADB IME): {text}")
                return True
                
            except Exception as e:
                logger.warning(f"[INPUT-M1] ADB IME 失败: {e}")
                time.sleep(0.5)
            
            # 方法2：使用 uiautomator2 的 send_keys
            try:
                logger.info(f"[INPUT-M2] 尝试 uiautomator2 send_keys")
                self.d.send_keys(text)
                time.sleep(0.5)
                logger.info(f"[OK] 已输入文本(send_keys): {text}")
                return True
            except Exception as e:
                logger.warning(f"[INPUT-M2] send_keys 失败: {e}")
                time.sleep(0.5)
            
            # 方法3：使用 input text 命令（仅支持英文）
            try:
                logger.info(f"[INPUT-M3] 尝试 input text 命令")
                # 使用双引号包装以支持空格
                self.d.shell(['input', 'text', text])
                time.sleep(0.5)
                logger.info(f"[OK] 已输入文本(input text): {text}")
                return True
            except Exception as e:
                logger.warning(f"[INPUT-M3] input text 失败: {e}")
            
            # 方法4：字符一个一个输入
            try:
                logger.info(f"[INPUT-M4] 尝试逐字符输入")
                for char in text:
                    encoded = base64.b64encode(char.encode('utf-8')).decode('utf-8')
                    self.d.shell(['am', 'broadcast', '-a', 'ADB_INPUT_B64', 
                                 '--es', 'msg', encoded])
                    time.sleep(0.1)
                time.sleep(0.5)
                logger.info(f"[OK] 已逐字符输入文本: {text}")
                return True
            except Exception as e:
                logger.warning(f"[INPUT-M4] 逐字符输入失败: {e}")
            
            # 如果所有方法都失败
            logger.error(f"[ERROR] 所有输入方法都失败!")
            return False
            
        except Exception as e:
            logger.error(f"[ERROR] 输入失败: {e}")
            return False

    def swipe(self, direction):
        """滑动屏幕"""
        try:
            self.d.swipe_ext(direction=direction, scale=0.6)
            logger.info(f"[OK] 已滑动: {direction}")
        except Exception as e:
            logger.error(f"[ERROR] 滑动失败: {e}")

    def dump_hierarchy(self):
        """获取UI层级树"""
        try:
            return self.d.dump_hierarchy()
        except Exception as e:
            logger.error(f"[ERROR] 获取UI树失败: {e}")
            return None


def get_screenshot_b64(device):
    """获取截图并转换为Base64"""
    try:
        path = device.screenshot()
        img = Image.open(path)
        # 压缩截图
        img = img.resize((int(img.width * SCREENSHOT_FACTOR), 
                         int(img.height * SCREENSHOT_FACTOR)), 
                        Image.Resampling.LANCZOS)
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        logger.error(f"[ERROR] Base64转换失败: {e}")
        return None


def parse_json_response(response_str):
    """解析JSON响应（带降级处理）"""
    try:
        # 第1级：直接解析
        return json.loads(response_str)
    except json.JSONDecodeError:
        try:
            # 第2级：提取```json...```代码块
            match = re.search(r'```json\s*(\{.*?\})\s*```', 
                            response_str, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            
            # 第3级：提取任意{}
            match = re.search(r'(\{.*?\})', response_str, re.DOTALL)
            if match:
                return json.loads(match.group(1))
        except:
            pass
    
    logger.warning(f"[WARNING] 无法解析JSON: {response_str[:100]}")
    return None


def get_decider_action(task, history_text, screenshot_b64, model=""):
    """调用Decider模型获取动作"""
    try:
        prompt = f"""根据当前任务和UI界面，决定下一步动作。

任务: {task}

执行历史:
{history_text}

请按以下格式返回JSON:
{{
    "reasoning": "你的思考过程",
    "action": "click|input|swipe|done",
    "parameters": {{
        "target_element": "要操作的元素描述（仅限click）",
        "text": "要输入的文本（仅限input）",
        "direction": "滑动方向（仅限swipe）"
    }}
}}"""
        
        response = decider_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", 
                         "image_url": {"url": f"data:image/jpeg;base64,{screenshot_b64}"}},
                        {"type": "text", "text": prompt}
                    ]
                }
            ],
            temperature=0
        )
        
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"[ERROR] Decider调用失败: {e}")
        return None


def get_grounder_coords(target_element, screenshot_b64, model=""):
    """调用Grounder模型获取坐标"""
    try:
        prompt = f"""在给定的UI截图中，找到以下元素的坐标:
{target_element}

请返回JSON:
{{
    "bbox": [x1, y1, x2, y2]
}}

其中[x1, y1, x2, y2]是元素的边界框坐标。"""
        
        response = grounder_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", 
                         "image_url": {"url": f"data:image/jpeg;base64,{screenshot_b64}"}},
                        {"type": "text", "text": prompt}
                    ]
                }
            ],
            temperature=0
        )
        
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"[ERROR] Grounder调用失败: {e}")
        return None


def execute_task(app_name, task_desc, device, data_dir, 
                decider_model="", grounder_model=""):
    """执行任务的主循环"""
    
    os.makedirs(data_dir, exist_ok=True)
    
    history = []
    actions = []
    step = 0
    
    logger.info(f"[START] 开始执行任务: {task_desc}")
    
    while step < MAX_STEPS:
        step += 1
        logger.info(f"\n[STEP] 第 {step} 步")
        
        # 获取截图
        screenshot_b64 = get_screenshot_b64(device)
        if not screenshot_b64:
            logger.error("[ERROR] 无法获取截图，任务终止")
            break
        
        # 保存截图
        try:
            path = device.screenshot()
            save_path = os.path.join(data_dir, f"{step:02d}_screenshot.jpg")
            if os.path.exists(path):
                import shutil
                shutil.copy(path, save_path)
                logger.info(f"[OK] 已保存截图: {save_path}")
        except Exception as e:
            logger.warning(f"[WARNING] 保存截图失败: {e}")
        
        # 构建历史文本
        history_text = "\n".join(f"{i}. {h}" for i, h in enumerate(history, 1)) \
                      if history else "(无历史)"
        
        # 调用Decider
        logger.info("[THINK] 调用Decider模型...")
        decider_response_str = get_decider_action(
            task_desc, history_text, screenshot_b64, decider_model
        )
        
        if not decider_response_str:
            logger.error("[ERROR] Decider返回空，任务终止")
            break
        
        logger.info(f"[RESPONSE] Decider响应: {decider_response_str[:100]}")
        
        # 解析响应
        decider_response = parse_json_response(decider_response_str)
        if not decider_response:
            logger.error("[ERROR] 无法解析Decider响应")
            break
        
        action = decider_response.get("action", "done")
        reasoning = decider_response.get("reasoning", "")
        parameters = decider_response.get("parameters", {})
        
        logger.info(f"[REASON] 思考: {reasoning}")
        logger.info(f"[ACTION] 动作: {action}")
        
        # 执行动作
        if action == "done":
            logger.info("[OK] 任务完成")
            actions.append({"step": step, "action": "done"})
            break
        
        elif action == "click":
            target = parameters.get("target_element", "")
            logger.info(f"[TARGET] 寻找目标: {target}")
            
            # 调用Grounder
            grounder_response_str = get_grounder_coords(
                target, screenshot_b64, grounder_model
            )
            
            if grounder_response_str:
                grounder_response = parse_json_response(grounder_response_str)
                if grounder_response and "bbox" in grounder_response:
                    bbox = grounder_response["bbox"]
                    x = int((bbox[0] + bbox[2]) / 2)
                    y = int((bbox[1] + bbox[3]) / 2)
                    
                    device.click(x, y)
                    actions.append({
                        "step": step,
                        "action": "click",
                        "position": [x, y],
                        "target": target
                    })
        
        elif action == "input":
            text = parameters.get("text", "")
            device.input_text(text)
            actions.append({
                "step": step,
                "action": "input",
                "text": text
            })
        
        elif action == "swipe":
            direction = parameters.get("direction", "down").lower()
            device.swipe(direction)
            actions.append({
                "step": step,
                "action": "swipe",
                "direction": direction
            })
        
        # 更新历史
        history.append(decider_response_str)
        time.sleep(1)
    
    # 保存结果
    result = {
        "app_name": app_name,
        "task_description": task_desc,
        "action_count": len(actions),
        "actions": actions
    }
    
    result_path = os.path.join(data_dir, "actions.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    logger.info(f"[OK] 结果已保存: {result_path}")
    return result


def extract_app_from_task(task_desc, app_package_names):
    """从任务描述中提取应用名称"""
    for app_name in app_package_names.keys():
        if app_name in task_desc:
            return app_name
    return None


def load_task_from_file(task_path):
    """从文件中加载任务列表或单个任务"""
    import json
    try:
        with open(task_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 如果是列表，取第一个任务
        if isinstance(data, list) and len(data) > 0:
            return data[0]
        # 如果是字符串或其他，直接返回
        return str(data)
    except Exception as e:
        logger.error(f"[ERROR] 无法读取任务文件 {task_path}: {e}")
        return None


def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(description="MobiAgent Mobile版")
    
    parser.add_argument("--service_ip", type=str, default="localhost",
                       help="LLM服务IP地址")
    parser.add_argument("--decider_port", type=int, default=8000,
                       help="Decider服务端口")
    parser.add_argument("--grounder_port", type=int, default=8001,
                       help="Grounder服务端口")
    parser.add_argument("--planner_port", type=int, default=8002,
                       help="Planner服务端口")
    parser.add_argument("--device", type=str, default="Android",
                       help="设备类型: Android 或 Harmony")
    parser.add_argument("--device_endpoint", type=str, default=None,
                       help="设备ADB端点")
    parser.add_argument("--app", type=str, default=None,
                       help="要打开的应用（如不指定则从task自动提取）")
    parser.add_argument("--task", type=str, required=True,
                       help="要执行的任务描述")
    parser.add_argument("--data_dir", type=str, default="./data",
                       help="数据保存目录")
    parser.add_argument("--decider_model", type=str, default="",
                       help="Decider模型名称")
    parser.add_argument("--grounder_model", type=str, default="",
                       help="Grounder模型名称")
    
    args = parser.parse_args()
    
    # 如果task是文件路径，先读取文件
    task_desc = args.task
    if os.path.isfile(args.task):
        logger.info(f"[LOAD] 从文件读取任务: {args.task}")
        loaded_task = load_task_from_file(args.task)
        if loaded_task:
            task_desc = loaded_task
            logger.info(f"[LOAD] 任务内容: {task_desc}")
        else:
            logger.error("[ERROR] 无法加载任务文件")
            return
    
    args.task = task_desc  # 更新args.task为实际任务描述
    
    # 应用包名映射
    app_package_names = {
        "携程": "ctrip.android.view",
        "支付宝": "com.eg.android.AlipayGphone",
        "微信": "com.tencent.mm",
        "淘宝": "com.taobao.taobao",
        "京东": "com.jingdong.app.mall",
        "美团": "com.sankuai.meituan",
        "滴滴": "com.sdu.didi.psnger",
        "微博": "com.sina.weibo",
    }
    
    # 如果未指定应用，则从任务描述中提取
    if not args.app:
        detected_app = extract_app_from_task(args.task, app_package_names)
        if detected_app:
            args.app = detected_app
            logger.info(f"[DETECT] 从任务描述中提取应用: {args.app}")
        else:
            logger.warning("[WARNING] 无法从任务中提取应用名称，使用默认应用")
            args.app = "微信"  # 默认微信
    
    try:
        # 初始化客户端
        logger.info("[CONNECT] 正在连接LLM服务...")
        init_clients(args.service_ip, args.decider_port, 
                    args.grounder_port, args.planner_port)
        
        # 连接设备
        logger.info("[DEVICE] 正在连接设备...")
        if args.device == "Android":
            device = AndroidDevice(args.device_endpoint)
        else:
            logger.error("[ERROR] 暂不支持其他设备类型")
            return
        
        # 启动应用
        logger.info(f"[APP] 正在启动应用: {args.app}")
        device.start_app(args.app)
        time.sleep(2)
        
        # 执行任务
        result = execute_task(
            args.app,
            args.task,
            device,
            args.data_dir,
            args.decider_model,
            args.grounder_model
        )
        
        # 停止应用
        logger.info(f"[STOP] 正在停止应用: {args.app}")
        device.stop_app(args.app)
        
        logger.info("[DONE] 执行完成")
    
    except KeyboardInterrupt:
        logger.info("[WARNING] 用户中断")
    except Exception as e:
        logger.error(f"[ERROR] 发生错误: {e}", exc_info=True)


if __name__ == "__main__":
    main()
