from openai import OpenAI
import uiautomator2 as u2
import base64
from PIL import Image
import json
import io
import logging
import time
import re
import os
import argparse
import textwrap
import cv2
import sys
import random
from abc import ABC, abstractmethod
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont


import numpy as np
from utils.local_experience import PromptTemplateSearch 
from pathlib import Path
from hmdriver2.driver import Driver
from hmdriver2.proto import KeyCode
from utils.load_md_prompt import load_prompt
from dotenv import load_dotenv
from utils.local_experience import PromptTemplateSearch 
from pathlib import Path
from .user_preference_extractor import (
    PreferenceExtractor, 
    retrieve_user_preferences, 
    should_extract_preferences,
    combine_context
)
 
# 清除可能已存在的 handlers，避免重复配置
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
    force=True  # Python 3.8+ 支持 force=True，强制重置
)
# >>>>>>>>>> logging 配置结束 <<<<<<<<<<



# 截图缩放比例
factor = 0.5

# ============ 常数定义 ============
MAX_STEPS = 35
MAX_RETRIES = 5
TEMP_INCREMENT = 0.1
INITIAL_TEMP = 0.0
API_TIMEOUT = 30
DECIDER_MAX_TOKENS = 256
GROUNDER_MAX_TOKENS = 128
DEVICE_WAIT_TIME = 0.5
APP_STOP_WAIT = 3

# 滑动坐标缩放比例
SWIPE_V_START = 0.3
SWIPE_V_END = 0.7
SWIPE_H_START = 0.3
SWIPE_H_END = 0.7

class Device(ABC):
    @abstractmethod
    def start_app(self, app):
        pass
    
    @abstractmethod
    def app_stop(self, package_name):
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
    def swipe_with_coords(self, start_x, start_y, end_x, end_y):
        pass

    @abstractmethod
    def keyevent(self, key):
        pass

    @abstractmethod
    def dump_hierarchy(self):
        pass

class AndroidDevice(Device):
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
            "携程": "ctrip.android.view",
            "华为商城": "com.vmall.client",
            "华为视频": "com.huawei.himovie",
            "华为音乐": "com.huawei.music",
            "华为应用市场": "com.huawei.appmarket",
            "拼多多": "com.xunmeng.pinduoduo",
            "大众点评": "com.dianping.v1",
            "小红书": "com.xingin.xhs",
            "浏览器": "com.microsoft.emmx"
        }

    def start_app(self, app):
        package_name = self.app_package_names.get(app)
        if not package_name:
            raise ValueError(f"App '{app}' is not registered with a package name.")
        self.d.app_start(package_name, stop=True)
        time.sleep(3)
        if not self.d.app_wait(package_name, timeout=10):
            raise RuntimeError(f"Failed to start app '{app}' with package '{package_name}'")
    
    def app_start(self, package_name):
        self.d.app_start(package_name, stop=True)
        time.sleep(DEVICE_WAIT_TIME * 2)
        if not self.d.app_wait(package_name, timeout=10):
            raise RuntimeError(f"Failed to start package '{package_name}'")

    def app_stop(self, package_name):
        self.d.app_stop(package_name)

    def screenshot(self, path):
        self.d.screenshot(path)

    def click(self, x, y):
        self.d.click(x, y)
        time.sleep(DEVICE_WAIT_TIME)

    def clear_input(self):
    # 按下全选（需要 Android 支持 keyevent META_CTRL_ON）
        self.d.shell(['input', 'keyevent', 'KEYCODE_MOVE_END'])
        self.d.shell(['input', 'keyevent', 'KEYCODE_MOVE_HOME'])
        self.d.shell(['input', 'keyevent', 'KEYCODE_DEL'])

    def input(self, text):
        current_ime = self.d.current_ime()
        self.d.shell(['settings', 'put', 'secure', 'default_input_method', 'com.android.adbkeyboard/.AdbIME'])
        time.sleep(DEVICE_WAIT_TIME)
        # add clear text command, depending on 'ADB Keyboard'
        self.d.shell(['am', 'broadcast', '-a', 'ADB_CLEAR_TEXT'])
        time.sleep(0.2)

        charsb64 = base64.b64encode(text.encode('utf-8')).decode('utf-8')
        self.d.shell(['am', 'broadcast', '-a', 'ADB_INPUT_B64', '--es', 'msg', charsb64])
        time.sleep(DEVICE_WAIT_TIME)
        self.d.shell(['settings', 'put', 'secure', 'default_input_method', current_ime])
        # time.sleep(DEVICE_WAIT_TIME)
        # Press Enter key to confirm input
        self.d.shell(['input', 'keyevent', 'KEYCODE_ENTER'])


    def swipe(self, direction, scale=0.5):
        # self.d.swipe_ext(direction, scale)
        # self.d.swipe_ext(direction=direction, scale=scale)
        if direction.lower() == "up":
            self.d.swipe(0.5,SWIPE_V_END,0.5,SWIPE_V_START,duration=0.2)
        elif direction.lower() == "down":
            self.d.swipe(0.5,SWIPE_V_START,0.5,SWIPE_V_END,duration=0.2)
        elif direction.lower() == "left":
            self.d.swipe(SWIPE_H_END,0.5,SWIPE_H_START,0.5,duration=0.2)
        elif direction.lower() == "right":
            self.d.swipe(SWIPE_H_START,0.5,SWIPE_H_END,0.5,duration=0.2)

    def swipe_with_coords(self, start_x, start_y, end_x, end_y):
        """Swipe from (start_x, start_y) to (end_x, end_y)"""
        self.d.swipe(start_x, start_y, end_x, end_y, duration=0.2)

    def keyevent(self, key):
        self.d.keyevent(key)

    def dump_hierarchy(self):
        return self.d.dump_hierarchy()

class HarmonyDevice(Device):
    def __init__(self):
        super().__init__()
        self.d = Driver()
        self.app_package_names = {
            "携程": "com.ctrip.harmonynext",
            "飞猪": "com.fliggy.hmos",
            "IntelliOS": "ohos.hongmeng.intellios",
            "同城": "com.tongcheng.hmos",
            "携程旅行": "com.ctrip.harmonynext",
            "饿了么": "me.ele.eleme",
            "知乎": "com.zhihu.hmos",
            "哔哩哔哩": "yylx.danmaku.bili",
            "微信": "com.tencent.wechat",
            "小红书": "com.xingin.xhs_hos",
            "QQ音乐": "com.tencent.hm.qqmusic",
            "高德地图": "com.amap.hmapp",
            "淘宝": "com.taobao.taobao4hmos",
            "微博": "com.sina.weibo.stage",
            "京东": "com.jd.hm.mall",
            "飞猪旅行": "com.fliggy.hmos",
            "天气": "com.huawei.hmsapp.totemweather",
            "什么值得买": "com.smzdm.client.hmos",
            "闲鱼": "com.taobao.idlefish4ohos",
            "慧通差旅": "com.smartcom.itravelhm",
            "PowerAgent": "com.example.osagent",
            "航旅纵横": "com.umetrip.hm.app",
            "滴滴出行": "com.sdu.didi.hmos.psnger",
            "电子邮件": "com.huawei.hmos.email",
            "图库": "com.huawei.hmos.photos",
            "日历": "com.huawei.hmos.calendar",
            "心声社区": "com.huawei.it.hmxinsheng",
            "信息": "com.ohos.mms",
            "文件管理": "com.huawei.hmos.files",
            "运动健康": "com.huawei.hmos.health",
            "智慧生活": "com.huawei.hmos.ailife",
            "豆包": "com.larus.nova.hm",
            "WeLink": "com.huawei.it.welink",
            "设置": "com.huawei.hmos.settings",
            "懂车帝": "com.ss.dcar.auto",
            "美团外卖": "com.meituan.takeaway",
            "大众点评": "com.sankuai.dianping",
            "美团": "com.sankuai.hmeituan",
            "浏览器": "com.huawei.hmos.browser",
            "微博": "com.sina.weibo.stage",
            "饿了么": "me.ele.eleme",
            "拼多多": "com.xunmeng.pinduoduo.hos"
        }

    def start_app(self, app):
        package_name = self.app_package_names.get(app)
        if not package_name:
            raise ValueError(f"App '{app}' is not registered with a package name.")
        self.d.start_app(package_name)
        time.sleep(2)

    def app_start(self, package_name):
        # self.d.start_app(package_name)
        self.d.force_start_app(package_name)
        time.sleep(1.5)

    def app_stop(self, package_name):
        self.d.stop_app(package_name)

    def screenshot(self, path):
        self.d.screenshot(path)

    def click(self, x, y):
        self.d.click(x, y)
        time.sleep(DEVICE_WAIT_TIME)

    def input(self, text):
        self.d.shell("uitest uiInput keyEvent 2072 2017")
        self.d.press_key(2071)
        self.d.input_text(text)
        # Press Enter key to confirm input
        self.d.press_key(KeyCode.ENTER)
        # self.d.press_key(KeyCode.SEND)

    def swipe(self, direction, scale=0.5):
        # self.d.swipe_ext(direction, scale=scale)
        if direction.lower() == "up":
            self.d.swipe(0.5,SWIPE_V_END,0.5,SWIPE_V_START,speed=1000)
        elif direction.lower() == "down":
            self.d.swipe(0.5,SWIPE_V_START,0.5,SWIPE_V_END,speed=1000)
        elif direction.lower() == "left":
            self.d.swipe(SWIPE_H_END,0.5,SWIPE_H_START,0.5,speed=1000)
        elif direction.lower() == "right":
            self.d.swipe(SWIPE_H_START,0.5,SWIPE_H_END,0.5,speed=1000)

    def swipe_with_coords(self, start_x, start_y, end_x, end_y):
        """Swipe from (start_x, start_y) to (end_x, end_y)"""
        # Convert absolute coordinates to normalized coordinates if needed
        # For Harmony Device, swipe expects coordinates in format (x, y, x, y)
        self.d.swipe(start_x, start_y, end_x, end_y, speed=1000)

    def keyevent(self, key):
        self.d.press_key(key)

    def dump_hierarchy(self):
        return self.d.dump_hierarchy()

decider_client = None
grounder_client = None
planner_client = None

planner_model = ""
decider_model = ""
grounder_model = ""


# 全局偏好提取器
preference_extractor = None
def init(service_ip, decider_port, grounder_port, planner_port, enable_user_profile=False, use_graphrag=False):
    global decider_client, grounder_client, planner_client, general_client, general_model, apps, preference_extractor
    
    # 加载环境变量
    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path)
    decider_client = OpenAI(
        api_key = "0",
        base_url = f"http://{service_ip}:{decider_port}/v1",
    )
    grounder_client = OpenAI(
        api_key = "0",
        base_url = f"http://{service_ip}:{grounder_port}/v1",
    )
    planner_client = OpenAI(
        api_key = "0",
        base_url = f"http://{service_ip}:{planner_port}/v1",
    )
    
    # 初始化偏好提取器（可由命令行开关控制）
    if enable_user_profile:
        preference_extractor = PreferenceExtractor(planner_client, planner_model, use_graphrag=use_graphrag)
    else:
        preference_extractor = None
    

# ============ 工具函数 ============

def call_model_with_validation_retry(client, model, messages, validator_func, max_retries=MAX_RETRIES, max_tokens=256, context="Model"):
    """
    通用模型API调用函数：支持JSON解析 + 自定义校验 + 校验失败自动重试和温度递增
    
    Args:
        client: OpenAI客户端
        model: 模型名称
        messages: 消息列表
        validator_func: 校验函数，接收解析后的JSON对象，如果校验失败则抛异常
        max_retries: 最大重试次数
        max_tokens: 最大生成Token数
        context: 日志上下文（如"Decider", "Grounder"）
    
    Returns:
        解析后的 JSON 响应对象（已通过校验）
    
    Raises:
        Exception: 重试失败或校验一直失败后抛出异常
    """
    temperature = INITIAL_TEMP
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            response_str = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                timeout=API_TIMEOUT,
                max_tokens=max_tokens,
            ).choices[0].message.content
            end_time = time.time()
            logging.info(f"[evaluation] {context} time taken: {end_time - start_time:.2f} seconds")
            logging.info(f"{context} response: \n{response_str}")
            
            # 尝试解析 JSON
            parsed_response = robust_json_loads(response_str)
            
            # 执行校验函数
            validator_func(parsed_response)
            
            return parsed_response
            
        except Exception as e:
            temperature = INITIAL_TEMP + (attempt + 1) * TEMP_INCREMENT
            logging.error(f"{context} 调用或校验失败: {e}, 正在重试 temperature={temperature:.1f}...")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                raise
    
    raise RuntimeError(f"{context} API calls exhausted after {max_retries} attempts")


def _load_json_from_text(raw_text):
    """尝试从混杂文本中提取 JSON；失败返回 None。"""
    if raw_text is None:
        return None
    if not isinstance(raw_text, str):
        raw_text = str(raw_text)

    text = raw_text.strip()

    def _try_load(candidate):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None

    # 直接解析
    parsed = _try_load(text)
    if parsed is not None:
        return parsed

    # ```json ... ``` 或 ``` ... ``` 代码块
    for pattern in [r"```json\s*([\s\S]*?)\s*```", r"```\s*([\s\S]*?)\s*```"]:
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            parsed = _try_load(match.group(1).strip())
            if parsed is not None:
                return parsed

    # 中文省略号替换
    normalized = text.replace("…", "...")
    if normalized != text:
        parsed = _try_load(normalized)
        if parsed is not None:
            return parsed
        text = normalized

    # 从最外层花括号截取
    start_idx = text.find('{')
    if start_idx != -1:
        brace_count = 0
        for i in range(start_idx, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    candidate = text[start_idx:i+1]
                    parsed = _try_load(candidate)
                    if parsed is not None:
                        return parsed

    return None


def parse_json_response(response_str: str, is_guided_decoding: bool = True) -> dict:
    """
    解析 JSON 响应，支持 guided decoding 和普通模式
    
    Args:
        response_str: 模型返回的响应字符串
        is_guided_decoding: 是否启用了 guided decoding（默认 True）
        
    Returns:
        解析后的 JSON 对象
        
    说明：
        - 当启用 guided decoding 时，模型输出应该是纯 JSON 格式
        - 当禁用 guided decoding 时，可能包含 markdown code block 或其他文本
    """
    parsed = _load_json_from_text(response_str)
    if parsed is None:
        logging.error(f"无法在响应中找到有效的 JSON")
        logging.error(f"原始响应: {str(response_str)[:200]}...")
        raise ValueError(f"无法解析 JSON 响应，响应格式不正确")
    return parsed

def get_screenshot(device, device_type="Android"):
    """
    获取设备截图并编码为base64
    
    Args:
        device: 设备对象
        device_type: 设备类型，"Android" 或 "Harmony"
        
    Returns:
        Base64编码的截图字符串
    """
    # 根据设备类型使用不同的截图路径，避免冲突
    if device_type == "Android":
        screenshot_path = "screenshot-Android.jpg"
    else:
        screenshot_path = "screenshot-Harmony.jpg"
    
    device.screenshot(screenshot_path)
    # resize the screenshot to reduce the size for processing
    img = Image.open(screenshot_path)
    img = img.resize((int(img.width * factor), int(img.height * factor)), Image.Resampling.LANCZOS)
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    screenshot = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return screenshot

def convert_qwen3_coordinates_to_absolute(bbox_or_coords, img_width, img_height, is_bbox=True):
    """
    将 Qwen3 模型返回的相对坐标（0-1000范围）转换为绝对坐标
    
    Args:
        bbox_or_coords: 相对坐标或边界框，范围为 0-1000
        img_width: 图像宽度
        img_height: 图像高度
        is_bbox: 是否为边界框（True）或坐标点（False）
        
    Returns:
        转换后的绝对坐标或边界框
    """
    if is_bbox:
        # bbox: [x1, y1, x2, y2]
        x1, y1, x2, y2 = bbox_or_coords
        x1 = int(x1 / 1000 * img_width)
        x2 = int(x2 / 1000 * img_width)
        y1 = int(y1 / 1000 * img_height)
        y2 = int(y2 / 1000 * img_height)
        return [x1, y1, x2, y2]
    else:
        # coordinates: [x, y]
        x, y = bbox_or_coords
        x = int(x / 1000 * img_width)
        y = int(y / 1000 * img_height)
        return [x, y]

def validate_action_parameters(decider_response):
    """
    校验不同动作的字段完整性
    
    Args:
        decider_response: 解析后的 JSON 响应字典
    
    Raises:
        ValueError: 当必需字段缺失时
    """
    action = decider_response.get("action")
    parameters = decider_response.get("parameters", {})
    
    if not action:
        raise ValueError("Missing required field: 'action'")
    
    if not decider_response.get("reasoning"):
        raise ValueError("Missing required field: 'reasoning'")
    
    # 根据不同动作类型校验必需参数
    if action == "click":
        if not parameters.get("target_element"):
            raise ValueError("Click action missing required parameter: 'target_element'")
        # e2e模式下需要校验bbox
        # 注意：这里不直接检查bbox，因为可能在非e2e模式下不需要
    elif action == "click_input":
        if not parameters.get("target_element"):
            raise ValueError("Click_input action missing required parameter: 'target_element'")
        if not parameters.get("bbox"):
            raise ValueError("Click_input action missing required parameter: 'bbox'")
        if not parameters.get("text"):
            raise ValueError("Click_input action missing required parameter: 'text'")
        
    elif action == "input":
        if "text" not in parameters:
            raise ValueError("Input action missing required parameter: 'text'")
        # text可以为空字符串，所以只检查是否存在该字段
    
    elif action == "swipe":
        direction = parameters.get("direction")
        if not direction:
            raise ValueError("Swipe action missing required parameter: 'direction'")
        if direction.upper() not in ["UP", "DOWN", "LEFT", "RIGHT"]:
            raise ValueError(f"Invalid swipe direction: '{direction}'. Must be one of: UP, DOWN, LEFT, RIGHT")
    
    elif action == "done":
        status = parameters.get("status")
        if not status:
            raise ValueError("Done action missing required parameter: 'status'")
    
    elif action == "long_press":
        if not parameters.get("target_element"):
            raise ValueError("Long_press action missing required parameter: 'target_element'")
    
    elif action == "open_app":
        if not parameters.get("app_name"):
            raise ValueError("Open_app action missing required parameter: 'app_name'")
    
    elif action == "wait":
        # wait动作通常不需要额外参数
        pass
    
    else:
        raise ValueError(f"Unknown action: '{action}'")
    
    return True
def create_swipe_visualization(data_dir, image_index, direction, start_x=None, start_y=None, end_x=None, end_y=None):
    """为滑动动作创建可视化图像"""
    try:
        # 读取原始截图
        img_path = os.path.join(data_dir, f"{image_index}.jpg")
        if not os.path.exists(img_path):
            return
            
        img = cv2.imread(img_path)
        if img is None:
            return
            
        height, width = img.shape[:2]
        
        # 如果提供了具体坐标，使用具体坐标；否则根据方向计算
        if start_x is not None and start_y is not None and end_x is not None and end_y is not None:
            start_point = (int(start_x), int(start_y))
            end_point = (int(end_x), int(end_y))
        else:
            # 根据方向计算箭头起点和终点
            center_x, center_y = width // 2, height // 2
            arrow_length = min(width, height) // 4
            
            if direction == "up":
                start_point = (center_x, center_y + arrow_length // 2)
                end_point = (center_x, center_y - arrow_length // 2)
            elif direction == "down":
                start_point = (center_x, center_y - arrow_length // 2)
                end_point = (center_x, center_y + arrow_length // 2)
            elif direction == "left":
                start_point = (center_x + arrow_length // 2, center_y)
                end_point = (center_x - arrow_length // 2, center_y)
            elif direction == "right":
                start_point = (center_x - arrow_length // 2, center_y)
                end_point = (center_x + arrow_length // 2, center_y)
            else:
                return
            
        # 绘制箭头
        cv2.arrowedLine(img, start_point, end_point, (255, 0, 0), 8, tipLength=0.3)  # 蓝色箭头
        
        # 添加文字说明
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"SWIPE {direction.upper()}"
        text_size = cv2.getTextSize(text, font, 1.5, 3)[0]
        text_x = (width - text_size[0]) // 2
        text_y = 50
        cv2.putText(img, text, (text_x, text_y), font, 1.5, (255, 0, 0), 3)  # 蓝色文字
        
        # 保存可视化图像
        swipe_path = os.path.join(data_dir, f"{image_index}_swipe.jpg")
        cv2.imwrite(swipe_path, img)
        
    except Exception as e:
        logging.warning(f"Failed to create swipe visualization: {e}")

# 预处理 response_str，增强健壮性
def robust_json_loads(s):
    """健壮解析 response_str，使用统一解析逻辑。"""
    parsed = _load_json_from_text(s)
    if parsed is None:
        logging.error(f"解析 response_str 失败")
        logging.error(f"原始内容: {str(s)[:300]}...")
        raise ValueError(f"无法解析 JSON 响应: 响应格式不正确")
    return parsed


def get_device_paths(device_type, image_index=None):
    """
    获取设备相关的文件路径
    
    Args:
        device_type: 设备类型 ("Android" 或 "Harmony")
        image_index: 图像索引（可选）
    
    Returns:
        dict: 包含 img_path, screenshot_name, current_dir 的字典
    """
    current_dir = os.getcwd()
    screenshot_name = "screenshot-Android.jpg" if device_type == "Android" else "screenshot-Harmony.jpg"
    img_path = os.path.join(current_dir, screenshot_name)
    return {
        "img_path": img_path,
        "screenshot_name": screenshot_name,
        "current_dir": current_dir
    }


def validate_decider_response(response_dict, use_e2e=False):
    """
    校验 Decider 模型响应，包括基础参数校验和e2e模式特殊校验
    
    Args:
        response_dict: 解析后的JSON响应对象
        use_e2e: 是否使用e2e模式
    
    Raises:
        ValueError: 校验失败时抛出异常
    """
    # 基础参数校验
    validate_action_parameters(response_dict)
    
    # e2e模式的特殊校验
    if use_e2e:
        action_name = response_dict["action"]
        parameters = response_dict["parameters"]
        
        if action_name == "click":
            # e2e模式下click动作需要bbox
            if "bbox" not in parameters or parameters["bbox"] is None:
                raise ValueError("E2E mode: click action missing required parameter: 'bbox'")
        
        elif action_name == "swipe":
            # e2e模式下swipe动作可能需要起始和结束坐标
            if "start_coords" in parameters and "end_coords" in parameters:
                if parameters["start_coords"] is None or parameters["end_coords"] is None:
                    logging.warning("E2E mode: swipe action has null start_coords or end_coords, will fall back to direction-based swipe")


def validate_grounder_response(response_dict):
    """
    校验 Grounder 模型响应
    
    Args:
        response_dict: 解析后的JSON响应对象
    
    Raises:
        ValueError: 校验失败时抛出异常
    """
    # Grounder需要至少返回coordinates或bbox
    if "coordinates" not in response_dict and not any(
        key.lower() in ["bbox", "bbox_2d", "bbox-2d", "bbox_2D", "bbox2d"]
        for key in response_dict.keys()
    ):
        raise ValueError("Grounder response must contain 'coordinates' or 'bbox' field")

def compute_swipe_positions(direction, img_width, img_height):
    direction = direction.upper()
    if direction == "DOWN":
        return (
            0.5 * img_width,
            SWIPE_V_START * img_height,
            0.5 * img_width,
            SWIPE_V_END * img_height,
        )
    if direction == "UP":
        return (
            0.5 * img_width,
            SWIPE_V_END * img_height,
            0.5 * img_width,
            SWIPE_V_START * img_height,
        )
    if direction == "LEFT":
        return (
            SWIPE_H_END * img_width,
            0.5 * img_height,
            SWIPE_H_START * img_width,
            0.5 * img_height,
        )
    if direction == "RIGHT":
        return (
            SWIPE_H_START * img_width,
            0.5 * img_height,
            SWIPE_H_END * img_width,
            0.5 * img_height,
        )
    raise ValueError(f"Unknown swipe direction: {direction}")


def build_decider_messages(task, history, screenshot, e2e):
    from prompts.decider_qwen3_e2e import DECIDER_SYSTEM_PROMPT, DECIDER_USER_PROMPT, DECIDER_CURRENT_STEP_PROMPT
    
    # 1. 处理历史记录字符串
    if len(history) == 0:
        history_str = "(No history)"
    else:
        history_str = "\n".join(f"{idx}. {h}" for idx, h in enumerate(history, 1))

    # 2. 准备前半部分文本（对应训练数据中 <image> 之前的内容）
    # 包含：Task, History, Constraints
    context_text = DECIDER_USER_PROMPT.format(task=task, history=history_str)
    
    # 3. 准备后半部分文本（对应训练数据中 <image> 之后的内容）
    # 包含：Instruction (Please provide the next action...)
    instruction_text = DECIDER_CURRENT_STEP_PROMPT

    # 4. 构建单一的 User Message
    # 结构严格遵循：[前半段文本] -> [图片] -> [后半段指令]
    # 这样模型看到的输入序列就是：Text(Context) + ImageToken + Text(Instruction)
    messages = [
        {
            "role": "system",
            "content": DECIDER_SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": context_text  # 对应 <image> 上方的文本
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{screenshot}"
                    }
                },
                {
                    "type": "text",
                    "text": instruction_text  # 对应 <image> 下方的文本
                }
            ]
        }
    ]
    
    # 打印用于调试（实际生产建议去掉）
    print(json.dumps(messages, ensure_ascii=False, indent=2)[:1000] + "...")
    
    return messages

def task_in_app(app, old_task, task, device, data_dir, bbox_flag=True, use_qwen3=True, device_type="Android", use_e2e=False):
    history = []
    actions = []
    reacts = []

    if use_e2e:
        # 在e2e模式下使用e2e_qwen3.md，否则使用decider_v2.md
        decider_prompt_template = load_prompt("e2e_qwen3.md")
        logging.info("Using e2e mode with e2e_qwen3.md")

    elif use_qwen3:
        grounder_prompt_template_bbox = load_prompt("grounder_qwen3_bbox.md")
        grounder_prompt_template_no_bbox = load_prompt("grounder_qwen3_coordinates.md")
        
        decider_prompt_template = load_prompt("decider_v2.md")

    else:
        grounder_prompt_template_bbox = load_prompt("grounder_bbox.md")
        grounder_prompt_template_no_bbox = load_prompt("grounder_coordinates.md")
        decider_prompt_template = load_prompt("decider_v2.md")
    while True:     
        if len(actions) >= MAX_STEPS:
            logging.info("Reached maximum steps, stopping the task.")
            break
        
        if len(history) == 0:
            history_str = "(No history)"
        else:
            history_str = "\n".join(f"{idx}. {h}" for idx, h in enumerate(history, 1))
        screenshot_resize = get_screenshot(device, device_type)

        
        messages = build_decider_messages(task, history, screenshot_resize, use_e2e)
        logging.info(f"Decider messages[200]: \n{messages[200:]}")

        # --- 调用 Decider 模型 ---
        try:
            # 为e2e模式创建特定的校验器
            def decider_validator(response):
                validate_decider_response(response, use_e2e=use_e2e)
            
            decider_response = call_model_with_validation_retry(
                decider_client,
                decider_model,
                messages,
                validator_func=decider_validator,
                max_retries=MAX_RETRIES,
                max_tokens=DECIDER_MAX_TOKENS,
                context="Decider"
            )

            converted_item = {
                "reasoning": decider_response["reasoning"],
                "function": {
                    "name": decider_response["action"],
                    "parameters": decider_response["parameters"]
                }
            }
        except Exception as e:
            logging.error(f"Decider 处理失败: {e}")
            raise

        
        reacts.append(converted_item)
        action = decider_response["action"]

        # compute image index for this loop iteration (1-based)
        image_index = len(actions) + 1
        device_paths = get_device_paths(device_type)
        img_path = device_paths["img_path"]
        current_image = device_paths["screenshot_name"]
        
        save_path = os.path.join(data_dir, f"{image_index}.jpg")
        img = Image.open(img_path)
        img.save(save_path)

        # attach index to the most recent react (reasoning)
        if reacts:
            try:
                reacts[-1]["action_index"] = image_index
            except Exception:
                pass

        # 根据设备类型保存hierarchy
        
        if device_type == "Android":
            logging.info("Dumping UI hierarchy...")
            try:
                hierarchy = device.dump_hierarchy()
            except Exception as e:
                logging.error(f"Failed to dump UI hierarchy: {e}")
                hierarchy = "<hierarchy_dump_failed/>"
            # Android设备保存为XML格式
            hierarchy_path = os.path.join(data_dir, f"{image_index}.xml")
            with open(hierarchy_path, "w", encoding="utf-8") as f:
                f.write(hierarchy)
        else:
            try:
                hierarchy = device.dump_hierarchy()
            except Exception as e:
                logging.error(f"Failed to dump UI hierarchy: {e}")
                hierarchy = {}
            # Harmony设备保存为JSON格式
            hierarchy_path = os.path.join(data_dir, f"{image_index}.json")
            try:
                # 尝试将hierarchy解析为JSON（如果已是JSON字符串）
                if isinstance(hierarchy, str):
                    hierarchy_json = json.loads(hierarchy)
                else:
                    hierarchy_json = hierarchy
                with open(hierarchy_path, "w", encoding="utf-8") as f:
                    json.dump(hierarchy_json, f, ensure_ascii=False, indent=2)
            except (json.JSONDecodeError, TypeError):
                # 如果解析失败，直接保存为字符串
                logging.warning(f"Failed to parse hierarchy as JSON, saving as plain text")
                with open(hierarchy_path, "w", encoding="utf-8") as f:
                    f.write(str(hierarchy))
        
        if action == "done":
            print("Task completed.")
            status = decider_response["parameters"]["status"]
            actions.append({
                "type": "done",
                "status": status,
                "action_index": image_index
            })
            break
        if action == "click":
            reasoning = decider_response["reasoning"]
            target_element = decider_response["parameters"]["target_element"]
            
            # e2e模式：直接从decider获取bbox，不调用grounder
            if use_e2e:
                bbox = decider_response["parameters"]["bbox"]
                if bbox is None:
                    logging.error("E2E mode: bbox not found in decider response")
                    raise ValueError("E2E mode requires bbox in decider response")
                
                logging.info(f"E2E mode: Using bbox directly from decider: {bbox}")
                # 使用 Qwen3 模型进行坐标转换
                if use_qwen3:
                    bbox = convert_qwen3_coordinates_to_absolute(bbox, img.width, img.height, is_bbox=True)
                x1, y1, x2, y2 = bbox
            else:
                # 调用 Grounder 获取坐标
                grounder_prompt = (grounder_prompt_template_bbox if bbox_flag else grounder_prompt_template_no_bbox).format(reasoning=reasoning, description=target_element)
                
                try:
                    grounder_response = call_model_with_validation_retry(
                        grounder_client,
                        grounder_model,
                        [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{screenshot_resize}"}},
                                    {"type": "text", "text": grounder_prompt},
                                ]
                            }
                        ],
                        validator_func=validate_grounder_response,
                        max_retries=MAX_RETRIES,
                        max_tokens=GROUNDER_MAX_TOKENS,
                        context="Grounder"
                    )
                except Exception as e:
                    logging.error(f"Grounder 处理失败: {e}")
                    raise

                if bbox_flag:
                    # 获取bbox字段（validate_grounder_response已确保存在）
                    bbox = None
                    for key in grounder_response:
                        if key.lower() in ["bbox", "bbox_2d", "bbox-2d", "bbox_2D", "bbox2d"]:
                            bbox = grounder_response[key]
                            break
                    
                    if bbox is None:
                        raise ValueError("Grounder response validation failed: no bbox field found")

                    # 如果使用 Qwen3 模型，进行坐标转换
                    if use_qwen3:
                        bbox = convert_qwen3_coordinates_to_absolute(bbox, img.width, img.height, is_bbox=True)
                        x1, y1, x2, y2 = bbox
                    else:
                        x1, y1, x2, y2 = [int(coord/factor) for coord in bbox]

                else:
                    coordinates = grounder_response["coordinates"]
                    if use_qwen3:
                        coordinates = convert_qwen3_coordinates_to_absolute(coordinates, img.width, img.height, is_bbox=False)
                        x, y = coordinates
                    else:
                        x, y = [int(coord / factor) for coord in coordinates]
            
            # 通用的click处理逻辑（e2e和非e2e都使用）
            if bbox_flag or use_e2e:
                print(f"Clicking on bbox: [{x1}, {y1}, {x2}, {y2}]")
                print(f"Image size: width={img.width}, height={img.height}")
                print(f"Adjusted bbox: [{x1}, {y1}, {x2}, {y2}]")
                position_x = (x1 + x2) // 2
                position_y = (y1 + y2) // 2
                device.click(position_x, position_y)
                # save action (record index only)
                actions.append({
                    "type": "click",
                    "position_x": position_x,
                    "position_y": position_y,
                    "bounds": [x1, y1, x2, y2],
                    "action_index": image_index
                })
                history.append(json.dumps(decider_response, ensure_ascii=False))

                # 创建可视化图像
                img_path = os.path.join(device_paths["current_dir"], current_image)
                save_path = os.path.join(data_dir, f"{image_index}_highlighted.jpg")
                img = Image.open(img_path)
                draw = ImageDraw.Draw(img)
                font = ImageFont.truetype("msyh.ttf", 40)
                text = f"CLICK [{position_x}, {position_y}]"
                text = textwrap.fill(text, width=20)
                text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:]
                draw.text((img.width / 2 - text_width / 2, 0), text, fill="red", font=font)
                img.save(save_path)

                # 绘制边界框
                bounds_path = os.path.join(data_dir, f"{image_index}_bounds.jpg")
                img_bounds = Image.open(save_path)
                draw_bounds = ImageDraw.Draw(img_bounds)
                draw_bounds.rectangle([x1, y1, x2, y2], outline='red', width=5)
                img_bounds.save(bounds_path)

                # 绘制点
                cv2image = cv2.imread(bounds_path)
                if cv2image is not None:
                    # 在点击位置画圆点
                    cv2.circle(cv2image, (position_x, position_y), 15, (0, 255, 0), -1)  # 绿色实心圆
                    # 保存带点击点的图像
                    click_point_path = os.path.join(data_dir, f"{image_index}_click_point.jpg")
                    cv2.imwrite(click_point_path, cv2image)
            else:
                # 非bbox_flag的情况（使用coordinates）
                device.click(x, y)
                actions.append({
                    "type": "click",
                    "position_x": x,
                    "position_y": y,
                    "action_index": image_index
                })
                history.append(json.dumps(decider_response, ensure_ascii=False))
                
        elif action == "click_input":
            reasoning = decider_response["reasoning"]
            target_element = decider_response["parameters"]["target_element"]
            text = decider_response["parameters"]["text"]
            bbox = decider_response["parameters"]["bbox"]
            bbox = convert_qwen3_coordinates_to_absolute(bbox, img.width, img.height, is_bbox=True)
            x1, y1, x2, y2 = bbox
            position_x = (x1 + x2) // 2
            position_y = (y1 + y2) // 2

            device.click(position_x, position_y)
            actions.append({
                "type": "click_input",
                "position_x": position_x,
                "position_y": position_y,
                "bounds": [x1, y1, x2, y2],
                "text": f"{text}",
                "action_index": image_index
            })

            device.input(text)
            history.append(json.dumps(decider_response, ensure_ascii=False))

        elif action == "input":
            text = decider_response["parameters"]["text"]
            device.input(text)
            actions.append({
                "type": "input",
                "text": text,
                "action_index": image_index
            })
            
            history.append(json.dumps(decider_response, ensure_ascii=False))

        elif action == "swipe":
            direction = decider_response["parameters"]["direction"]
            direction = direction.upper()
            
            # e2e模式：尝试获取起始和结束坐标
            if use_e2e:
                start_coords = decider_response["parameters"].get("start_coords")
                end_coords = decider_response["parameters"].get("end_coords")
                
                if start_coords and end_coords:
                    # 进行坐标转换（如果需要）
                    if use_qwen3:
                        start_coords = convert_qwen3_coordinates_to_absolute(start_coords, img.width, img.height, is_bbox=False)
                        end_coords = convert_qwen3_coordinates_to_absolute(end_coords, img.width, img.height, is_bbox=False)
                    
                    start_x, start_y = start_coords
                    end_x, end_y = end_coords
                    
                    logging.info(f"E2E mode: swipe from [{start_x}, {start_y}] to [{end_x}, {end_y}]")
                    device.swipe_with_coords(start_x, start_y, end_x, end_y)
                    
                    actions.append({
                        "type": "swipe",
                        "press_position_x": start_x,
                        "press_position_y": start_y,
                        "release_position_x": end_x,
                        "release_position_y": end_y,
                        "direction": direction.lower(),
                        "action_index": image_index
                    })
                    history.append(json.dumps(decider_response, ensure_ascii=False))
                    create_swipe_visualization(data_dir, image_index, direction.lower(), start_x, start_y, end_x, end_y)
                else:
                    logging.warning("E2E mode: start_coords or end_coords not found, falling back to direction-based swipe")
                    press_position_x, press_position_y, release_position_x, release_position_y = compute_swipe_positions(direction, img.width, img.height)
                    device.swipe_with_coords(press_position_x, press_position_y, release_position_x, release_position_y,)
                    actions.append({
                        "type": "swipe",
                        "press_position_x": press_position_x,
                        "press_position_y": press_position_y,
                        "release_position_x": release_position_x,
                        "release_position_y": release_position_y,
                        "direction": direction.lower(),
                        "action_index": image_index
                    })
                    history.append(json.dumps(decider_response, ensure_ascii=False))
                    create_swipe_visualization(data_dir, image_index, direction.lower())
            else:
                press_position_x, press_position_y, release_position_x, release_position_y = compute_swipe_positions(direction, img.width, img.height)
                device.swipe_with_coords(press_position_x, press_position_y, release_position_x, release_position_y,)
                actions.append({
                    "type": "swipe",
                    "press_position_x": press_position_x,
                    "press_position_y": press_position_y,
                    "release_position_x": release_position_x,
                    "release_position_y": release_position_y,
                    "direction": direction.lower(),
                    "action_index": image_index
                })
                    
                history.append(json.dumps(decider_response, ensure_ascii=False))
                
                # 为滑动创建可视化
                create_swipe_visualization(data_dir, image_index, direction.lower())

            # else:
            #     raise ValueError(f"Unknown swipe direction: {direction}")
        elif action == "wait":
            print("Waiting for a while...")
            actions.append({
                "type": "wait",
                "action_index": image_index
            })
            time.sleep(DEVICE_WAIT_TIME * 2)
            history.append(json.dumps(decider_response, ensure_ascii=False))
        else:
            raise ValueError(f"Unknown action: {action}")
        
    
    data = {
        "app_name": app,
        "task_type": None,
        "old_task_description": old_task,
        "task_description": task,
        "action_count": len(actions),
        "actions": actions
    }

    with open(os.path.join(data_dir, "actions.json"), "w", encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    with open(os.path.join(data_dir, "react.json"), "w", encoding='utf-8') as f:
        json.dump(reacts, f, ensure_ascii=False, indent=4)
    
    # 任务完成后，异步提取用户偏好
    if preference_extractor and should_extract_preferences(data):
        task_data = {
            'task_description': task,
            'actions': actions,
            'reacts': reacts,
            'app_name': app
        }
        preference_extractor.extract_async(task_data)
        logging.info("Submitted preference extraction task")


def parse_planner_response(response_str: str):
    """
    解析 Planner 模型响应，使用统一的 JSON 解析逻辑
    
    Args:
        response_str: 模型返回的响应字符串
    
    Returns:
        解析后的 JSON 对象，或 None 如果解析失败
    """
    parsed = _load_json_from_text(response_str)
    if parsed is None:
        logging.error(f"解析 Planner 响应失败")
        logging.error(f"原始内容: {str(response_str)[:300]}...")
    return parsed

def get_app_package_name(task_description, use_graphrag=False, device_type="Android", use_experience=False):
    """单阶段：本地检索经验，调用模型完成应用选择和任务描述生成。"""
    current_file_path = Path(__file__).resolve()
    current_dir = current_file_path.parent
    default_template_path = current_dir.parent.parent / "utils" /"experience" / "templates-new.json"
    logging.debug("Using template path: %s", default_template_path)

    # 本地检索经验
    search_engine = PromptTemplateSearch(default_template_path)
    experience_content = search_engine.get_experience(task_description, 1)
    logging.debug("检索到的相关经验:\n%s", experience_content)
    if device_type == "Android":
        planner_prompt_template = load_prompt("planner_oneshot.md")
    elif device_type == "Harmony":
        planner_prompt_template = load_prompt("planner_oneshot_harmony.md")
    
    # 检索用户偏好
    user_preferences = {}
    if preference_extractor and preference_extractor.mem:
        user_preferences = retrieve_user_preferences(
            task_description,
            preference_extractor.mem,
            use_graphrag=use_graphrag
        )
        if user_preferences:
            print(f"检索到的用户偏好 (使用{'GraphRAG' if use_graphrag else '向量检索'}):\n{user_preferences}")
        else:
            print("未找到相关用户偏好")
    # 结合上下文
    enhanced_context = combine_context(experience_content, user_preferences)
    # 构建Prompt
    prompt = planner_prompt_template.format(
        task_description=task_description,
        experience_content=enhanced_context
    )
    response_str = planner_client.chat.completions.create(
        model = planner_model,
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ],
    ).choices[0].message.content
    logging.info(f"Planner 响应: \n{response_str}")
    response_json = parse_planner_response(response_str)
    if response_json is None:
        logging.error("无法解析模型响应为 JSON。")
        logging.error(f"原始响应内容: {response_str}")
        raise ValueError("无法解析模型响应为 JSON。")
    app_name = response_json.get("app_name")
    package_name = response_json.get("package_name")
    final_desc = response_json.get("final_task_description", task_description)
    return app_name, package_name, final_desc


def execute_single_task(task_description, device, data_dir, use_experience, use_graphrag, current_device_type, use_qwen3_model, use_e2e=False):
    """
    执行单个任务的通用函数
    
    Args:
        task_description: 任务描述
        device: 设备对象
        data_dir: 数据保存目录
        use_experience: 是否使用经验改写任务
        use_graphrag: 是否使用GraphRAG
        current_device_type: 设备类型
        use_qwen3_model: 是否使用Qwen3模型
        use_e2e: 是否使用e2e模式（skip grounder调用）
    """
    # 调用 planner 获取应用名称和包名
    logging.info(f"Calling planner to get app_name and package_name")
    app_name, package_name, planner_task_description = get_app_package_name(
        task_description, use_graphrag=use_graphrag, device_type=current_device_type, use_experience=use_experience
    )

    # 根据 use_experience 参数决定是否使用 planner 改写的任务描述
    if use_experience:
        logging.info(f"Using experience: using planner-rewritten task description")
        new_task_description = planner_task_description
        logging.info(f"New task description: {new_task_description}")
    else:
        logging.info(f"Not using experience: using original task description")
        new_task_description = task_description

    logging.info(f"Starting task in app: {app_name} (package: {package_name})")
    device.app_start(package_name)
    task_in_app(app_name, task_description, new_task_description, device, data_dir, True, use_qwen3_model, current_device_type, use_e2e)
    time.sleep(APP_STOP_WAIT)  # 等待后再停止应用
    logging.info(f"Stopping app: {app_name} (package: {package_name})")
    device.app_stop(package_name)


# for testing purposes
if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="MobiMind Agent")
    parser.add_argument("--service_ip", type=str, default="localhost", help="Ip for the services (default: localhost)")
    parser.add_argument("--decider_port", type=int, default=8000, help="Port for decider service (default: 8000)")
    parser.add_argument("--grounder_port", type=int, default=8001, help="Port for grounder service (default: 8001)")
    parser.add_argument("--planner_port", type=int, default=8002, help="Port for planner service (default: 8002)")
    parser.add_argument("--user_profile", choices=["on", "off"], default="off", help="Enable user profile memory (default: off)")
    parser.add_argument("--use_graphrag", choices=["on", "off"], default="off", help="Use GraphRAG for user profile preference memory (default: off)")
    parser.add_argument("--clear_memory", action="store_true", help="Force clear all stored user memories and exit")
    parser.add_argument("--device", type=str, default="Android", choices=["Android", "Harmony"], help="Device type: Android or Harmony (default: Android)")
    parser.add_argument("--use_qwen3", choices=["on", "off"], default="on", help="Whether to use Qwen3VL-based model (default: on)")
    parser.add_argument("--use_experience", choices=["on", "off"], default="off", help="Whether to use experience (use planner for task rewriting) (default: off)")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory to save data (default: ./data relative to script location)")
    parser.add_argument("--task_file", type=str, default=None, help="Path to task.json file (default: ./task.json relative to script location)")
    parser.add_argument("--e2e", action="store_true", default=False, help="Enable e2e mode: use e2e_qwen3.md as decider prompt and return coordinates directly from decider (default: False)")
    args = parser.parse_args()

    # 使用命令行参数初始化
    enable_user_profile = (args.user_profile == "on")
    use_graphrag = (args.use_graphrag == "on")
    init(args.service_ip, args.decider_port, args.grounder_port, args.planner_port,
        enable_user_profile=enable_user_profile, use_graphrag=use_graphrag)

    # 如果需要清除记忆，优先执行并退出
    if args.clear_memory:
        if enable_user_profile and preference_extractor and getattr(preference_extractor, 'mem', None):
            try:
                count = preference_extractor.clear_all_memories()
                print(f"Memory cleared. Deleted {count} item(s).")
            except Exception as e:
                print(f"Failed to clear memory: {e}")
        else:
            print("User profile is disabled or memory client not initialized; nothing to clear.")
        raise SystemExit(0)
    # 根据 --device 参数选择设备类型
    if args.device == "Android":
        device = AndroidDevice()
        logging.info("Using AndroidDevice")
    elif args.device == "Harmony":
        device = HarmonyDevice()
        logging.info("Using HarmonyDevice")
    else:
        raise ValueError(f"Unknown device type: {args.device}")
    
    logging.info(f"Connected to device: {args.device}")
    use_qwen3_model = (args.use_qwen3 == "on")
    use_experience = (args.use_experience == "on")
    current_device_type = args.device  # 保存设备类型用于后续使用
    logging.info(f"Use Qwen3 model: {use_qwen3_model}")
    logging.info(f"Use experience (planner task rewriting): {use_experience}")
    logging.info(f"Device type: {current_device_type}")
    logging.info(f"Use E2E mode: {args.e2e}")
    # 配置数据保存目录
    if args.data_dir:
        data_base_dir = args.data_dir
        logging.info(f"Using custom data directory: {data_base_dir}")
    else:
        data_base_dir = os.path.join(os.path.dirname(__file__), 'data')
        logging.info(f"Using default data directory: {data_base_dir}")
    
    if not os.path.exists(data_base_dir):
        os.makedirs(data_base_dir)
        logging.info(f"Created data directory: {data_base_dir}")

    # 读取任务列表
    if args.task_file:
        task_json_path = args.task_file
        logging.info(f"Using custom task file: {task_json_path}")
    else:
        task_json_path = os.path.join(os.path.dirname(__file__), "task.json")
    with open(task_json_path, "r", encoding="utf-8") as f:
        task_list = json.load(f)
    
    # print(task_list)

    for task_item in task_list:
        # 支持两种格式：
        # 1. 简单字符串格式: ["task1", "task2", ...]
        # 2. 结构化格式: {"app": "app_name", "type": "type_name", "tasks": ["task1", "task2", ...]}
        
        if isinstance(task_item, dict):
            # 新格式：结构化任务
            app_name_from_file = task_item.get("app")
            task_type = task_item.get("type", "default")
            tasks_list = task_item.get("tasks", [])
            
            
            # 遍历该应用和类型下的所有任务
            for task_index, task_description in enumerate(tasks_list, 1):
                # 创建 data_dir: data_base_dir/app/type/task_index
                data_dir = os.path.join(data_base_dir, app_name_from_file, task_type, str(task_index))
                os.makedirs(data_dir, exist_ok=True)
                logging.info(f"Processing task {task_index} of {app_name_from_file}/{task_type}: {task_description}")

                
                execute_single_task(task_description, device, data_dir, use_experience, use_graphrag, current_device_type, use_qwen3_model, args.e2e)
        else:
            # 旧格式：简单任务列表
            existing_dirs = [d for d in os.listdir(data_base_dir) if os.path.isdir(os.path.join(data_base_dir, d)) and d.isdigit()]
            if existing_dirs:
                data_index = max(int(d) for d in existing_dirs) + 1
            else:
                data_index = 1
            data_dir = os.path.join(data_base_dir, str(data_index))
            os.makedirs(data_dir, exist_ok=True)
            task_description = task_item
            
            execute_single_task(task_description, device, data_dir, use_experience, use_graphrag, current_device_type, use_qwen3_model, args.e2e)
    
    # 等待所有偏好提取任务完成
    if preference_extractor and hasattr(preference_extractor, 'executor'):
        logging.info("Waiting for all preference extraction tasks to complete...")
        preference_extractor.executor.shutdown(wait=True)
        logging.info("All preference extraction tasks completed")
