from openai import OpenAI
import uiautomator2 as u2
import base64
from PIL import Image
import json
import io
import logging
from abc import ABC, abstractmethod
import time
import re
import os
import argparse
from PIL import Image, ImageDraw, ImageFont
import textwrap
import cv2
import numpy as np
from utils.local_experience import PromptTemplateSearch 
from pathlib import Path
import sys
from hmdriver2.driver import Driver
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

MAX_STEPS = 35

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
        time.sleep(1)
        if not self.d.app_wait(package_name, timeout=10):
            raise RuntimeError(f"Failed to start package '{package_name}'")

    def app_stop(self, package_name):
        self.d.app_stop(package_name)

    def screenshot(self, path):
        self.d.screenshot(path)

    def click(self, x, y):
        self.d.click(x, y)
        time.sleep(0.5)

    def clear_input(self):
    # 按下全选（需要 Android 支持 keyevent META_CTRL_ON）
        self.d.shell(['input', 'keyevent', 'KEYCODE_MOVE_END'])
        self.d.shell(['input', 'keyevent', 'KEYCODE_MOVE_HOME'])
        self.d.shell(['input', 'keyevent', 'KEYCODE_DEL'])

    def input(self, text):
        current_ime = self.d.current_ime()
        self.d.shell(['settings', 'put', 'secure', 'default_input_method', 'com.android.adbkeyboard/.AdbIME'])
        time.sleep(0.5)
        # add clear text command, depending on 'ADB Keyboard'
        self.d.shell(['am', 'broadcast', '-a', 'ADB_CLEAR_TEXT'])
        time.sleep(0.2)

        charsb64 = base64.b64encode(text.encode('utf-8')).decode('utf-8')
        self.d.shell(['am', 'broadcast', '-a', 'ADB_INPUT_B64', '--es', 'msg', charsb64])
        time.sleep(0.5)
        self.d.shell(['settings', 'put', 'secure', 'default_input_method', current_ime])
        time.sleep(0.5)

    def swipe(self, direction, scale=0.5):
        # self.d.swipe_ext(direction, scale)
        self.d.swipe_ext(direction=direction, scale=scale)

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
        time.sleep(0.5)

    def input(self, text):
        self.d.shell("uitest uiInput keyEvent 2072 2017")
        self.d.press_key(2071)
        self.d.input_text(text)

    def swipe(self, direction, scale=0.5):
        # self.d.swipe_ext(direction, scale=scale)
        if direction.lower() == "up":
            self.d.swipe(0.5,0.7,0.5,0.3,speed=2000)
        elif direction.lower() == "down":
            self.d.swipe(0.5,0.3,0.5,0.7,speed=2000)
        elif direction.lower() == "left":
            self.d.swipe(0.7,0.5,0.3,0.5,speed=2000)
        elif direction.lower() == "right":
            self.d.swipe(0.3,0.5,0.7,0.5,speed=2000)

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
    
# 截图缩放比例
factor = 0.5



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
    if not response_str or not isinstance(response_str, str):
        logging.error(f"Invalid response: {response_str}")
        raise ValueError(f"无效的响应格式: {response_str}")
    
    response_str = response_str.strip()
    
    # 首先尝试直接解析 JSON（guided decoding 输出纯 JSON）
    try:
        return json.loads(response_str)
    except json.JSONDecodeError:
        pass
    
    # 如果直接解析失败，尝试提取 JSON 部分（兼容非 guided decoding 的情况）
    try:
        # 方法1: 提取 ```json ... ``` 代码块
        json_match = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', response_str, re.MULTILINE)
        if json_match:
            json_str = json_match.group(1).strip()
            return json.loads(json_str)
        
        # 方法2: 提取 ``` ... ``` 代码块
        json_match = re.search(r'```\s*(\{[\s\S]*?\})\s*```', response_str, re.MULTILINE)
        if json_match:
            json_str = json_match.group(1).strip()
            return json.loads(json_str)
        
        # 方法3: 查找最外层的花括号包围的 JSON
        # 这种方法需要更仔细的处理，避免误匹配嵌套结构
        start_idx = response_str.find('{')
        if start_idx != -1:
            # 从第一个 { 开始，找到匹配的 }
            brace_count = 0
            for i in range(start_idx, len(response_str)):
                if response_str[i] == '{':
                    brace_count += 1
                elif response_str[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_str = response_str[start_idx:i+1]
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError:
                            continue
        
        # 如果都失败了
        logging.error(f"无法在响应中找到有效的 JSON")
        logging.error(f"原始响应: {response_str[:200]}...")
        raise ValueError(f"无法解析 JSON 响应，响应格式不正确")
        
    except json.JSONDecodeError as e:
        logging.error(f"JSON 解析失败: {e}")
        logging.error(f"原始响应: {response_str[:200]}...")
        raise ValueError(f"无法解析 JSON 响应: {e}")

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

def create_swipe_visualization(data_dir, image_index, direction):
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


# 预处理 decider_response_str，增强健壮性
def robust_json_loads(s):
    """
    健壮的 JSON 加载函数
    支持 guided decoding 和普通模式的混合输出
    """
    if not isinstance(s, str):
        s = str(s)
    
    s = s.strip()
    
    # 首先尝试直接解析 JSON（guided decoding 纯输出）
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    
    # 提取 ```json ... ``` 代码块
    codeblock = re.search(r"```json\s*([\s\S]*?)\s*```", s, re.MULTILINE)
    if codeblock:
        s = codeblock.group(1).strip()
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
    
    # 替换中文省略号为英文 ...
    s = s.replace("…", "...")
    
    # 尝试再次解析
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    
    # 尝试提取 JSON 对象（从第一个 { 到最后一个 }）
    start_idx = s.find('{')
    if start_idx != -1:
        brace_count = 0
        for i in range(start_idx, len(s)):
            if s[i] == '{':
                brace_count += 1
            elif s[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_str = s[start_idx:i+1]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        continue
    
    # 解析失败，记录错误
    logging.error(f"解析 decider_response_str 失败")
    logging.error(f"原始内容: {s[:300]}...")
    raise ValueError(f"无法解析 JSON 响应: 响应格式不正确")

def task_in_app(app, old_task, task, device, data_dir, bbox_flag=True, use_qwen3=True, device_type="Android"):
    history = []
    actions = []
    reacts = []

    if use_qwen3:
        grounder_prompt_template_bbox = load_prompt("grounder_qwen3_bbox.md")
        grounder_prompt_template_no_bbox = load_prompt("grounder_qwen3_coordinates.md")
        # decider_prompt_template = load_prompt("decider_qwen3.md")
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

        
        decider_prompt = decider_prompt_template.format(
            task=task,
            history=history_str
        )
        logging.info(f"Decider prompt: \n{decider_prompt}")

        decider_start_time = time.time()
        # --- 修改 API 调用 ---
        # vLLM 将会强制输出一个符合 ActionPlan 结构的 JSON 字符串
        # 若响应超时或者返回结果解析失败，则进行重试
        temperature = 0.0
        for attempt in range(5):  # 最多重试5次
        # while True:
            try:
                decider_start_time = time.time()
                decider_response_str = decider_client.chat.completions.create(
                    model=decider_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{screenshot_resize}"}},
                                {"type": "text", "text": decider_prompt},
                            ]
                        }
                    ],
                    temperature=temperature,

                    timeout=30,
                    max_tokens=256,
                ).choices[0].message.content
                decider_end_time = time.time()
                logging.info(f"[evaluation] Decider time taken: {decider_end_time - decider_start_time} seconds")
                logging.info(f"Decider response: \n{decider_response_str}")
                decider_response = robust_json_loads(decider_response_str)
                converted_item = {
                    "reasoning": decider_response["reasoning"],
                    "function": {
                        "name": decider_response["action"],
                        "parameters": decider_response["parameters"]
                    }
                }
                break  # 成功获取响应，跳出重试循环
            except Exception as e:
                temperature = 0.1 + attempt * 0.1  # 每次重试时增加温度，增加多样性
                logging.error(f"Decider 调用失败: {e}, 正在重试 temperature={temperature}...")
                time.sleep(2)

        
        reacts.append(converted_item)
        action = decider_response["action"]

        # compute image index for this loop iteration (1-based)
        image_index = len(actions) + 1
        current_dir = os.getcwd()
        current_image = ""
        if device_type == "Android":
            img_path = os.path.join(current_dir, f"screenshot-Android.jpg")
            save_path = os.path.join(data_dir, f"{image_index}.jpg")
            current_image = f"screenshot-Android.jpg"
        else:
            img_path = os.path.join(current_dir, f"screenshot-Harmony.jpg")
            save_path = os.path.join(data_dir, f"{image_index}.jpg")
            current_image = f"screenshot-Harmony.jpg"
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
            hierarchy = device.dump_hierarchy()
            # Android设备保存为XML格式
            hierarchy_path = os.path.join(data_dir, f"{image_index}.xml")
            with open(hierarchy_path, "w", encoding="utf-8") as f:
                f.write(hierarchy)
        else:
            hierarchy = device.dump_hierarchy()
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
            grounder_prompt = (grounder_prompt_template_bbox if bbox_flag else grounder_prompt_template_no_bbox).format(reasoning=reasoning, description=target_element)

            # 重试5次获取grounder响应，同时调整temperature
            temperature = 0.0
            for attempt in range(5):
                try:
                    grounder_start_time = time.time()
                    grounder_response_str = grounder_client.chat.completions.create(
                        model=grounder_model,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{screenshot_resize}"}},
                                    {"type": "text", "text": grounder_prompt},
                                ]
                            }
                        ],
                        temperature=temperature,
                        timeout=30,
                        max_tokens=128,
                    ).choices[0].message.content
                    grounder_end_time = time.time()
                    logging.info(f"[evaluation] Grounder time taken: {grounder_end_time - grounder_start_time} seconds")
                    logging.info(f"Grounder response: \n{grounder_response_str}")
                    # grounder_response = json.loads(grounder_response_str)
                    grounder_response = parse_json_response(grounder_response_str)
                    break  # 成功获取响应，跳出重试循环
                except Exception as e:
                    temperature = 0.1 + attempt * 0.1  # 每次重试时增加温度，增加多样性
                    logging.error(f"Grounder 调用失败: {e}, 正在重试 temperature={temperature}...")
                    time.sleep(2)

            if(bbox_flag):
                # 直接尝试获取含有bbox的字段，而不要求完全匹配
                bbox = None
                for key in grounder_response:
                    if key.lower() in ["bbox", "bbox_2d", "bbox-2d", "bbox_2D", "bbox2d", "bbox_2009"]:
                        bbox = grounder_response[key]
                        break
                

                # 如果使用 Qwen3 模型，进行坐标转换
                if use_qwen3:
                    bbox = convert_qwen3_coordinates_to_absolute(bbox, img.width, img.height, is_bbox=True)
                    x1, y1, x2, y2 = bbox
                else:
                    x1, y1, x2, y2 = [int(coord/factor) for coord in bbox]

                
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
                history.append(decider_response_str)

                current_dir = os.getcwd()
                img_path = os.path.join(current_dir, current_image)
                save_path = os.path.join(data_dir, f"{image_index}_highlighted.jpg")
                img = Image.open(img_path)
                draw = ImageDraw.Draw(img)
                font = ImageFont.truetype("msyh.ttf", 40)
                text = f"CLICK [{position_x}, {position_y}]"
                text = textwrap.fill(text, width=20)
                text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:]
                draw.text((img.width / 2 - text_width / 2, 0), text, fill="red", font=font)
                img.save(save_path)

                # 拉框
                bounds_path = os.path.join(data_dir, f"{image_index}_bounds.jpg")
                img_bounds = Image.open(save_path)
                draw_bounds = ImageDraw.Draw(img_bounds)
                draw_bounds.rectangle([x1, y1, x2, y2], outline='red', width=5)
                img_bounds.save(bounds_path)

                # 画点
                cv2image = cv2.imread(bounds_path)
                if cv2image is not None:
                    # 在点击位置画圆点
                    cv2.circle(cv2image, (position_x, position_y), 15, (0, 255, 0), -1)  # 绿色实心圆
                    # 保存带点击点的图像
                    click_point_path = os.path.join(data_dir, f"{image_index}_click_point.jpg")
                    cv2.imwrite(click_point_path, cv2image)

            else:
                coordinates = grounder_response["coordinates"]
                if use_qwen3:
                    coordinates = convert_qwen3_coordinates_to_absolute(coordinates, img.width, img.height, is_bbox=False)
                    x, y = coordinates
                else:
                    x, y = [int(coord / factor) for coord in coordinates]
                device.click(x, y)
                actions.append({
                    "type": "click",
                    "position_x": x,
                    "position_y": y,
                    "action_index": image_index
                })
                history.append(decider_response_str)
          

        elif action == "input":
            text = decider_response["parameters"]["text"]
            device.input(text)
            actions.append({
                "type": "input",
                "text": text,
                "action_index": image_index
            })
            
            history.append(decider_response_str)

        elif action == "swipe":
            direction = decider_response["parameters"]["direction"]
            direction = direction.upper()

            if direction == "DOWN":
                device.swipe(direction.lower(), 0.4)
                # record the swipe as an action (index only)
                actions.append({
                    "type": "swipe",
                    "press_position_x": None,
                    "press_position_y": None,
                    "release_position_x": None,
                    "release_position_y": None,
                    "direction": direction.lower(),
                    "action_index": image_index
                })
                
                history.append(decider_response_str)
                
                # 为向下滑动创建可视化
                create_swipe_visualization(data_dir, image_index, direction.lower())
                continue

            if direction in ["UP", "LEFT", "RIGHT"]:
                device.swipe(direction.lower(), 0.4)
                actions.append({
                    "type": "swipe",
                    "press_position_x": None,
                    "press_position_y": None,
                    "release_position_x": None,
                    "release_position_y": None,
                    "direction": direction.lower(),
                    "action_index": image_index
                })
                
                history.append(decider_response_str)
                
                # 为滑动创建可视化
                create_swipe_visualization(data_dir, image_index, direction.lower())

            else:
                raise ValueError(f"Unknown swipe direction: {direction}")
        elif action == "wait":
            print("Waiting for a while...")
            actions.append({
                "type": "wait",
                "action_index": image_index
            })
            history.append(decider_response_str)
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

    # 尝试匹配 ```json ... ``` 代码块
    pattern = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)
    match = pattern.search(response_str)

    json_str = None
    if match:
        json_str = match.group(1)
    else:
        # 如果没有代码块，直接当成 JSON
        json_str = response_str.strip()

    try:
        data = json.loads(json_str)
        return data
    except json.JSONDecodeError as e:
        logging.error(f"解析 JSON 失败: {e}\n内容为:\n{json_str}")
        return None

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


def execute_single_task(task_description, device, data_dir, use_experience, use_graphrag, current_device_type, use_qwen3_model):
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
    task_in_app(app_name, task_description, new_task_description, device, data_dir, True, use_qwen3_model, current_device_type)
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
    parser.add_argument("--user_profile", choices=["on", "off"], default="off", help="Enable user profile memory (on/off). Default: off")
    parser.add_argument("--use_graphrag", choices=["on", "off"], default="off", help="Use GraphRAG for user profile preference memory (on/off). Default: off")
    parser.add_argument("--clear_memory", action="store_true", help="Force clear all stored user memories and exit")
    parser.add_argument("--device", type=str, default="Android", choices=["Android", "Harmony"], help="Device type: Android or Harmony (default: Android)")
    parser.add_argument("--use_qwen3", action="store_true", default=True, help="Whether to use Qwen3 model (default: True)")
    parser.add_argument("--use_experience", action="store_true", default=False, help="Whether to use experience (use planner for task rewriting) (default: False)")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory to save data (default: ./data relative to script location)")
    parser.add_argument("--task_file", type=str, default=None, help="Path to task.json file (default: ./task.json relative to script location)")
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
    use_qwen3_model = args.use_qwen3
    use_experience = args.use_experience
    current_device_type = args.device  # 保存设备类型用于后续使用
    logging.info(f"Use Qwen3 model: {use_qwen3_model}")
    logging.info(f"Use experience (planner task rewriting): {use_experience}")
    logging.info(f"Device type: {current_device_type}")
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
                
                execute_single_task(task_description, device, data_dir, use_experience, use_graphrag, current_device_type, use_qwen3_model)
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
            
            execute_single_task(task_description, device, data_dir, use_experience, use_graphrag, current_device_type, use_qwen3_model)
    
    # 等待所有偏好提取任务完成
    if preference_extractor and hasattr(preference_extractor, 'executor'):
        logging.info("Waiting for all preference extraction tasks to complete...")
        preference_extractor.executor.shutdown(wait=True)
        logging.info("All preference extraction tasks completed")
