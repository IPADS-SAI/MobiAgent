#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MobiAgent Mobile Standalone
手机端独立部署版本 - 单文件，无本地包依赖

依赖（Termux）:
  pkg install python android-tools libjpeg-turbo
  pip install openai Pillow
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
import subprocess
from datetime import datetime

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)

# ============ 提示词（内联自 prompts/decider_qwen3_e2e.py）============

DECIDER_SYSTEM_PROMPT = """You are a phone-use AI agent.

### Action Space
Your action space includes:
- Name: click, Parameters: target_element (a high-level description of the UI element to click), bbox (a bounding box of the target element, [x1, y1, x2, y2]).
- Name: swipe, Parameters: direction (one of UP, DOWN, LEFT, RIGHT), start_coords (the starting coordinate [x, y]), end_coords (the ending coordinate [x, y]).
- Name: click_input, Parameters: target_element (a high-level description of the UI element to click), text (the text to input), bbox (a bounding box of the target element, [x1, y1, x2, y2]).
- Name: input, Parameters: text (the text to input).
- Name: open_app, Parameters: app_name (the name of the application to open).
- Name: press_home, Parameters: (no parameters, returns to the home screen).
- Name: press_back, Parameters: (no parameters, goes back to the previous screen).
- Name: wait, Parameters: (no parameters, will wait for 1 second).
- Name: done, Parameters: status (the completion status of the current task, one of `success`, `suspended` and `failed`).

### Response Format
Your output should be a JSON object with the following format:
{
  "reasoning": "Your reasoning here",
  "action": "The next action (one of click, click_input, input, swipe, open_app, press_home, press_back, wait, done)",
  "parameters": {"param1": "value1", "param2": "value2", ...}
}
"""

DECIDER_USER_PROMPT = """
### Current Task
"{task}"
### Action History
The sequence of actions you have already taken:
{history}
### Constraints
- If the screen has not changed after your last action, do not repeat the exact same action. Try a different method or slightly adjust coordinates.
- If the task is completed, verify the result before outputting 'done'.
"""

DECIDER_CURRENT_STEP_PROMPT = """
Please provide the next action based on the screenshot and your action history. You should do careful reasoning before providing the action."""

# ============ 常数 ============

MAX_STEPS = 15
MAX_RETRIES = 5
TEMP_INCREMENT = 0.1
INITIAL_TEMP = 0.0
API_TIMEOUT = 30
DECIDER_MAX_TOKENS = 256
DEVICE_WAIT_TIME = 0.5
APP_STOP_WAIT = 3
SCREENSHOT_FACTOR = 0.5

SWIPE_V_START = 0.3
SWIPE_V_END = 0.7
SWIPE_H_START = 0.3
SWIPE_H_END = 0.7

# ============ Planner 提示词（内联自 prompts/planner_oneshot.md）============

PLANNER_PROMPT = """## 角色定义
你是一个任务规划专家，负责理解用户意图，选择最合适的应用，并生成一个结构化、可执行的最终任务描述。

## 已知输入
1. 原始用户任务描述："{task_description}"

## 可用应用列表
以下是可用的应用及其包名：
- 支付宝: com.eg.android.AlipayGphone
- 微信: com.tencent.mm
- QQ: com.tencent.mobileqq
- 新浪微博: com.sina.weibo
- 今日头条: com.ss.android.article.news
- [外卖默认]饿了么: me.ele
- 美团: com.sankuai.meituan
- bilibili: tv.danmaku.bili
- 爱奇艺: com.qiyi.video
- 腾讯视频: com.tencent.qqlive
- 优酷: com.youku.phone
- [购物默认]淘宝: com.taobao.taobao
- 京东: com.jingdong.app.mall
- [旅行、酒店、机票默认]携程: ctrip.android.view
- 同城: com.tongcheng.android
- 飞猪: com.taobao.trip
- 去哪儿: com.Qunar
- 华住会: com.htinns
- 知乎: com.zhihu.android
- 小红书: com.xingin.xhs
- QQ音乐: com.tencent.qqmusic
- 网易云音乐: com.netease.cloudmusic
- 酷狗音乐: com.kugou.android
- 抖音: com.ss.android.ugc.aweme
- [导航、打车默认]高德地图: com.autonavi.minimap
- 咸鱼: com.taobao.idlefish
- 华为商城: com.vmall.client
- 华为音乐: com.huawei.music
- 华为视频: com.huawei.himovie
- 华为应用市场: com.huawei.appmarket
- 拼多多: com.xunmeng.pinduoduo
- 大众点评: com.dianping.v1
- 浏览器: com.microsoft.emmx
- 同程旅行: com.tongcheng.android
- 滴滴出行: com.sdu.didi.psnger
- 快手: com.smile.gifmaker
- 备忘录: com.huawei.notepad

## 任务要求
1. **选择应用**：根据用户任务描述，从"可用应用列表"中选择最合适的应用，未提及指定APP时选择该类任务默认应用。
2. **生成最终任务描述**：将用户的原始任务描述转化为一个详细、完整、结构化的任务描述。
   - **语义保持一致**：最终描述必须与用户原始意图完全相同。
   - **自然表达**：输出的描述应符合中文自然语言习惯，避免冗余。

## 输出格式
请严格按照以下JSON格式输出，不要包含任何额外内容或注释：
```json
{{
  "reasoning": "简要说明你为什么选择这个应用，以及你是如何生成最终任务描述的。",
  "app_name": "选择的应用名称",
  "package_name": "所选应用的包名",
  "final_task_description": "最终生成的完整、结构化的任务描述文本。"
}}
```"""

# ============ 全局客户端 ============

decider_client = None
decider_model = ""
planner_client = None
planner_model = ""


def init_clients(service_ip, decider_port, planner_port):
    global decider_client, planner_client
    decider_client = OpenAI(api_key="mobiagent-key", base_url=f"http://{service_ip}:{decider_port}/v1")
    planner_client = OpenAI(api_key="mobiagent-key", base_url=f"http://{service_ip}:{planner_port}/v1")
    logger.info(f"[OK] 已连接到 Decider 服务: {service_ip}:{decider_port}")
    logger.info(f"[OK] 已连接到 Planner 服务: {service_ip}:{planner_port}")


# ============ 设备控制（纯 ADB subprocess，无需 uiautomator2）============

class AndroidDevice:
    """
    通过 subprocess 调用 adb 命令控制设备。
    无需安装 uiautomator2 Python 包及其后端 APK。
    """

    APP_PACKAGES = {
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
        "大众点评": "com.dianping.v1",
        "小红书": "com.xingin.xhs",
        "浏览器": "com.microsoft.emmx",
    }

    def __init__(self, adb_endpoint="127.0.0.1:5555"):
        self._s = ["-s", adb_endpoint] if adb_endpoint else []
        result = self._run("get-state")
        if result.returncode != 0 or "device" not in result.stdout:
            raise RuntimeError(f"ADB 连接失败: {adb_endpoint}\n{result.stderr}")
        logger.info(f"[OK] ADB 已连接: {adb_endpoint}")

    def _run(self, *args, binary=False):
        cmd = ["adb"] + self._s + list(args)
        return subprocess.run(
            cmd,
            capture_output=True,
            text=not binary,
            encoding=None if binary else "utf-8",
            errors=None if binary else "replace",
        )

    def _shell(self, *args):
        return self._run("shell", *args)

    def screenshot(self, path="screenshot-Android.png"):
        # exec-out 直接获取二进制 PNG，无需 pull 文件
        result = self._run("exec-out", "screencap", "-p", binary=True)
        with open(path, "wb") as f:
            f.write(result.stdout)
        return path

    def click(self, x, y):
        self._shell("input", "tap", str(int(x)), str(int(y)))
        time.sleep(DEVICE_WAIT_TIME)

    def input(self, text):
        """使用 ADB Keyboard 输入文本（支持中文）"""
        current_ime = self._shell(
            "settings", "get", "secure", "default_input_method"
        ).stdout.strip()
        self._shell("settings", "put", "secure", "default_input_method",
                    "com.android.adbkeyboard/.AdbIME")
        time.sleep(DEVICE_WAIT_TIME)
        self._shell("am", "broadcast", "-a", "ADB_CLEAR_TEXT")
        time.sleep(0.2)
        charsb64 = base64.b64encode(text.encode("utf-8")).decode("utf-8")
        self._shell("am", "broadcast", "-a", "ADB_INPUT_B64", "--es", "msg", charsb64)
        time.sleep(DEVICE_WAIT_TIME)
        self._shell("settings", "put", "secure", "default_input_method", current_ime)
        self._shell("input", "keyevent", "KEYCODE_ENTER")

    def swipe_with_coords(self, start_x, start_y, end_x, end_y):
        self._shell("input", "swipe",
                    str(int(start_x)), str(int(start_y)),
                    str(int(end_x)), str(int(end_y)), "200")
        time.sleep(DEVICE_WAIT_TIME)

    def keyevent(self, key):
        self._shell("input", "keyevent", str(key))

    def start_app(self, app_name):
        package = self.APP_PACKAGES.get(app_name, app_name)
        self._shell("am", "force-stop", package)
        time.sleep(0.5)
        self._shell("monkey", "-p", package,
                    "-c", "android.intent.category.LAUNCHER", "1")
        time.sleep(3)
        if not self._wait_app(package, timeout=10):
            raise RuntimeError(f"启动应用失败: {app_name} ({package})")

    def app_start(self, package_name):
        self._shell("monkey", "-p", package_name,
                    "-c", "android.intent.category.LAUNCHER", "1")
        time.sleep(DEVICE_WAIT_TIME * 2)

    def app_stop(self, package_name):
        self._shell("am", "force-stop", package_name)

    def _wait_app(self, package_name, timeout=10):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._shell("pidof", package_name).stdout.strip():
                return True
            time.sleep(0.5)
        return False

    def dump_hierarchy(self):
        """使用 Android 内置 uiautomator dump，无需 uiautomator2 APK。"""
        self._shell("uiautomator", "dump", "/sdcard/_wd.xml")
        return self._shell("cat", "/sdcard/_wd.xml").stdout


# ============ 工具函数 ============

def get_screenshot_b64(device):
    path = device.screenshot()
    img = Image.open(path)
    # 保留原始尺寸用于坐标转换（adb 需要实际屏幕坐标）
    original = img
    resized = img.resize(
        (int(img.width * SCREENSHOT_FACTOR), int(img.height * SCREENSHOT_FACTOR)),
        Image.Resampling.LANCZOS,
    )
    if resized.mode != "RGB":
        resized = resized.convert("RGB")
    buf = io.BytesIO()
    resized.save(buf, format="JPEG")
    # 返回 base64、缩放图（用于保存）、原始图（用于坐标转换）
    return base64.b64encode(buf.getvalue()).decode("utf-8"), resized, original


def _load_json_from_text(raw_text):
    if not isinstance(raw_text, str):
        raw_text = str(raw_text) if raw_text is not None else ""
    text = raw_text.strip()

    def _try(s):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return None

    parsed = _try(text)
    if parsed is not None:
        return parsed

    for pat in [r"```json\s*([\s\S]*?)\s*```", r"```\s*([\s\S]*?)\s*```"]:
        m = re.search(pat, text, re.MULTILINE)
        if m:
            parsed = _try(m.group(1).strip())
            if parsed is not None:
                return parsed

    normalized = text.replace("…", "...")
    if normalized != text:
        parsed = _try(normalized)
        if parsed is not None:
            return parsed
        text = normalized

    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    parsed = _try(text[start: i + 1])
                    if parsed is not None:
                        return parsed
    return None


def validate_action_parameters(resp):
    action = resp.get("action")
    params = resp.get("parameters", {})
    if not action:
        raise ValueError("Missing field: 'action'")
    if not resp.get("reasoning"):
        raise ValueError("Missing field: 'reasoning'")
    if action == "click":
        if not params.get("target_element"):
            raise ValueError("click: missing 'target_element'")
        if "bbox" not in params or params["bbox"] is None:
            raise ValueError("E2E click: missing 'bbox'")
    elif action == "click_input":
        if not params.get("target_element"):
            raise ValueError("click_input: missing 'target_element'")
        if not params.get("bbox"):
            raise ValueError("click_input: missing 'bbox'")
        if "text" not in params:
            raise ValueError("click_input: missing 'text'")
    elif action == "input":
        if "text" not in params:
            raise ValueError("input: missing 'text'")
    elif action == "swipe":
        d = params.get("direction", "")
        if d.upper() not in ("UP", "DOWN", "LEFT", "RIGHT"):
            raise ValueError(f"swipe: invalid direction '{d}'")
    elif action == "done":
        if not params.get("status"):
            raise ValueError("done: missing 'status'")
    elif action == "open_app":
        if not params.get("app_name"):
            raise ValueError("open_app: missing 'app_name'")
    elif action not in ("press_home", "press_back", "wait", "long_press"):
        raise ValueError(f"Unknown action: '{action}'")


def call_decider_with_retry(messages):
    temperature = INITIAL_TEMP
    for attempt in range(MAX_RETRIES):
        try:
            start = time.time()
            raw = decider_client.chat.completions.create(
                model=decider_model,
                messages=messages,
                temperature=temperature,
                timeout=API_TIMEOUT,
                max_tokens=DECIDER_MAX_TOKENS,
            ).choices[0].message.content
            logger.info(f"[Decider] {time.time() - start:.2f}s | {raw[:200]}")
            parsed = _load_json_from_text(raw)
            if parsed is None:
                raise ValueError("无法解析 JSON 响应")
            validate_action_parameters(parsed)
            return parsed
        except Exception as e:
            temperature = INITIAL_TEMP + (attempt + 1) * TEMP_INCREMENT
            logger.warning(f"[Decider] 第{attempt + 1}次失败: {e}, 重试 temp={temperature:.1f}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
            else:
                raise
    raise RuntimeError("Decider 重试耗尽")


def convert_qwen3_coords(bbox_or_pt, img_w, img_h, is_bbox=True):
    """Qwen3 坐标系（0-1000）→ 绝对像素坐标。"""
    if is_bbox:
        x1, y1, x2, y2 = bbox_or_pt
        return [
            int(x1 / 1000 * img_w), int(y1 / 1000 * img_h),
            int(x2 / 1000 * img_w), int(y2 / 1000 * img_h),
        ]
    x, y = bbox_or_pt
    return [int(x / 1000 * img_w), int(y / 1000 * img_h)]


def compute_swipe_positions(direction, w, h):
    d = direction.upper()
    if d == "UP":
        return 0.5 * w, SWIPE_V_END * h, 0.5 * w, SWIPE_V_START * h
    if d == "DOWN":
        return 0.5 * w, SWIPE_V_START * h, 0.5 * w, SWIPE_V_END * h
    if d == "LEFT":
        return SWIPE_H_END * w, 0.5 * h, SWIPE_H_START * w, 0.5 * h
    if d == "RIGHT":
        return SWIPE_H_START * w, 0.5 * h, SWIPE_H_END * w, 0.5 * h
    raise ValueError(f"Unknown swipe direction: {direction}")


def build_decider_messages(task, history, screenshot_b64):
    history_str = (
        "\n".join(f"{i}. {h}" for i, h in enumerate(history, 1))
        if history else "(No history)"
    )
    return [
        {"role": "system", "content": DECIDER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": DECIDER_USER_PROMPT.format(task=task, history=history_str)},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{screenshot_b64}"}},
                {"type": "text", "text": DECIDER_CURRENT_STEP_PROMPT},
            ],
        },
    ]


# ============ 动作执行器 ============

def _record(actions, history, resp, action_record):
    actions.append(action_record)
    history.append(json.dumps(resp, ensure_ascii=False))


def execute_action(resp, device, img, actions, history, image_index):
    action = resp["action"]
    params = resp.get("parameters", {})

    if action == "click":
        bbox = convert_qwen3_coords(params["bbox"], img.width, img.height, is_bbox=True)
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        device.click(cx, cy)
        _record(actions, history, resp, {
            "type": "click", "position_x": cx, "position_y": cy,
            "bounds": bbox, "action_index": image_index,
        })

    elif action == "click_input":
        bbox = convert_qwen3_coords(params["bbox"], img.width, img.height, is_bbox=True)
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        device.click(cx, cy)
        device.input(params["text"])
        _record(actions, history, resp, {
            "type": "click_input", "position_x": cx, "position_y": cy,
            "bounds": bbox, "text": params["text"], "action_index": image_index,
        })

    elif action == "input":
        device.input(params["text"])
        _record(actions, history, resp, {
            "type": "input", "text": params["text"], "action_index": image_index,
        })

    elif action == "swipe":
        direction = params["direction"].upper()
        sc = params.get("start_coords")
        ec = params.get("end_coords")
        if sc and ec:
            sx, sy = convert_qwen3_coords(sc, img.width, img.height, is_bbox=False)
            ex, ey = convert_qwen3_coords(ec, img.width, img.height, is_bbox=False)
        else:
            sx, sy, ex, ey = compute_swipe_positions(direction, img.width, img.height)
        device.swipe_with_coords(sx, sy, ex, ey)
        _record(actions, history, resp, {
            "type": "swipe", "direction": direction.lower(),
            "press_position_x": sx, "press_position_y": sy,
            "release_position_x": ex, "release_position_y": ey,
            "action_index": image_index,
        })

    elif action == "open_app":
        app_name = params["app_name"]
        try:
            device.start_app(app_name)
        except Exception:
            device.app_start(app_name)
        _record(actions, history, resp, {
            "type": "open_app", "app_name": app_name, "action_index": image_index,
        })

    elif action == "press_home":
        device.keyevent("home")
        _record(actions, history, resp, {"type": "press_home", "action_index": image_index})

    elif action == "press_back":
        device.keyevent("back")
        _record(actions, history, resp, {"type": "press_back", "action_index": image_index})

    elif action == "wait":
        time.sleep(DEVICE_WAIT_TIME * 2)
        _record(actions, history, resp, {"type": "wait", "action_index": image_index})

    else:
        raise ValueError(f"Unknown action: {action}")


# ============ Planner ============

def plan_task(task_description):
    """
    调用 Planner 模型分析任务，返回 (app_name, package_name, final_task_description)。
    对应 mobiagent.py 的 get_app_package_name（去掉 experience/graphrag 依赖）。
    """
    prompt = PLANNER_PROMPT.format(task_description=task_description)
    logger.info(f"[Planner] 分析任务: {task_description}")
    response_str = planner_client.chat.completions.create(
        model=planner_model,
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    ).choices[0].message.content
    logger.info(f"[Planner] 响应: {response_str[:300]}")

    parsed = _load_json_from_text(response_str)
    if parsed is None:
        raise ValueError(f"Planner 响应无法解析为 JSON: {response_str[:200]}")

    app_name = parsed.get("app_name", "unknown")
    package_name = parsed.get("package_name", "")
    final_desc = parsed.get("final_task_description", task_description)
    reasoning = parsed.get("reasoning", "")
    logger.info(f"[Planner] 选择应用: {app_name} ({package_name})")
    logger.info(f"[Planner] 推理: {reasoning}")
    logger.info(f"[Planner] 最终任务描述: {final_desc}")
    return app_name, package_name, final_desc


# ============ 主任务循环 ============

def run_task(app_name, task_desc, device, data_dir):
    os.makedirs(data_dir, exist_ok=True)
    history, actions, reacts = [], [], []
    logger.info(f"[START] 任务: {task_desc}")

    for step in range(1, MAX_STEPS + 1):
        logger.info(f"\n[STEP {step}]")

        screenshot_b64, img_resized, img_original = get_screenshot_b64(device)

        img_resized.save(os.path.join(data_dir, f"{step}.jpg"))

        try:
            hierarchy = device.dump_hierarchy()
            with open(os.path.join(data_dir, f"{step}.xml"), "w", encoding="utf-8") as f:
                f.write(hierarchy)
        except Exception as e:
            logger.warning(f"[WARN] hierarchy 保存失败: {e}")

        messages = build_decider_messages(task_desc, history, screenshot_b64)

        try:
            resp = call_decider_with_retry(messages)
        except Exception as e:
            logger.error(f"[ERROR] Decider 失败，终止任务: {e}")
            break

        reacts.append({
            "reasoning": resp.get("reasoning"),
            "function": {"name": resp["action"], "parameters": resp.get("parameters", {})},
            "action_index": step,
        })

        action = resp["action"]
        logger.info(f"[ACTION] {action} | {resp.get('parameters', {})} | screen={img_original.width}x{img_original.height}")

        if action == "done":
            status = resp.get("parameters", {}).get("status", "unknown")
            logger.info(f"[DONE] status={status}")
            actions.append({"type": "done", "status": status, "action_index": step})
            break

        try:
            execute_action(resp, device, img_original, actions, history, step)
        except Exception as e:
            logger.error(f"[ERROR] 执行动作失败: {e}")
            break

        time.sleep(1)

    now = datetime.now()
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    data = {
        "app_name": app_name,
        "task_description": task_desc,
        "execution_timestamp": {
            "date": now.strftime("%Y-%m-%d"),
            "weekday": weekdays[now.weekday()],
            "time": now.strftime("%H:%M:%S"),
        },
        "action_count": len(actions),
        "actions": actions,
    }
    with open(os.path.join(data_dir, "actions.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    with open(os.path.join(data_dir, "react.json"), "w", encoding="utf-8") as f:
        json.dump(reacts, f, ensure_ascii=False, indent=4)

    logger.info(f"[OK] 结果已保存至: {data_dir}")
    return data


# ============ 入口 ============

def main():
    parser = argparse.ArgumentParser(description="MobiAgent Mobile Standalone")
    parser.add_argument("--service_ip", default="127.0.0.1", help="模型服务 IP")
    parser.add_argument("--decider_port", type=int, default=7003, help="Decider 服务端口")
    parser.add_argument("--planner_port", type=int, default=7003, help="Planner 服务端口")
    parser.add_argument("--decider_model", default="", help="Decider 模型名称（留空自动获取）")
    parser.add_argument("--planner_model", default="", help="Planner 模型名称（留空自动获取）")
    parser.add_argument("--device_endpoint", default="127.0.0.1:5555", help="ADB 端点")
    parser.add_argument("--task", required=True, help="任务描述字符串或 task.json 路径")
    parser.add_argument("--data_dir", default="./data", help="结果保存目录")
    args = parser.parse_args()

    global decider_model, planner_model
    decider_model = args.decider_model
    planner_model = args.planner_model

    # 支持从 task.json 读取任务列表
    if os.path.isfile(args.task):
        with open(args.task, "r", encoding="utf-8") as f:
            raw = json.load(f)
        task_list = raw if isinstance(raw, list) else [str(raw)]
    else:
        task_list = [args.task]

    init_clients(args.service_ip, args.decider_port, args.planner_port)
    device = AndroidDevice(args.device_endpoint)

    for idx, task_item in enumerate(task_list, 1):
        if isinstance(task_item, dict):
            # 结构化格式已明确指定 app，跳过 planner
            app_name = task_item.get("app", "unknown")
            task_type = task_item.get("type", "default")
            package_name = AndroidDevice.APP_PACKAGES.get(app_name, "")
            for t_idx, task_desc in enumerate(task_item.get("tasks", []), 1):
                data_dir = os.path.join(args.data_dir, app_name, task_type, str(t_idx))
                logger.info(f"[TASK {t_idx}] {task_desc}")
                if package_name:
                    device.app_start(package_name)
                else:
                    try:
                        device.start_app(app_name)
                    except Exception as e:
                        logger.warning(f"[WARN] 启动应用失败: {e}")
                run_task(app_name, task_desc, device, data_dir)
                if package_name:
                    device.app_stop(package_name)
                time.sleep(APP_STOP_WAIT)
        else:
            # 自由文本任务：先用 Planner 分析选择 app 和优化任务描述
            task_desc = str(task_item)
            data_dir = os.path.join(args.data_dir, str(idx))
            app_name, package_name, final_task_desc = plan_task(task_desc)
            logger.info(f"[TASK {idx}] app={app_name} pkg={package_name}")
            if package_name:
                device.app_start(package_name)
            else:
                try:
                    device.start_app(app_name)
                except Exception as e:
                    logger.warning(f"[WARN] 启动应用失败: {e}")
            run_task(app_name, final_task_desc, device, data_dir)
            if package_name:
                device.app_stop(package_name)
            time.sleep(APP_STOP_WAIT)


if __name__ == "__main__":
    main()
