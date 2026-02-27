import argparse
import base64
import io
import json
import logging
import os
import re
import time
import random
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

try:
    import uiautomator2 as u2
except Exception:
    u2 = None

try:
    from hmdriver2.driver import Driver
except Exception:
    Driver = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

DEFAULT_VLM_MODEL = "qwen/qwen3-vl-30a3"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
UI_KIND_CHOICES = {"按钮", "图标", "文字", "图片", "输入框", "容器"}
UI_KIND_TO_NOUN = {
    "按钮": "按钮",
    "图标": "图标",
    "文字": "文字",
    "图片": "图片",
    "输入框": "输入框",
    "容器": "区域",
}
UI_KIND_ALIASES = {
    "按钮": "按钮",
    "button": "按钮",
    "btn": "按钮",
    "图标": "图标",
    "icon": "图标",
    "文字": "文字",
    "文本": "文字",
    "text": "文字",
    "图片": "图片",
    "图像": "图片",
    "image": "图片",
    "photo": "图片",
    "输入框": "输入框",
    "输入": "输入框",
    "input": "输入框",
    "edittext": "输入框",
    "textbox": "输入框",
    "容器": "容器",
    "container": "容器",
    "区域": "容器",
    "panel": "容器",
}
TASK_TEMPLATES = [
    "请帮我点击屏幕上的{label}",
    "请点击屏幕上的{label}",
    "请帮我点一下屏幕上的{label}",
    "麻烦点击屏幕上的{label}",
    "请在屏幕上点击{label}",
]
REASONING_TEMPLATE = "当前界面中存在'{label}'这个{ui_kind}，直接点击该'{ui_kind}'对应的UI元素"
APP_NAME_TO_PACKAGE = {
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
    "拼多多": "com.xunmeng.pinduoduo.hos",
}


class AndroidDeviceAdapter:
    def __init__(self, adb_endpoint: Optional[str] = None) -> None:
        if u2 is None:
            raise RuntimeError("uiautomator2 is required for Android device access")
        self._device = u2.connect(adb_endpoint) if adb_endpoint else u2.connect()

    def screenshot(self, path: str) -> None:
        self._device.screenshot(path)

    def dump_hierarchy(self) -> str:
        return self._device.dump_hierarchy()


class HarmonyDeviceAdapter:
    def __init__(self) -> None:
        if Driver is None:
            raise RuntimeError("hmdriver2 is required for Harmony device access")
        self._device = Driver()

    def screenshot(self, path: str) -> None:
        self._device.screenshot(path)

    def dump_hierarchy(self) -> Any:
        return self._device.dump_hierarchy()


def _load_font(size: int = 28) -> ImageFont.FreeTypeFont:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    font_path = os.path.join(project_root, "msyh.ttf")
    try:
        return ImageFont.truetype(font_path, size)
    except Exception:
        return ImageFont.load_default()


def _parse_bounds(bounds_str: str) -> Optional[List[int]]:
    if not bounds_str:
        return None
    match = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", bounds_str)
    if not match:
        return None
    left, top, right, bottom = map(int, match.groups())
    return [left, top, right, bottom]


def _coerce_bounds(value: Any) -> Optional[List[int]]:
    if isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            return [int(v) for v in value]
        except Exception:
            return None
    if isinstance(value, dict):
        keys = {"left", "top", "right", "bottom"}
        if keys.issubset(value.keys()):
            try:
                return [int(value["left"]), int(value["top"]), int(value["right"]), int(value["bottom"])]
            except Exception:
                return None
        rect_keys = {"x", "y", "width", "height"}
        if rect_keys.issubset(value.keys()):
            try:
                left = int(value["x"])
                top = int(value["y"])
                right = left + int(value["width"])
                bottom = top + int(value["height"])
                return [left, top, right, bottom]
            except Exception:
                return None
    if isinstance(value, str):
        return _parse_bounds(value)
    return None


def _is_clickable(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        return "click" in normalized
    return False


def _extract_nodes_from_android_xml(hierarchy_xml: str) -> List[Dict[str, Any]]:
    import xml.etree.ElementTree as ET

    nodes: List[Dict[str, Any]] = []
    try:
        root = ET.fromstring(hierarchy_xml)
    except Exception:
        return nodes

    def _collect_texts(start_node: ET.Element) -> List[str]:
        texts: List[str] = []
        for n in start_node.iter():
            raw = n.get("text") or n.get("content-desc") or n.get("contentDescription")
            if raw:
                value = str(raw).strip()
                if value and value not in texts:
                    texts.append(value)
        return texts

    def _collect_descendant_stats(start_node: ET.Element) -> Dict[str, int]:
        image_like = 0
        text_like = 0
        for n in start_node.iter():
            cls = (n.get("class") or "").lower()
            if "imageview" in cls or "icon" in cls:
                image_like += 1
            if "textview" in cls or "edittext" in cls:
                text_like += 1
        return {
            "image_descendant_count": image_like,
            "text_descendant_count": text_like,
        }

    def _walk(node: ET.Element) -> None:
        bounds = _parse_bounds(node.get("bounds", ""))
        clickable = _is_clickable(node.get("clickable"))

        if clickable and bounds:
            texts = _collect_texts(node)
            stats = _collect_descendant_stats(node)
            text_value = " ".join(texts) if texts else None
            class_name = (node.get("class") or "").strip()
            resource_id = (node.get("resource-id") or "").strip()
            content_desc = (node.get("content-desc") or node.get("contentDescription") or "").strip()
            long_clickable = _is_clickable(node.get("long-clickable"))
            nodes.append(
                {
                    "bbox": bounds,
                    "text": text_value,
                    "meta": {
                        "class_name": class_name,
                        "resource_id": resource_id,
                        "content_desc": content_desc,
                        "clickable": clickable,
                        "long_clickable": long_clickable,
                        "text_count": len(texts),
                        "image_descendant_count": stats["image_descendant_count"],
                        "text_descendant_count": stats["text_descendant_count"],
                        "device_source": "android_xml",
                    },
                }
            )

        for child in list(node):
            _walk(child)

    _walk(root)
    return nodes


def _extract_nodes_from_harmony_json(hierarchy_obj: Any) -> List[Dict[str, Any]]:
    nodes: List[Dict[str, Any]] = []
    text_keys = {
        "text",
        "label",
        "name",
        "title",
        "contentDescription",
        "content-desc",
        "desc",
        "accessibilityLabel",
        "accessibilityText",
        "hint",
    }
    clickable_keys = {"clickable", "isClickable", "clickableState", "clickable_state"}

    def _get_clickable_flag(node: Dict[str, Any]) -> bool:
        for key in clickable_keys:
            if key in node:
                return _is_clickable(node.get(key))
        return False

    def _collect_texts(node: Any) -> List[str]:
        texts: List[str] = []

        def _gather(cur: Any) -> None:
            if isinstance(cur, dict):
                for key, val in cur.items():
                    if key in text_keys and isinstance(val, str):
                        value = val.strip()
                        if value and value not in texts:
                            texts.append(value)
                    _gather(val)
            elif isinstance(cur, list):
                for item in cur:
                    _gather(item)

        _gather(node)
        return texts

    def _collect_descendant_stats(node: Any) -> Dict[str, int]:
        image_like = 0
        text_like = 0

        def _gather(cur: Any) -> None:
            nonlocal image_like, text_like
            if isinstance(cur, dict):
                cls = str(cur.get("class") or cur.get("type") or cur.get("component") or "").lower()
                if any(token in cls for token in ["image", "icon"]):
                    image_like += 1
                if any(token in cls for token in ["text", "label", "input", "edit"]):
                    text_like += 1
                for val in cur.values():
                    _gather(val)
            elif isinstance(cur, list):
                for item in cur:
                    _gather(item)

        _gather(node)
        return {
            "image_descendant_count": image_like,
            "text_descendant_count": text_like,
        }

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            bounds = None
            if "bounds" in node:
                bounds = _coerce_bounds(node.get("bounds"))
            elif "rect" in node:
                bounds = _coerce_bounds(node.get("rect"))

            clickable = _get_clickable_flag(node)
            if clickable and bounds:
                texts = _collect_texts(node)
                stats = _collect_descendant_stats(node)
                text_value = " ".join(texts) if texts else None
                class_name = str(node.get("class") or node.get("type") or node.get("component") or "").strip()
                resource_id = str(node.get("resource-id") or node.get("resourceId") or node.get("id") or "").strip()
                content_desc = str(
                    node.get("content-desc")
                    or node.get("contentDescription")
                    or node.get("accessibilityLabel")
                    or node.get("accessibilityText")
                    or ""
                ).strip()
                nodes.append(
                    {
                        "bbox": bounds,
                        "text": text_value,
                        "meta": {
                            "class_name": class_name,
                            "resource_id": resource_id,
                            "content_desc": content_desc,
                            "clickable": clickable,
                            "long_clickable": _is_clickable(node.get("longClickable") or node.get("long-clickable")),
                            "text_count": len(texts),
                            "image_descendant_count": stats["image_descendant_count"],
                            "text_descendant_count": stats["text_descendant_count"],
                            "device_source": "harmony_json",
                        },
                    }
                )

            for val in node.values():
                _walk(val)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(hierarchy_obj)
    return nodes


def _dedupe_nodes(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    dedup: Dict[Tuple[int, int, int, int], Dict[str, Any]] = {}
    for item in nodes:
        bbox = item.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        key = tuple(int(v) for v in bbox)
        current = dedup.get(key)
        if not current:
            dedup[key] = {"bbox": list(key), "text": item.get("text"), "meta": item.get("meta") or {}}
        else:
            if (not current.get("text")) and item.get("text"):
                current["text"] = item.get("text")
            elif current.get("text") and item.get("text"):
                incoming = str(item.get("text")).strip()
                if incoming and incoming not in str(current.get("text")):
                    merged = f"{current.get('text')} {incoming}".strip()
                    current["text"] = merged
            current_meta = current.get("meta") or {}
            incoming_meta = item.get("meta") or {}
            for k, v in incoming_meta.items():
                if current_meta.get(k) in {None, ""} and v not in {None, ""}:
                    current_meta[k] = v
            current["meta"] = current_meta
    return list(dedup.values())


def _clip_bbox(bbox: List[int], img_w: int, img_h: int) -> Optional[List[int]]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w))
    y2 = max(0, min(y2, img_h))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _boxes_overlap(a: List[int], b: List[int]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)


def _select_non_overlapping(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def area(b: List[int]) -> int:
        return max(0, (b[2] - b[0]) * (b[3] - b[1]))

    sorted_nodes = sorted(nodes, key=lambda n: area(n["bbox"]))
    selected: List[Dict[str, Any]] = []
    for item in sorted_nodes:
        bbox = item["bbox"]
        if any(_boxes_overlap(bbox, s["bbox"]) for s in selected):
            continue
        selected.append(item)
    return selected


def _select_limited_items(nodes: List[Dict[str, Any]], max_items: int) -> List[Dict[str, Any]]:
    if max_items <= 0:
        return []
    if len(nodes) <= max_items:
        return nodes

    with_text = [item for item in nodes if item.get("text")]
    without_text = [item for item in nodes if not item.get("text")]

    if len(with_text) >= max_items:
        return random.sample(with_text, max_items)

    remaining = max_items - len(with_text)
    if remaining <= 0:
        return with_text
    if remaining >= len(without_text):
        return with_text + without_text
    return with_text + random.sample(without_text, remaining)


def _convert_bbox_to_relative(bbox: List[int], img_w: int, img_h: int) -> List[int]:
    if img_w <= 0 or img_h <= 0:
        return bbox
    x1, y1, x2, y2 = bbox
    return [
        int(round(x1 / img_w * 1000)),
        int(round(y1 / img_h * 1000)),
        int(round(x2 / img_w * 1000)),
        int(round(y2 / img_h * 1000)),
    ]


def _get_current_package(device, device_type: str, hierarchy_raw: Any) -> Optional[str]:
    if device_type == "Android":
        current = None
        try:
            if hasattr(device, "_device") and hasattr(device._device, "app_current"):
                current = device._device.app_current()
        except Exception:
            current = None
        if isinstance(current, dict):
            return current.get("package")
        return None

    if isinstance(hierarchy_raw, dict):
        def _find_bundle(node: Any) -> Optional[str]:
            if isinstance(node, dict):
                attrs = node.get("attributes") or {}
                if isinstance(attrs, dict) and attrs.get("bundleName"):
                    return attrs.get("bundleName")
                for val in node.values():
                    found = _find_bundle(val)
                    if found:
                        return found
            elif isinstance(node, list):
                for item in node:
                    found = _find_bundle(item)
                    if found:
                        return found
            return None
        return _find_bundle(hierarchy_raw)
    return None


def _annotate_single(image: Image.Image, item: Dict[str, Any]) -> Image.Image:
    return _annotate_image(image, [item])


def _load_json_from_text(raw_text: str) -> Optional[Any]:
    if not raw_text:
        return None
    text = raw_text.strip()

    def _try_load(candidate: str) -> Optional[Any]:
        try:
            return json.loads(candidate)
        except Exception:
            return None

    parsed = _try_load(text)
    if parsed is not None:
        return parsed

    for pattern in [r"```json\s*([\s\S]*?)\s*```", r"```\s*([\s\S]*?)\s*```"]:
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            parsed = _try_load(match.group(1).strip())
            if parsed is not None:
                return parsed

    start_idx = text.find("{")
    if start_idx != -1:
        brace_count = 0
        for i in range(start_idx, len(text)):
            if text[i] == "{":
                brace_count += 1
            elif text[i] == "}":
                brace_count -= 1
                if brace_count == 0:
                    return _try_load(text[start_idx : i + 1])

    array_start = text.find("[")
    if array_start != -1:
        bracket_count = 0
        for i in range(array_start, len(text)):
            if text[i] == "[":
                bracket_count += 1
            elif text[i] == "]":
                bracket_count -= 1
                if bracket_count == 0:
                    return _try_load(text[array_start : i + 1])
    return None


def _normalize_ui_kind(ui_kind: Optional[str]) -> str:
    if not ui_kind:
        return "图片"
    value = str(ui_kind).strip().lower()
    return UI_KIND_ALIASES.get(value, UI_KIND_ALIASES.get(str(ui_kind).strip(), "图片"))


def _infer_kind_from_fallback(label_text: str) -> str:
    text = (label_text or "").strip().lower()
    if not text:
        return "图片"
    if any(token in text for token in {"input", "edittext", "输入", "搜索"}):
        return "输入框"
    if "icon" in text or "图标" in text:
        return "图标"
    if "按钮" in text or "button" in text:
        return "按钮"
    if "容器" in text or "区域" in text or "container" in text:
        return "容器"
    if re.search(r"[\u4e00-\u9fff]", text):
        return "文字"
    return "图片"


def _caption_with_vlm(client: OpenAI, model: str, image: Image.Image) -> Tuple[str, str]:
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    prompt = (
        "请判断该UI元素属于以下哪一类：按钮、图标、文字、图片、输入框、容器，并给出中文简短语义标签。"
        "只返回JSON格式：{\"label\": \"...\", \"ui_kind\": \"按钮|图标|文字|图片|输入框|容器\"}。"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }
    ]

    content = None
    for attempt in range(2):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=128,
            timeout=45,
        )
        try:
            if response and response.choices:
                content = response.choices[0].message.content
        except Exception as e:
            red = "\033[91m"
            reset = "\033[0m"
            logging.warning(f"{red}Failed to read VLM response content: {e}{reset}")
            content = None

        if content:
            break
        if attempt == 0:
            time.sleep(0.3)

    if not content:
        red = "\033[91m"
        reset = "\033[0m"
        logging.warning(f"{red}Empty VLM response content, returning fallback label{reset}")
        return "未识别", "图片"

    parsed = _load_json_from_text(content)
    if parsed and isinstance(parsed.get("label"), str):
        label = parsed["label"].strip()
        ui_kind = _normalize_ui_kind(parsed.get("ui_kind"))
        return (label or "未识别", ui_kind)
    fallback = content.strip()
    label = fallback or "未识别"
    return label, _infer_kind_from_fallback(label)


def _normalize_confidence(value: Optional[str]) -> str:
    normalized = (value or "").strip().lower()
    if normalized in {"high", "高", "h"}:
        return "high"
    if normalized in {"medium", "mid", "中", "m"}:
        return "medium"
    return "low"


def _has_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def _classify_ui_kind_rule(item: Dict[str, Any], img_w: int, img_h: int) -> Tuple[str, str]:
    bbox = item["bbox"]
    text = (item.get("text") or "").strip()
    meta = item.get("meta") or {}
    class_name = str(meta.get("class_name") or "").lower()
    resource_id = str(meta.get("resource_id") or "").lower()
    content_desc = str(meta.get("content_desc") or "").strip()
    content_desc_l = content_desc.lower()
    text_count = int(meta.get("text_count") or 0)
    image_descendant_count = int(meta.get("image_descendant_count") or 0)
    text_descendant_count = int(meta.get("text_descendant_count") or 0)

    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    area = width * height
    screen_area = max(1, img_w * img_h)
    area_ratio = area / screen_area
    bottom_nav = y1 >= int(img_h * 0.85)
    has_text = bool(text)
    text_len = len(text.replace(" ", ""))
    has_icon_descendant = image_descendant_count > 0
    viewgroup_like = any(token in class_name for token in ["viewgroup", "framelayout", "linearlayout", "relativelayout"])

    if any(token in f"{class_name} {resource_id} {text.lower()} {content_desc_l}" for token in ["edittext", "input", "搜索", "输入"]):
        return "输入框", "high"

    if any(token in class_name for token in ["button", "compoundbutton", "materialbutton"]):
        return "按钮", "high"
    if any(token in resource_id for token in ["button", "_btn", "btn_", "submit", "confirm", "action"]):
        return "按钮", "high"

    if bottom_nav and content_desc and width <= int(img_w * 0.3):
        return "图标", "high"
    if bottom_nav and has_text and text_len <= 6 and width <= int(img_w * 0.3):
        return "图标", "high"
    if any(token in class_name for token in ["imageview", "icon"]) and bottom_nav:
        return "图标", "high"

    if has_text and text_len <= 10 and area_ratio <= 0.03:
        if any(token in text for token in ["完善资料", "立即", "速领", "签到", "登录", "注册", "确认", "保存"]):
            return "按钮", "medium"

    # 宫格入口常见形态：容器内同时有图标和短文本标签
    if has_icon_descendant and has_text and viewgroup_like:
        if width <= int(img_w * 0.32) and height <= int(img_h * 0.18):
            return "图标", "high"
        return "图标", "medium"

    # 大多数聚合容器的文本节点不止一个，不应误判成普通文字
    if viewgroup_like and text_descendant_count >= 2 and area_ratio >= 0.04:
        return "容器", "medium"
        if any(token in resource_id for token in ["edit", "action", "cta", "entry"]):
            return "按钮", "medium"

    if area_ratio >= 0.12:
        return "容器", "high"
    if area_ratio >= 0.06 and (text_count >= 2 or text_len > 14):
        return "容器", "medium"

    if any(token in class_name for token in ["imageview"]) and not has_text:
        return "图片", "medium"

    if has_text:
        return "文字", "medium"
    return "图片", "low"


def _dump_image_to_b64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def _review_kinds_with_vlm_page(
    client: OpenAI,
    model: str,
    annotated_image: Image.Image,
    items: List[Dict[str, Any]],
    max_retry: int,
    debug_out: Optional[Dict[str, Any]] = None,
) -> Dict[int, Dict[str, str]]:
    if not items:
        if debug_out is not None:
            debug_out.update({"status": "empty_items"})
        return {}

    item_lines = []
    for item in items:
        bbox = item["bbox"]
        item_lines.append(
            {
                "id": item["id"],
                "text": item.get("text") or "",
                "bbox": bbox,
                "ui_kind_rule": item.get("ui_kind_rule"),
                "kind_confidence_rule": item.get("kind_confidence_rule"),
            }
        )

    prompt = (
        "你将看到一张带红框和id标记的移动端截图。"
        "请根据每个id对应的UI元素，对其类型做复核。"
        "可选类型仅限：按钮、图标、文字、图片、输入框、容器。"
        "请严格返回JSON："
        "{\"items\":[{\"id\":1,\"ui_kind\":\"按钮|图标|文字|图片|输入框|容器\",\"confidence\":\"high|medium|low\",\"reason\":\"简短中文\"}]}"
        f"\n待复核元素列表：{json.dumps(item_lines, ensure_ascii=False)}"
    )

    if debug_out is not None:
        debug_out["request_items"] = item_lines
        debug_out["prompt"] = prompt

    b64 = _dump_image_to_b64(annotated_image)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }
    ]

    attempts = max(1, max_retry)
    for attempt in range(attempts):
        content = None
        parsed = None
        attempt_debug: Dict[str, Any] = {"attempt": attempt + 1}
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=512,
                timeout=60,
            )
            if response and response.choices:
                content = response.choices[0].message.content
            attempt_debug["raw_response"] = content
        except Exception as e:
            logging.warning("Kind review VLM call failed on attempt %s: %s", attempt + 1, e)
            content = None
            attempt_debug["error"] = str(e)

        parsed = _load_json_from_text(content or "")
        attempt_debug["parsed"] = parsed
        if debug_out is not None:
            debug_out.setdefault("attempts", []).append(attempt_debug)

        rows: Optional[List[Any]] = None
        if isinstance(parsed, dict):
            if isinstance(parsed.get("items"), list):
                rows = parsed.get("items")
            elif isinstance(parsed.get("data"), list):
                rows = parsed.get("data")
        elif isinstance(parsed, list):
            rows = parsed

        if not rows:
            if attempt < attempts - 1:
                time.sleep(0.4)
                continue
            if debug_out is not None:
                debug_out["status"] = "parse_failed"
            return {}

        reviewed: Dict[int, Dict[str, str]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            item_id = row.get("id")
            try:
                item_id = int(item_id)
            except Exception:
                continue
            reviewed[item_id] = {
                "ui_kind": _normalize_ui_kind(row.get("ui_kind")),
                "confidence": _normalize_confidence(row.get("confidence")),
                "reason": str(row.get("reason") or "").strip(),
            }
        if reviewed:
            if debug_out is not None:
                debug_out["status"] = "ok"
                debug_out["reviewed"] = reviewed
            return reviewed
    if debug_out is not None:
        debug_out["status"] = "no_reviewed_items"
    return {}


def _truncate_label(label: str, max_len: int = 24) -> str:
    label = " ".join(label.split())
    if len(label) <= max_len:
        return label
    return label[: max_len - 1] + "~"


def _normalize_label_for_prompt(label: str) -> str:
    text = " ".join((label or "").split()).strip(" ,，。")
    return text or "该元素"


def _build_target_phrase(label: str, ui_kind: str) -> str:
    safe_label = _normalize_label_for_prompt(label)
    noun = UI_KIND_TO_NOUN.get(ui_kind, "元素")
    return f"‘{safe_label}’{noun}"


def _annotate_image(image: Image.Image, items: List[Dict[str, Any]]) -> Image.Image:
    draw = ImageDraw.Draw(image)
    font = _load_font()
    for item in items:
        bbox = item["bbox"]
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        label = f"{item['id']}:{_truncate_label(item['text'])}"
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        tx = x1
        ty = max(0, y1 - text_h - 4)
        draw.rectangle([tx, ty, tx + text_w + 6, ty + text_h + 4], fill="red")
        draw.text((tx + 3, ty + 2), label, fill="white", font=font)
    return image


def _ensure_output_dir(output_dir: Optional[str]) -> str:
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = os.path.join(os.path.dirname(__file__), "ui-semantic-output", timestamp)
    os.makedirs(base, exist_ok=True)
    return base


def _dump_json_file(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="UI semantic boxing with hierarchy + VLM")
    parser.add_argument("--device", choices=["Android", "Harmony"], default="Android")
    parser.add_argument("--adb_endpoint", type=str, default=None)
    parser.add_argument("--app_name", type=str, required=True, help="App name for actions.json")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--vlm_model", type=str, default=DEFAULT_VLM_MODEL)
    parser.add_argument("--base_url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--api_key", type=str, default=os.getenv("OPENROUTER_API_KEY", ""))
    parser.add_argument("--use_vlm", choices=["on", "off"], default="on")
    parser.add_argument("--max_vlm_calls", type=int, default=12)
    parser.add_argument("--max_items", type=int, default=20, help="UI objects limit")
    parser.add_argument("--min_area", type=int, default=16)
    parser.add_argument("--enable_kind_vlm", choices=["on", "off"], default="on")
    parser.add_argument("--kind_vlm_mode", choices=["page_once"], default="page_once")
    parser.add_argument("--kind_vlm_max_retry", type=int, default=2)
    parser.add_argument("--task_desc_with_kind", choices=["on", "off"], default="on")
    args = parser.parse_args()

    output_dir = _ensure_output_dir(args.output_dir)
    logging.info("Output dir: %s", output_dir)

    if args.device == "Android":
        device = AndroidDeviceAdapter(args.adb_endpoint)
    else:
        device = HarmonyDeviceAdapter()

    screenshot_path = os.path.join(output_dir, "screenshot.jpg")
    hierarchy_path = os.path.join(output_dir, "hierarchy.xml" if args.device == "Android" else "hierarchy.json")

    device.screenshot(screenshot_path)
    hierarchy_raw = device.dump_hierarchy()

    current_package = _get_current_package(device, args.device, hierarchy_raw)
    expected_package = APP_NAME_TO_PACKAGE.get(args.app_name)
    if current_package and expected_package and current_package != expected_package:
        red = "\033[91m"
        reset = "\033[0m"
        logging.warning(
            f"{red}App mismatch: current package={current_package} expected={expected_package} ({args.app_name}){reset}"
        )

    if args.device == "Android":
        hierarchy_text = hierarchy_raw if isinstance(hierarchy_raw, str) else str(hierarchy_raw)
        with open(hierarchy_path, "w", encoding="utf-8") as f:
            f.write(hierarchy_text)
        nodes = _extract_nodes_from_android_xml(hierarchy_text)
    else:
        if isinstance(hierarchy_raw, str):
            try:
                hierarchy_obj = json.loads(hierarchy_raw)
            except Exception:
                hierarchy_obj = {}
        else:
            hierarchy_obj = hierarchy_raw
        with open(hierarchy_path, "w", encoding="utf-8") as f:
            json.dump(hierarchy_obj, f, ensure_ascii=False, indent=2)
        nodes = _extract_nodes_from_harmony_json(hierarchy_obj)

    if not nodes:
        logging.warning("No hierarchy nodes found")

    img = Image.open(screenshot_path).convert("RGB")
    img_w, img_h = img.size

    dedup_nodes = _dedupe_nodes(nodes)
    filtered_nodes: List[Dict[str, Any]] = []
    for item in dedup_nodes:
        bbox = _clip_bbox(item["bbox"], img_w, img_h)
        if not bbox:
            continue
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if area < args.min_area:
            continue
        filtered_nodes.append({"bbox": bbox, "text": item.get("text"), "meta": item.get("meta") or {}})

    selected_nodes = _select_non_overlapping(filtered_nodes)
    selected_nodes = _select_limited_items(selected_nodes, args.max_items)
    logging.info("Selected %s UI items", len(selected_nodes))

    use_vlm = args.use_vlm == "on"
    if use_vlm and OpenAI is None:
        raise RuntimeError("openai package is required for VLM labeling")

    if use_vlm and not args.api_key:
        raise RuntimeError("Missing API key for VLM labeling")

    client = OpenAI(api_key=args.api_key, base_url=args.base_url) if use_vlm else None

    items: List[Dict[str, Any]] = []
    vlm_calls = 0
    effective_max_vlm_calls = max(args.max_vlm_calls, args.max_items)
    for idx, item in enumerate(selected_nodes, 1):
        text = item.get("text")
        source = "hierarchy"
        rule_kind, rule_conf = _classify_ui_kind_rule(item, img_w, img_h)
        if not text:
            if not use_vlm:
                raise RuntimeError("Missing text requires VLM; set --use_vlm on")
            if vlm_calls >= effective_max_vlm_calls:
                logging.warning("VLM call budget reached, forcing extra call for missing text")
            x1, y1, x2, y2 = item["bbox"]
            crop = img.crop((x1, y1, x2, y2))
            text, _ = _caption_with_vlm(client, args.vlm_model, crop)
            source = "vlm"
            vlm_calls += 1
        items.append({
            "id": idx,
            "bbox": item["bbox"],
            "text": text,
            "source": source,
            "meta": item.get("meta") or {},
            "ui_kind_rule": rule_kind,
            "kind_confidence_rule": rule_conf,
            "ui_kind": rule_kind,
            "kind_source": "rule",
            "kind_confidence": rule_conf,
        })

    annotated = _annotate_image(img.copy(), items)
    annotated_path = os.path.join(output_dir, "annotated.jpg")
    annotated.save(annotated_path)

    kind_debug: Dict[str, Any] = {
        "enabled": args.enable_kind_vlm,
        "use_vlm": args.use_vlm,
        "kind_vlm_mode": args.kind_vlm_mode,
        "kind_vlm_max_retry": args.kind_vlm_max_retry,
        "model": args.vlm_model,
        "merge_decisions": [],
    }
    if args.enable_kind_vlm == "on" and use_vlm:
        reviewed = _review_kinds_with_vlm_page(
            client=client,
            model=args.vlm_model,
            annotated_image=annotated,
            items=items,
            max_retry=args.kind_vlm_max_retry,
            debug_out=kind_debug,
        )
        for item in items:
            verdict = reviewed.get(item["id"])
            if not verdict:
                kind_debug["merge_decisions"].append(
                    {
                        "id": item["id"],
                        "rule_kind": item.get("ui_kind_rule"),
                        "rule_confidence": item.get("kind_confidence_rule"),
                        "final_kind": item.get("ui_kind"),
                        "decision": "no_vlm_verdict_keep_rule",
                    }
                )
                continue
            vlm_kind = verdict.get("ui_kind", item["ui_kind_rule"])
            vlm_conf = _normalize_confidence(verdict.get("confidence"))
            item["kind_reason_vlm"] = verdict.get("reason", "")
            rule_kind = item.get("ui_kind_rule", "图片")
            rule_conf = item.get("kind_confidence_rule", "low")

            should_override = False
            if rule_conf in {"medium", "low"}:
                should_override = True
            elif rule_kind == "容器" and vlm_kind in {"按钮", "图标"} and vlm_conf == "high":
                should_override = True

            if should_override:
                item["ui_kind"] = vlm_kind
                item["kind_source"] = "rule+vlm"
                item["kind_confidence"] = vlm_conf
                kind_debug["merge_decisions"].append(
                    {
                        "id": item["id"],
                        "rule_kind": rule_kind,
                        "rule_confidence": rule_conf,
                        "vlm_kind": vlm_kind,
                        "vlm_confidence": vlm_conf,
                        "final_kind": item["ui_kind"],
                        "decision": "use_vlm",
                    }
                )
            else:
                if vlm_kind == rule_kind:
                    item["kind_source"] = "rule+vlm"
                item["ui_kind"] = rule_kind
                item["kind_confidence"] = rule_conf
                kind_debug["merge_decisions"].append(
                    {
                        "id": item["id"],
                        "rule_kind": rule_kind,
                        "rule_confidence": rule_conf,
                        "vlm_kind": vlm_kind,
                        "vlm_confidence": vlm_conf,
                        "final_kind": item["ui_kind"],
                        "decision": "keep_rule",
                    }
                )
    elif args.enable_kind_vlm == "on" and not use_vlm:
        logging.warning("enable_kind_vlm=on but use_vlm=off, skip page-level kind review")
        kind_debug["status"] = "skipped_use_vlm_off"
        for item in items:
            kind_debug["merge_decisions"].append(
                {
                    "id": item["id"],
                    "rule_kind": item.get("ui_kind_rule"),
                    "rule_confidence": item.get("kind_confidence_rule"),
                    "final_kind": item.get("ui_kind"),
                    "decision": "skip_vlm_keep_rule",
                }
            )
    else:
        kind_debug["status"] = "disabled"
        for item in items:
            kind_debug["merge_decisions"].append(
                {
                    "id": item["id"],
                    "rule_kind": item.get("ui_kind_rule"),
                    "rule_confidence": item.get("kind_confidence_rule"),
                    "final_kind": item.get("ui_kind"),
                    "decision": "disabled_keep_rule",
                }
            )

    _dump_json_file(os.path.join(output_dir, "ui_kind_vlm_review_debug.json"), kind_debug)

    json_path = os.path.join(output_dir, "ui_semantics.json")
    payload = {
        "device": args.device,
        "app_name": args.app_name,
        "screenshot_file": screenshot_path,
        "annotated_file": annotated_path,
        "hierarchy_file": hierarchy_path,
        "items": items,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    for item in items:
        label = item["text"]
        ui_kind = item.get("ui_kind", "文字")
        if ui_kind not in UI_KIND_CHOICES:
            ui_kind = "图片"
            item["ui_kind"] = ui_kind
        target_phrase = _build_target_phrase(label, ui_kind) if args.task_desc_with_kind == "on" else label
        task_text = random.choice(TASK_TEMPLATES).format(label=target_phrase)
        reasoning = REASONING_TEMPLATE.format(label=label, ui_kind=ui_kind)

        sub_dir = os.path.join(output_dir, str(item["id"]))
        os.makedirs(sub_dir, exist_ok=True)

        sub_screenshot_path = os.path.join(sub_dir, "1.jpg")
        img.save(sub_screenshot_path)

        sub_annotated_path = os.path.join(sub_dir, "1_annotated.jpg")
        single_annotated = _annotate_single(img.copy(), item)
        single_annotated.save(sub_annotated_path)

        hierarchy_out = os.path.join(sub_dir, "1.xml" if args.device == "Android" else "1.json")
        if args.device == "Android":
            with open(hierarchy_out, "w", encoding="utf-8") as f:
                f.write(hierarchy_text)
        else:
            with open(hierarchy_out, "w", encoding="utf-8") as f:
                json.dump(hierarchy_obj, f, ensure_ascii=False, indent=2)

        x1, y1, x2, y2 = item["bbox"]
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        action_record = {
            "type": "click",
            "position_x": center_x,
            "position_y": center_y,
            "bounds": [x1, y1, x2, y2],
            "action_index": 1,
        }

        actions_payload = {
            "app_name": args.app_name,
            "task_type": None,
            "old_task_description": task_text,
            "task_description": task_text,
            "action_count": 1,
            "actions": [action_record],
        }

        rel_bbox = _convert_bbox_to_relative([x1, y1, x2, y2], img_w, img_h)
        react_payload = [
            {
                "reasoning": reasoning,
                "function": {
                    "name": "click",
                    "parameters": {
                        "target_element": label,
                        "bbox": rel_bbox,
                    },
                },
                "action_index": 1,
            }
        ]

        with open(os.path.join(sub_dir, "actions.json"), "w", encoding="utf-8") as f:
            json.dump(actions_payload, f, ensure_ascii=False, indent=4)
        with open(os.path.join(sub_dir, "react.json"), "w", encoding="utf-8") as f:
            json.dump(react_payload, f, ensure_ascii=False, indent=4)

    logging.info("Saved annotated image: %s", annotated_path)
    logging.info("Saved semantic JSON: %s", json_path)


if __name__ == "__main__":
    main()
