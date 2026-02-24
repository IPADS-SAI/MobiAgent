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

    def _walk(node: ET.Element) -> None:
        bounds = _parse_bounds(node.get("bounds", ""))
        clickable = _is_clickable(node.get("clickable"))

        if clickable and bounds:
            texts = _collect_texts(node)
            text_value = " ".join(texts) if texts else None
            nodes.append({"bbox": bounds, "text": text_value})

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
                text_value = " ".join(texts) if texts else None
                nodes.append({"bbox": bounds, "text": text_value})

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
            dedup[key] = {"bbox": list(key), "text": item.get("text")}
        else:
            if (not current.get("text")) and item.get("text"):
                current["text"] = item.get("text")
            elif current.get("text") and item.get("text"):
                incoming = str(item.get("text")).strip()
                if incoming and incoming not in str(current.get("text")):
                    merged = f"{current.get('text')} {incoming}".strip()
                    current["text"] = merged
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


def _load_json_from_text(raw_text: str) -> Optional[Dict[str, Any]]:
    if not raw_text:
        return None
    text = raw_text.strip()

    def _try_load(candidate: str) -> Optional[Dict[str, Any]]:
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
    return None


def _caption_with_vlm(client: OpenAI, model: str, image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    prompt = (
        "请用中文给出该UI元素的简短语义标签。"
        "只返回JSON格式：{\"label\": \"...\"}。"
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
        return "未识别"

    parsed = _load_json_from_text(content)
    if parsed and isinstance(parsed.get("label"), str):
        label = parsed["label"].strip()
        return label or "未识别"
    fallback = content.strip()
    return fallback or "未识别"


def _truncate_label(label: str, max_len: int = 24) -> str:
    label = " ".join(label.split())
    if len(label) <= max_len:
        return label
    return label[: max_len - 1] + "~"


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


def main() -> None:
    parser = argparse.ArgumentParser(description="UI semantic boxing with hierarchy + VLM")
    parser.add_argument("--device", choices=["Android", "Harmony"], default="Android")
    parser.add_argument("--adb_endpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--vlm_model", type=str, default=DEFAULT_VLM_MODEL)
    parser.add_argument("--base_url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--api_key", type=str, default=os.getenv("OPENROUTER_API_KEY", ""))
    parser.add_argument("--use_vlm", choices=["on", "off"], default="on")
    parser.add_argument("--max_vlm_calls", type=int, default=12)
    parser.add_argument("--max_items", type=int, default=20, help="UI objects limit")
    parser.add_argument("--min_area", type=int, default=16)
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
        filtered_nodes.append({"bbox": bbox, "text": item.get("text")})

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
        if not text:
            if not use_vlm:
                raise RuntimeError("Missing text requires VLM; set --use_vlm on")
            if vlm_calls >= effective_max_vlm_calls:
                logging.warning("VLM call budget reached, forcing extra call for missing text")
            x1, y1, x2, y2 = item["bbox"]
            crop = img.crop((x1, y1, x2, y2))
            text = _caption_with_vlm(client, args.vlm_model, crop)
            source = "vlm"
            vlm_calls += 1
        items.append({
            "id": idx,
            "bbox": item["bbox"],
            "text": text,
            "source": source,
        })

    annotated = _annotate_image(img.copy(), items)
    annotated_path = os.path.join(output_dir, "annotated.jpg")
    annotated.save(annotated_path)

    json_path = os.path.join(output_dir, "ui_semantics.json")
    payload = {
        "device": args.device,
        "screenshot_file": screenshot_path,
        "annotated_file": annotated_path,
        "hierarchy_file": hierarchy_path,
        "items": items,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logging.info("Saved annotated image: %s", annotated_path)
    logging.info("Saved semantic JSON: %s", json_path)


if __name__ == "__main__":
    main()
