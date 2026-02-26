import argparse
import difflib
import hashlib
import json
import logging
import os
import re
import time
import textwrap
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI
import cv2
from PIL import Image, ImageDraw, ImageFont
try:
    from utils.parse_xml import extract_all_bounds, parse_bounds
except Exception:
    extract_all_bounds = None
    parse_bounds = None

try:
    # 作为包运行
    from .mobiagent import (
        AndroidDevice,
        HarmonyDevice,
        build_decider_messages,
        call_model_with_validation_retry,
        compute_swipe_positions,
        convert_qwen3_coordinates_to_absolute,
        get_screenshot,
        robust_json_loads,
        validate_decider_response,
    )
except ImportError:
    # 直接运行脚本
    from mobiagent import (
        AndroidDevice,
        HarmonyDevice,
        build_decider_messages,
        call_model_with_validation_retry,
        compute_swipe_positions,
        convert_qwen3_coordinates_to_absolute,
        get_screenshot,
        robust_json_loads,
        validate_decider_response,
    )

try:
    from hmdriver2.proto import KeyCode
except Exception:
    KeyCode = None


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

API_TIMEOUT = 45
EXPLORER_MAX_TOKENS = 1024
MAX_RETRIES = 3
DEVICE_WAIT_TIME = 0.6
DECIDER_MODEL_PLACEHOLDER = ""


def _load_font(size: int = 36) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("msyh.ttf", size)
    except Exception:
        return ImageFont.load_default()


def annotate_action_visuals(
    action: str,
    action_record: Dict[str, Any],
    screenshot_file: str,
    data_dir: str,
    step_index: int,
) -> None:
    """为动作生成可视化标注图像。"""
    try:
        img = Image.open(screenshot_file)
    except Exception as e:
        logging.warning(f"Failed to open screenshot for annotation: {e}")
        return

    draw = ImageDraw.Draw(img)
    font = _load_font()
    text = f"{action.upper()} {action_record.get('source_task', '')}".strip()
    text = textwrap.fill(text, width=24)

    try:
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
    except Exception:
        text_width = 0

    text_x = max(0, (img.width - text_width) // 2)
    draw.text((text_x, 0), text, fill="red", font=font)

    highlighted_path = os.path.join(data_dir, f"{step_index}_highlighted.jpg")
    img.save(highlighted_path)

    if action in {"click", "click_input"}:
        bounds = action_record.get("bounds")
        if bounds and len(bounds) == 4:
            x1, y1, x2, y2 = bounds
            img_bounds = Image.open(highlighted_path)
            draw_bounds = ImageDraw.Draw(img_bounds)
            draw_bounds.rectangle([x1, y1, x2, y2], outline="red", width=5)
            bounds_path = os.path.join(data_dir, f"{step_index}_bounds.jpg")
            img_bounds.save(bounds_path)

            cv2image = cv2.imread(bounds_path)
            if cv2image is not None:
                x = action_record.get("position_x")
                y = action_record.get("position_y")
                if x is not None and y is not None:
                    cv2.circle(cv2image, (int(x), int(y)), 12, (0, 255, 0), -1)
                    click_point_path = os.path.join(data_dir, f"{step_index}_click_point.jpg")
                    cv2.imwrite(click_point_path, cv2image)

    if action == "swipe":
        sx = action_record.get("press_position_x")
        sy = action_record.get("press_position_y")
        ex = action_record.get("release_position_x")
        ey = action_record.get("release_position_y")
        if None not in (sx, sy, ex, ey):
            cv2image = cv2.imread(highlighted_path)
            if cv2image is not None:
                cv2.arrowedLine(
                    cv2image,
                    (int(sx), int(sy)),
                    (int(ex), int(ey)),
                    (255, 0, 0),
                    6,
                    tipLength=0.3,
                )
                swipe_path = os.path.join(data_dir, f"{step_index}_swipe.jpg")
                cv2.imwrite(swipe_path, cv2image)


def save_hierarchy(device, device_type: str, data_dir: str, step_index: int) -> None:
    """保存当前界面层级结构。"""
    try:
        hierarchy = device.dump_hierarchy()
    except Exception as e:
        logging.error(f"Dump hierarchy failed: {e}")
        hierarchy = "<hierarchy_dump_failed/>" if device_type == "Android" else {}

    if device_type == "Android":
        path = os.path.join(data_dir, f"{step_index}.xml")
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(hierarchy))
        return

    path = os.path.join(data_dir, f"{step_index}.json")
    try:
        obj = json.loads(hierarchy) if isinstance(hierarchy, str) else hierarchy
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    except Exception:
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(hierarchy))


def get_current_screenshot_path(device_type: str) -> str:
    name = "screenshot-Android.jpg" if device_type == "Android" else "screenshot-Harmony.jpg"
    return os.path.join(os.getcwd(), name)


def save_raw_screenshot(data_dir: str, step_index: int, device_type: str) -> str:
    src = get_current_screenshot_path(device_type)
    dst = os.path.join(data_dir, f"{step_index}.jpg")
    with open(src, "rb") as rf, open(dst, "wb") as wf:
        wf.write(rf.read())
    return dst


def persist_outputs(
    output_dir: str,
    app_name: str,
    actions: List[Dict[str, Any]],
    reacts: List[Dict[str, Any]],
    task_description: Optional[str] = None,
) -> None:
    step_words = {
        1: "第一步",
        2: "第二步",
        3: "第三步",
        4: "第四步",
        5: "第五步",
        6: "第六步",
        7: "第七步",
        8: "第八步",
        9: "第九步",
        10: "第十步",
    }

    step_tasks: List[str] = []
    normalized_actions: List[Dict[str, Any]] = []
    for idx, item in enumerate(actions, 1):
        normalized = dict(item)
        normalized["action_index"] = idx
        step_task = str(normalized.get("source_task", "")).strip()
        if step_task:
            step_tasks.append(step_task)
        normalized.pop("source_task", None)
        normalized_actions.append(normalized)

    normalized_reacts: List[Dict[str, Any]] = []
    for idx, item in enumerate(reacts, 1):
        normalized = dict(item)
        normalized["action_index"] = idx
        normalized.pop("source_task", None)
        normalized_reacts.append(normalized)

    if step_tasks:
        step_desc_parts = []
        for idx, step_task in enumerate(step_tasks, 1):
            step_label = step_words.get(idx, f"第{idx}步")
            step_desc_parts.append(f"{step_label}：{step_task}")
        computed_task_description = f"打开{app_name}，" + "，".join(step_desc_parts)
    else:
        computed_task_description = task_description or f"打开{app_name}"

    payload = {
        "app_name": app_name,
        "task_type": "auto_search",
        "old_task_description": None,
        "task_description": computed_task_description,
        "action_count": len(normalized_actions),
        "actions": normalized_actions,
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "actions.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=4)

    with open(os.path.join(output_dir, "react.json"), "w", encoding="utf-8") as f:
        json.dump(normalized_reacts, f, ensure_ascii=False, indent=4)


def persist_step_output(
    output_dir: str,
    app_name: str,
    action_record: Dict[str, Any],
    react_item: Dict[str, Any],
) -> None:
    persist_outputs(
        output_dir,
        app_name,
        [action_record],
        [react_item],
        task_description=action_record.get("source_task"),
    )


def copy_step_artifacts_to_path(steps_dir: str, path_dir: str, step_indices: List[int]) -> None:
    """将指定 step 目录中的截图/标注/xml 等文件复制到 path 目录，并在 path 内重编号。"""
    os.makedirs(path_dir, exist_ok=True)
    for new_idx, step_index in enumerate(step_indices, 1):
        step_dir = os.path.join(steps_dir, f"step_{step_index:04d}")
        if not os.path.isdir(step_dir):
            logging.warning(f"Step dir not found for path copy: {step_dir}")
            continue
        old_prefix = f"{step_index}"
        new_prefix = f"{new_idx}"
        for name in os.listdir(step_dir):
            src = os.path.join(step_dir, name)
            if not os.path.isfile(src):
                continue
            if name.startswith(old_prefix + ".") or name.startswith(old_prefix + "_"):
                renamed = new_prefix + name[len(old_prefix):]
            else:
                renamed = name
            dst = os.path.join(path_dir, renamed)
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                logging.warning(f"Failed to copy {src} -> {dst}: {e}")


def navigate_back(device, device_type: str) -> None:
    """执行返回上一层。"""
    try:
        if device_type == "Android":
            device.keyevent("back")
        else:
            if KeyCode is not None:
                device.keyevent(KeyCode.BACK)
            else:
                device.keyevent(2)
        time.sleep(DEVICE_WAIT_TIME)
    except Exception as e:
        logging.warning(f"Back navigation failed: {e}")


def _get_current_screen_size(device_type: str) -> Optional[tuple[int, int]]:
    try:
        path = get_current_screenshot_path(device_type)
        with Image.open(path) as img:
            return img.size
    except Exception as e:
        logging.warning(f"Failed to read current screenshot size: {e}")
        return None


def _reverse_direction(direction: str) -> str:
    mapping = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
    return mapping.get(direction.upper(), "DOWN")


def _convert_bbox_to_qwen3_relative(bbox: List[int], img_w: int, img_h: int) -> List[int]:
    if img_w <= 0 or img_h <= 0:
        return bbox
    x1, y1, x2, y2 = bbox
    return [
        int(round(x1 / img_w * 1000)),
        int(round(y1 / img_h * 1000)),
        int(round(x2 / img_w * 1000)),
        int(round(y2 / img_h * 1000)),
    ]


def _get_app_package_name(device, app_name: str) -> Optional[str]:
    if not app_name:
        return None
    mapping = getattr(device, "app_package_names", None)
    if isinstance(mapping, dict):
        return mapping.get(app_name)
    return None


def _extract_foreground_from_hierarchy(hierarchy_text: str) -> tuple[str, str]:
    if not hierarchy_text:
        return "", ""

    try:
        obj = json.loads(hierarchy_text) if isinstance(hierarchy_text, str) else hierarchy_text
    except Exception:
        return "", ""

    package_keys = {"bundleName", "bundle_name", "packageName", "package", "appPackage"}
    ability_keys = {"abilityName", "ability_name", "uiAbilityName", "uiAbility", "pageName", "page_name"}

    def _walk(node: Any) -> tuple[str, str]:
        if isinstance(node, dict):
            package = ""
            ability = ""
            for key, val in node.items():
                if key in package_keys and isinstance(val, str) and val.strip() and not package:
                    package = val.strip()
                if key in ability_keys and isinstance(val, str) and val.strip() and not ability:
                    ability = val.strip()
            attrs = node.get("attributes")
            if isinstance(attrs, dict):
                for key, val in attrs.items():
                    if key in package_keys and isinstance(val, str) and val.strip() and not package:
                        package = val.strip()
                    if key in ability_keys and isinstance(val, str) and val.strip() and not ability:
                        ability = val.strip()
            if package and ability:
                return package, ability
            for val in node.values():
                p, a = _walk(val)
                if p and not package:
                    package = p
                if a and not ability:
                    ability = a
                if package and ability:
                    return package, ability
            return package, ability
        if isinstance(node, list):
            package = ""
            ability = ""
            for item in node:
                p, a = _walk(item)
                if p and not package:
                    package = p
                if a and not ability:
                    ability = a
                if package and ability:
                    return package, ability
            return package, ability
        return "", ""

    return _walk(obj)


def _get_foreground_app_state(device, device_type: str, hierarchy_text: str = "") -> Dict[str, str]:
    state = {"package": "", "ability": "", "source": "unknown"}
    driver = getattr(device, "d", None)

    if device_type == "Android" and driver is not None:
        for api_name in ("app_current", "current_app"):
            api = getattr(driver, api_name, None)
            if callable(api):
                try:
                    raw = api()
                    if isinstance(raw, dict):
                        pkg = str(raw.get("package") or raw.get("appPackage") or raw.get("pkg") or "").strip()
                        act = str(raw.get("activity") or raw.get("appActivity") or raw.get("act") or "").strip()
                        if pkg:
                            state.update({"package": pkg, "ability": act, "source": f"android:{api_name}"})
                            return state
                    elif isinstance(raw, str) and raw.strip():
                        state.update({"package": raw.strip(), "source": f"android:{api_name}"})
                        return state
                except Exception:
                    continue

    if device_type == "Harmony" and driver is not None:
        shell = getattr(driver, "shell", None)
        if callable(shell):
            for cmd in ("aa dump --mission-list", "aa dump -l", "aa dump --stack"):
                try:
                    out = shell(cmd)
                    text = str(out)
                    pkg_match = re.search(r"(?:bundleName|bundle_name|packageName)\s*[:=]\s*([\w\.]+)", text)
                    ability_match = re.search(r"(?:abilityName|uiAbilityName|ability|uiAbility)\s*[:=]\s*([\w\.$]+)", text)
                    if pkg_match:
                        state["package"] = pkg_match.group(1)
                    if ability_match:
                        state["ability"] = ability_match.group(1)
                    if state["package"]:
                        state["source"] = f"harmony:shell:{cmd}"
                        return state
                except Exception:
                    continue

    pkg, ability = _extract_foreground_from_hierarchy(hierarchy_text)
    if pkg or ability:
        state.update({"package": pkg, "ability": ability, "source": "hierarchy"})
    return state


def _is_app_in_foreground(device, device_type: str, app_name: Optional[str], hierarchy_text: str) -> bool:
    if not app_name:
        return True
    package_name = _get_app_package_name(device, app_name)
    if not package_name:
        return True

    fg_state = _get_foreground_app_state(device, device_type, hierarchy_text)
    fg_package = fg_state.get("package", "")
    if fg_package:
        in_foreground = package_name == fg_package or package_name in fg_package or fg_package in package_name
        if not in_foreground:
            logging.warning(
                "\033[91mForeground app mismatch: expect=%s actual=%s ability=%s source=%s\033[0m",
                package_name,
                fg_package,
                fg_state.get("ability", ""),
                fg_state.get("source", "unknown"),
            )
        return in_foreground

    if hierarchy_text:
        return package_name in hierarchy_text
    return True


def replay_action_record(device, action_record: Dict[str, Any]) -> bool:
    action_type = str(action_record.get("type", "")).lower()
    try:
        if action_type == "click":
            x = action_record.get("position_x")
            y = action_record.get("position_y")
            if x is None or y is None:
                return False
            device.click(int(x), int(y))
            return True
        if action_type == "click_input":
            x = action_record.get("position_x")
            y = action_record.get("position_y")
            text = action_record.get("text", "")
            if x is None or y is None:
                return False
            device.click(int(x), int(y))
            if text:
                device.input(str(text))
            return True
        if action_type == "input":
            text = action_record.get("text", "")
            if text:
                device.input(str(text))
            return True
        if action_type == "swipe":
            sx = action_record.get("press_position_x")
            sy = action_record.get("press_position_y")
            ex = action_record.get("release_position_x")
            ey = action_record.get("release_position_y")
            if None in (sx, sy, ex, ey):
                return False
            device.swipe_with_coords(int(sx), int(sy), int(ex), int(ey))
            return True
        if action_type == "wait":
            time.sleep(1.0)
            return True
    except Exception as e:
        logging.warning(f"Replay action failed (type={action_type}): {e}")
        return False
    return False


def perform_backtrack_action(device, device_type: str, action_record: Optional[Dict[str, Any]]) -> None:
    if not action_record:
        navigate_back(device, device_type)
        return

    action_type = str(action_record.get("type", "")).lower()
    if action_type == "click_input":
        navigate_back(device, device_type)
        navigate_back(device, device_type)
        return

    if action_type == "swipe":
        direction = str(action_record.get("direction", "")).upper()
        reverse_direction = _reverse_direction(direction)
        size = _get_current_screen_size(device_type)
        if size:
            img_w, img_h = size
            sx, sy, ex, ey = compute_swipe_positions(reverse_direction, img_w, img_h)
            try:
                device.swipe_with_coords(sx, sy, ex, ey)
                time.sleep(DEVICE_WAIT_TIME)
                return
            except Exception as e:
                logging.warning(f"Reverse swipe backtrack failed: {e}")

    navigate_back(device, device_type)


def _normalize_hierarchy_text(text: str) -> str:
    text = re.sub(r"\d+", "#", text)
    return "".join(text.split())


def _hierarchy_fingerprint(hierarchy_text: str) -> str:
    if not hierarchy_text:
        return ""
    normalized = _normalize_hierarchy_text(hierarchy_text)
    if not normalized:
        return ""
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def _task_mentions_swipe(task_text: str) -> bool:
    if not task_text:
        return False
    tokens = ["滑动", "上滑", "下滑", "左滑", "右滑", "上划", "下划", "左划", "右划"]
    return any(t in task_text for t in tokens)


def _extract_click_target_text(task_text: str) -> Optional[str]:
    if not task_text or "点击" not in task_text:
        return None
    match = re.search(
        r"点击.*?[\"\u201c\u201d\u300c\u300d]([^\"\u201c\u201d\u300c\u300d]+)[\"\u201c\u201d\u300c\u300d]",
        task_text,
    )
    if match:
        return match.group(1).strip()
    return None


def _bbox_iou(a: List[int], b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    a_area = max(0, (ax2 - ax1)) * max(0, (ay2 - ay1))
    b_area = max(0, (bx2 - bx1)) * max(0, (by2 - by1))
    if a_area + b_area - inter_area == 0:
        return 0.0
    return inter_area / (a_area + b_area - inter_area)


def _extract_bounds_from_json(obj: Any) -> List[List[int]]:
    bounds_list: List[List[int]] = []

    def _coerce_bounds(value: Any) -> Optional[List[int]]:
        if isinstance(value, (list, tuple)) and len(value) == 4:
            try:
                return [int(v) for v in value]
            except Exception:
                return None
        if isinstance(value, str) and parse_bounds:
            return parse_bounds(value)
        return None

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, val in node.items():
                if key == "bounds":
                    bounds = _coerce_bounds(val)
                    if bounds:
                        bounds_list.append(bounds)
                else:
                    _walk(val)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(obj)
    return bounds_list


def _extract_bounds_from_hierarchy_text(hierarchy_text: str) -> List[List[int]]:
    if not hierarchy_text:
        return []
    if hierarchy_text.lstrip().startswith("<"):
        if extract_all_bounds:
            return extract_all_bounds(hierarchy_text, need_clickable=True)
        return []

    try:
        obj = json.loads(hierarchy_text)
    except Exception:
        return []
    return _extract_bounds_from_json(obj)


def _extract_text_bounds_from_hierarchy_text(hierarchy_text: str, target_text: str) -> List[List[int]]:
    if not hierarchy_text or not target_text:
        return []
    target = target_text.strip()
    if not target:
        return []

    if hierarchy_text.lstrip().startswith("<"):
        try:
            import xml.etree.ElementTree as ET

            root = ET.fromstring(hierarchy_text)
        except Exception:
            return []

        bounds_list: List[List[int]] = []
        for node in root.iter():
            node_text = node.get("text") or ""
            node_desc = node.get("content-desc") or node.get("contentDescription") or ""
            if target in node_text or target in node_desc:
                bounds_str = node.get("bounds")
                bounds = parse_bounds(bounds_str) if parse_bounds else None
                if bounds:
                    bounds_list.append(bounds)
        return bounds_list

    try:
        obj = json.loads(hierarchy_text)
    except Exception:
        return []

    bounds_list: List[List[int]] = []
    text_keys = {"text", "label", "name", "title", "contentDescription", "content-desc", "desc"}

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
        if isinstance(value, str) and parse_bounds:
            return parse_bounds(value)
        return None

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            matched = False
            for key, val in node.items():
                if key in text_keys and isinstance(val, str) and target in val:
                    matched = True
                    break
            if matched:
                bounds = _coerce_bounds(node.get("bounds"))
                if bounds:
                    bounds_list.append(bounds)
            for val in node.values():
                _walk(val)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(obj)
    return bounds_list


def refine_bbox_with_hierarchy(
    hierarchy_text: str,
    bbox: List[int],
    img_w: int,
    img_h: int,
) -> List[int]:
    bounds_list = _extract_bounds_from_hierarchy_text(hierarchy_text)
    green = "\033[92m"
    red = "\033[91m"
    reset = "\033[0m"
    if not bounds_list:
        logging.info(f"{red}BBox not refined (no hierarchy bounds found).{reset}")
        return bbox

    best_bbox = bbox
    best_iou = 0.0
    bx1, by1, bx2, by2 = bbox
    bcx = (bx1 + bx2) / 2
    bcy = (by1 + by2) / 2
    b_area = max(1, (bx2 - bx1) * (by2 - by1))
    max_center_dist = (img_w**2 + img_h**2) ** 0.5 * 0.08

    for cand in bounds_list:
        iou = _bbox_iou(bbox, cand)
        if iou > best_iou:
            best_iou = iou
            best_bbox = cand

    if best_iou >= 0.3:
        logging.info(
            f"{green}BBox refined by IoU (iou={best_iou:.3f}) from {bbox} to {best_bbox}.{reset}"
        )
        return best_bbox

    cx1, cy1, cx2, cy2 = best_bbox
    ccx = (cx1 + cx2) / 2
    ccy = (cy1 + cy2) / 2
    center_dist = ((ccx - bcx) ** 2 + (ccy - bcy) ** 2) ** 0.5
    c_area = max(1, (cx2 - cx1) * (cy2 - cy1))
    area_ratio = c_area / b_area if b_area else 1.0
    if center_dist <= max_center_dist and 0.5 <= area_ratio <= 2.0:
        logging.info(
            f"{green}BBox refined by center/area from {bbox} to {best_bbox}.{reset}"
        )
        return best_bbox

    logging.info(f"{red}BBox not refined (no close match).{reset}")
    return bbox


def format_action_history(action_history: List[Dict[str, Any]], max_items: int = 20) -> str:
    if not action_history:
        return "无"

    recent = action_history[-max_items:]
    lines = []
    for i, item in enumerate(recent, 1):
        task = str(item.get("source_task", "")).strip()
        action_type = str(item.get("type", "")).strip()
        detail_parts = []
        if action_type in {"input", "click_input"} and item.get("text"):
            detail_parts.append(f"text={item.get('text')}")
        if action_type == "swipe" and item.get("direction"):
            detail_parts.append(f"direction={item.get('direction')}")
        detail = f" ({', '.join(detail_parts)})" if detail_parts else ""
        lines.append(f"{i}. task={task or '-'} | action={action_type or '-'}{detail}")
    return "\n".join(lines)


def build_explorer_prompt(
    depth: int,
    breadth: int,
    hierarchy_text: str,
    action_history: List[Dict[str, Any]],
) -> str:
    """构建通用大模型的候选动作生成提示词。"""
    history_text = format_action_history(action_history)
    return f"""
你是移动端GUI探索助手。请结合截图、层级信息以及已发生的交互动作序列，输出当前界面“最有可能被用户下一步操作”的前{breadth}个单步任务，优先选择左侧、顶部或者底部的导航栏中的元素，并尽可能保证前后动作的连贯性。

要求：
1) 只输出 JSON，不要输出任何额外文本。
2) 输出字段必须是：
{{
  "candidates": [
    {{
      "rank": 1,
      "single_step_task": "一句话单步任务，例如：点击“搜索框”并输入“咖啡"",
      "reason": "为什么这个动作高概率"
    }}
  ]
}}
3) candidates 数量 <= {breadth}，按概率从高到低排序。
4) single_step_task 必须可执行、原子化（单步），避免多步串联。
5) 如果是点击输入框的动作，务必跟上合理的当前界面下的输入文本，例如：点击“搜索框”并输入“咖啡”。
6) 若界面左侧、底部或者顶部侧边栏存在导航列表项（例如设置列表，菜单列表等），请你优先输出对各个列表项的点击操作，如果当前界面显示的列表项不全，可以输出滑动操作以查看更多列表项。
7) 候选动作需要与最近的交互动作序列保持连贯，避免与已发生动作明显冲突；必要时可继续完成上一动作的后续步骤。
8) 如果界面的右下角有“我的”、“个人中心”之类的入口图标，并且该图标没有被选中（图标是实心或者如表下面有下滑杠，表示选中），建议优先输出点击该入口的动作。
9) 如果界面上有返回按钮，请不要点击返回按钮了，除非当前界面没有其他明显的可交互元素了。
10) 最近已经交互过的动作，请不要重复执行了，除非当前界面没有其他明显的可交互元素了。

当前探索深度: {depth}
最近交互动作序列(按时间顺序，最多展示20条):
{history_text}
""".strip()

# 当前UI层级(可能截断):
# {hierarchy_text[:12000]}


def call_explorer_model(
    explorer_client: OpenAI,
    explorer_model: str,
    screenshot_b64: str,
    hierarchy_text: str,
    depth: int,
    breadth: int,
    action_history: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """调用远端通用大模型，返回候选单步任务列表。"""
    prompt = build_explorer_prompt(depth, breadth, hierarchy_text, action_history)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{screenshot_b64}"},
                },
            ],
        }
    ]

    last_err: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            response = explorer_client.chat.completions.create(
                model=explorer_model,
                messages=messages,
                timeout=API_TIMEOUT,
                max_tokens=EXPLORER_MAX_TOKENS,
                temperature=0.4 + attempt * 0.2,
            )
            content = response.choices[0].message.content
            parsed = robust_json_loads(content)
            candidates = parsed.get("candidates", [])
            if not isinstance(candidates, list):
                raise ValueError("`candidates` must be a list")

            normalized = []
            for i, c in enumerate(candidates[:breadth], 1):
                task = str(c.get("single_step_task", "")).strip()
                if not task:
                    continue
                normalized.append(
                    {
                        "rank": c.get("rank", i),
                        "single_step_task": task,
                        "reason": str(c.get("reason", "")).strip(),
                    }
                )

            if not normalized:
                raise ValueError("No valid candidates returned")
            return normalized
        except Exception as e:
            last_err = e
            logging.warning(f"Explorer model parse/call failed (attempt={attempt + 1}): {e}")
            time.sleep(1.2)

    raise RuntimeError(f"Explorer model failed after retries: {last_err}")


def execute_decider_one_step(
    decider_client: OpenAI,
    decider_model: str,
    device,
    device_type: str,
    app_name: str,
    step_task: str,
    use_qwen3: bool,
    allow_hierarchy_text_decider: bool,
    output_dir: str,
    step_index: int,
) -> Dict[str, Any]:
    """使用 e2e decider 执行一个单步任务并记录数据。"""
    pre_action_hierarchy_text = get_hierarchy_text(device)
    target_text = _extract_click_target_text(step_task)
    bounds_from_text = (
        _extract_text_bounds_from_hierarchy_text(pre_action_hierarchy_text, target_text)
        if target_text
        else []
    )

    decider_resp: Optional[Dict[str, Any]] = None
    if allow_hierarchy_text_decider and target_text and bounds_from_text:
        green = "\033[92m"
        reset = "\033[0m"
        logging.info(
            f"{green}Using hierarchy_text_UI bbox for click target: {target_text}{reset}"
        )
        best_bbox = min(
            bounds_from_text,
            key=lambda b: max(1, (b[2] - b[0]) * (b[3] - b[1])),
        )
        if use_qwen3:
            size = _get_current_screen_size(device_type)
            if size:
                img_w, img_h = size
                best_bbox = _convert_bbox_to_qwen3_relative(best_bbox, img_w, img_h)
        decider_resp = {
            "action": "click",
            "parameters": {"bbox": best_bbox},
            "reasoning": f"观察到屏幕上存在{target_text}文字，直接点击{target_text}文本按钮",
        }
    else:
        red = "\033[91m"
        reset = "\033[0m"
        logging.info(f"{red}Using model output bbox (decider).{reset}")
        screenshot_b64 = get_screenshot(device, device_type)
        messages = build_decider_messages(f"当前处在{app_name}，请帮我{step_task}", [], screenshot_b64, True)

        def _validator(resp: Dict[str, Any]) -> None:
            validate_decider_response(resp, use_e2e=True)

        for attempt in range(3):
            decider_resp = call_model_with_validation_retry(
                decider_client,
                decider_model,
                messages,
                validator_func=_validator,
                max_retries=5,
                max_tokens=256,
                context="Decider",
            )
            action = decider_resp.get("action")
            if _task_mentions_swipe(step_task) and action == "click":
                if attempt < 2:
                    logging.warning(
                        "Decider action mismatch (attempt=%s): task expects swipe but got click. Retrying.",
                        attempt + 1,
                    )
                    time.sleep(0.6)
                    continue
                color = "\033[91m"
                reset = "\033[0m"
                logging.error(
                    f"{color}Decider action mismatch after retries: task expects swipe but got click. "
                    f"Skipping task: {step_task}{reset}"
                )
                raise RuntimeError("Decider action mismatch: swipe task returned click")
            break

    if decider_resp is None:
        raise RuntimeError("Decider response is empty")

    # 保存截图与层级
    screenshot_file = save_raw_screenshot(output_dir, step_index, device_type)
    save_hierarchy(device, device_type, output_dir, step_index)

    react_item = {
        "action_index": step_index,
        "source_task": step_task,
        "reasoning": decider_resp.get("reasoning", ""),
        "function": {
            "name": decider_resp.get("action", ""),
            "parameters": decider_resp.get("parameters", {}),
        },
    }

    action = decider_resp["action"]
    color = "\033[96m"
    reset = "\033[0m"
    logging.info(
        f"{color}Decider task: {step_task} -> action: {action}{reset}"
    )
    params = decider_resp.get("parameters", {})

    # 获取当前图像大小用于坐标换算
    from PIL import Image

    img = Image.open(screenshot_file)
    img_w, img_h = img.size

    action_record: Dict[str, Any] = {
        "action_index": step_index,
        "source_task": step_task,
        "type": action,
    }

    if action == "click":
        bbox = params.get("bbox")
        if use_qwen3:
            bbox = convert_qwen3_coordinates_to_absolute(bbox, img_w, img_h, is_bbox=True)
        if bbox:
            bbox = refine_bbox_with_hierarchy(pre_action_hierarchy_text, bbox, img_w, img_h)
        x1, y1, x2, y2 = bbox
        x, y = (x1 + x2) // 2, (y1 + y2) // 2
        device.click(x, y)
        action_record.update(
            {
                "position_x": x,
                "position_y": y,
                "bounds": [x1, y1, x2, y2],
            }
        )

    elif action == "click_input":
        bbox = params.get("bbox")
        text = params.get("text", "")
        if use_qwen3:
            bbox = convert_qwen3_coordinates_to_absolute(bbox, img_w, img_h, is_bbox=True)
        if bbox:
            bbox = refine_bbox_with_hierarchy(pre_action_hierarchy_text, bbox, img_w, img_h)
        x1, y1, x2, y2 = bbox
        x, y = (x1 + x2) // 2, (y1 + y2) // 2
        device.click(x, y)
        device.input(text)
        action_record.update(
            {
                "position_x": x,
                "position_y": y,
                "bounds": [x1, y1, x2, y2],
                "text": text,
            }
        )

    elif action == "input":
        text = params.get("text", "")
        device.input(text)
        action_record["text"] = text

    elif action == "swipe":
        direction = str(params.get("direction", "UP")).upper()
        start_coords = params.get("start_coords")
        end_coords = params.get("end_coords")
        if start_coords and end_coords:
            if use_qwen3:
                start_coords = convert_qwen3_coordinates_to_absolute(start_coords, img_w, img_h, is_bbox=False)
                end_coords = convert_qwen3_coordinates_to_absolute(end_coords, img_w, img_h, is_bbox=False)
            sx, sy = start_coords
            ex, ey = end_coords
        else:
            sx, sy, ex, ey = compute_swipe_positions(direction, img_w, img_h)

        device.swipe_with_coords(sx, sy, ex, ey)
        action_record.update(
            {
                "direction": direction.lower(),
                "press_position_x": sx,
                "press_position_y": sy,
                "release_position_x": ex,
                "release_position_y": ey,
            }
        )

    elif action == "wait":
        time.sleep(1.0)

    elif action == "done":
        action_record["status"] = params.get("status", "success")

    else:
        raise ValueError(f"Unsupported action from decider: {action}")

    try:
        annotate_action_visuals(action, action_record, screenshot_file, output_dir, step_index)
    except Exception as e:
        logging.warning(f"Failed to annotate action visuals: {e}")

    post_hierarchy_text = get_hierarchy_text(device)

    return {
        "decider_response": decider_resp,
        "action_record": action_record,
        "react_item": react_item,
        "screenshot_file": screenshot_file,
        "post_hierarchy_text": post_hierarchy_text,
    }


def get_hierarchy_text(device) -> str:
    try:
        h = device.dump_hierarchy()
        if isinstance(h, str):
            return h
        return json.dumps(h, ensure_ascii=False)
    except Exception as e:
        logging.warning(f"Failed to dump hierarchy for explorer: {e}")
        return ""


def explore_dfs(
    *,
    app_name: str,
    depth_limit: int,
    breadth: int,
    current_depth: int,
    decider_client: OpenAI,
    decider_model: str,
    explorer_client: OpenAI,
    explorer_model: str,
    device,
    device_type: str,
    use_qwen3: bool,
    allow_hierarchy_text_decider: bool,
    data_dir: str,
    actions: List[Dict[str, Any]],
    reacts: List[Dict[str, Any]],
    step_counter: List[int],
    path_counter: List[int],
    steps_dir: str,
    paths_dir: str,
    path_actions: Optional[List[Dict[str, Any]]] = None,
    path_reacts: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """DFS探索：每层挑选H个候选，逐个执行并回溯。"""
    if current_depth >= depth_limit:
        return

    screenshot_b64 = get_screenshot(device, device_type)
    hierarchy_text = get_hierarchy_text(device)
    action_history = list(actions)
    candidates = call_explorer_model(
        explorer_client,
        explorer_model,
        screenshot_b64,
        hierarchy_text,
        current_depth,
        breadth,
        action_history,
    )
    base_hierarchy_text = hierarchy_text

    logging.info(f"Depth={current_depth}, got {len(candidates)} candidates")
    for cand in candidates:
        color = "\033[96m"
        reset = "\033[0m"
        logging.info(
            f"{color}[Depth {current_depth}] candidate rank={cand.get('rank')} "
            f"task={cand.get('single_step_task')} reason={cand.get('reason')}{reset}"
        )

    cand_idx = 0
    while cand_idx < len(candidates):
        if cand_idx > 0:
            current_hierarchy_text = get_hierarchy_text(device)
            similarity = difflib.SequenceMatcher(
                None,
                _normalize_hierarchy_text(base_hierarchy_text),
                _normalize_hierarchy_text(current_hierarchy_text),
            ).ratio()
            if similarity < 0.9:
                remaining = max(breadth - cand_idx, 0)
                if remaining == 0:
                    break
                color = "\033[93m"
                reset = "\033[0m"
                logging.info(
                    f"{color}Page changed (similarity={similarity:.3f} < 0.900). "
                    f"Regenerating {remaining} candidates.{reset}"
                )
                screenshot_b64 = get_screenshot(device, device_type)
                base_hierarchy_text = current_hierarchy_text
                candidates = candidates[:cand_idx] + call_explorer_model(
                    explorer_client,
                    explorer_model,
                    screenshot_b64,
                    base_hierarchy_text,
                    current_depth,
                    remaining,
                    list(actions),
                )

        cand = candidates[cand_idx]
        task = cand["single_step_task"]
        step_counter[0] += 1
        step_idx = step_counter[0]
        step_output_dir = os.path.join(steps_dir, f"step_{step_idx:04d}")
        os.makedirs(step_output_dir, exist_ok=True)

        current_path_actions = list(path_actions) if path_actions else []
        current_path_reacts = list(path_reacts) if path_reacts else []

        cyan = "\033[96m"
        reset = "\033[0m"
        logging.info(
            f"{cyan}[Depth {current_depth}] execute rank={cand.get('rank')} task={task} "
            f"reason={cand.get('reason')}{reset}"
        )

        action_record = None
        pre_hierarchy_text = get_hierarchy_text(device)
        try:
            # step_result = execute_decider_one_step(
            #     decider_client=explorer_client,
            #     decider_model=explorer_model,
            #     device=device,
            #     device_type=device_type,
            #     step_task=task,
            #     use_qwen3=use_qwen3,
            #     output_dir=step_output_dir,
            #     step_index=step_idx,
            # )
            step_result = execute_decider_one_step(
                decider_client=decider_client,
                decider_model=decider_model,
                device=device,
                device_type=device_type,
                app_name=app_name,
                step_task=task,
                use_qwen3=use_qwen3,
                allow_hierarchy_text_decider=allow_hierarchy_text_decider,
                output_dir=step_output_dir,
                step_index=step_idx,
            )

            action_record = step_result["action_record"]
            react_item = step_result["react_item"]
            decider_resp = step_result["decider_response"]
            post_hierarchy_text = step_result.get("post_hierarchy_text", "")

            actions.append(action_record)
            reacts.append(react_item)
            current_path_actions.append(action_record)
            current_path_reacts.append(react_item)

            persist_step_output(step_output_dir, app_name, action_record, react_item)

            if current_depth + 1 >= depth_limit:
                path_counter[0] += 1
                path_id = path_counter[0]
                path_output_dir = os.path.join(paths_dir, f"path_{path_id:04d}")
                step_indices = [item.get("action_index") for item in current_path_actions if item.get("action_index")]
                copy_step_artifacts_to_path(steps_dir, path_output_dir, step_indices)
                persist_outputs(
                    path_output_dir,
                    app_name,
                    current_path_actions,
                    current_path_reacts,
                )
            else:
                explore_dfs(
                    app_name=app_name,
                    depth_limit=depth_limit,
                    breadth=breadth,
                    current_depth=current_depth + 1,
                    decider_client=decider_client,
                    decider_model=decider_model,
                    explorer_client=explorer_client,
                    explorer_model=explorer_model,
                    device=device,
                    device_type=device_type,
                    use_qwen3=use_qwen3,
                    allow_hierarchy_text_decider=allow_hierarchy_text_decider,
                    data_dir=data_dir,
                    actions=actions,
                    reacts=reacts,
                    step_counter=step_counter,
                    path_counter=path_counter,
                    steps_dir=steps_dir,
                    paths_dir=paths_dir,
                    path_actions=current_path_actions,
                    path_reacts=current_path_reacts,
                )

        except Exception as e:
            logging.error(f"Failed to execute candidate at depth {current_depth}: {e}")

        # 回溯：默认执行返回，再检查是否回到正确界面
        perform_backtrack_action(device, device_type, action_record)

        post_back_hierarchy = get_hierarchy_text(device)
        if not _is_app_in_foreground(device, device_type, app_name, post_back_hierarchy):
            logging.warning(
                "\033[91mBacktrack left app unexpectedly. Restarting app: %s\033[0m",
                app_name,
            )
            device.start_app(app_name)
            time.sleep(DEVICE_WAIT_TIME * 2)
            post_back_hierarchy = get_hierarchy_text(device)

        expected_fp = _hierarchy_fingerprint(pre_hierarchy_text)
        actual_fp = _hierarchy_fingerprint(post_back_hierarchy)
        if expected_fp and actual_fp and expected_fp != actual_fp:
            similarity_after_back = difflib.SequenceMatcher(
                None,
                _normalize_hierarchy_text(pre_hierarchy_text),
                _normalize_hierarchy_text(post_back_hierarchy),
            ).ratio()
            if len(current_path_actions) >= 2 and similarity_after_back < 0.9: #如果返回的不是上一级UI界面
                recovery_action = current_path_actions[-2]
                logging.warning(
                    "\033[93mBacktrack may overshoot (sim=%.3f). Replaying previous action(type=%s) to recover parent page.\033[0m",
                    similarity_after_back,
                    recovery_action.get("type", ""),
                )
                replay_action_record(device, recovery_action)
            else:
                logging.info(
                    "\033[93mBacktrack verification mismatch but no recovery action replay (sim=%.3f).\033[0m",
                    similarity_after_back,
                )

        cand_idx += 1


def init_decider_client(service_ip: str, decider_port: int) -> OpenAI:
    return OpenAI(api_key="0", base_url=f"http://{service_ip}:{decider_port}/v1")


def init_explorer_client(base_url: str, api_key: str) -> OpenAI:
    logging.info(f"Initializing explorer client with base_url={base_url}")
    logging.info(f"API Key is set: {bool(api_key)}")
    return OpenAI(api_key=api_key, base_url=base_url)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MobiAgent Auto Search (DFS + backtracking)")
    parser.add_argument("--app_name", type=str, required=True, help="目标App名称（与设备映射一致）")
    parser.add_argument("--depth", type=int, required=True, help="探索深度 D")
    parser.add_argument("--breadth", type=int, required=True, help="每层探索广度 H")

    parser.add_argument("--device", type=str, default="Android", choices=["Android", "Harmony"], help="设备类型")
    parser.add_argument("--service_ip", type=str, default="localhost", help="Decider 服务IP")
    parser.add_argument("--decider_port", type=int, default=8000, help="Decider 服务端口")

    parser.add_argument("--openrouter_base_url", type=str, default="https://openrouter.ai/api/v1", help="OpenRouter Base URL")
    parser.add_argument("--openrouter_api_key", type=str, default=os.getenv("OPENROUTER_API_KEY", ""), help="OpenRouter API Key")
    parser.add_argument("--explorer_model", type=str, default="google/gemini-3-flash-preview", help="通用大模型名称")

    parser.add_argument("--use_qwen3", choices=["on", "off"], default="on", help="是否按Qwen3坐标格式换算")
    parser.add_argument(
        "--allow_hierarchy_text_decider",
        choices=["on", "off"],
        default="on",
        help="是否允许使用层级文本作为 decider 输出动作",
    )
    parser.add_argument("--data_dir", type=str, default=None, help="结果目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.depth <= 0:
        raise ValueError("depth 必须 > 0")
    if args.breadth <= 0:
        raise ValueError("breadth 必须 > 0")
    if not args.openrouter_api_key:
        raise ValueError("请通过 --openrouter_api_key 或环境变量 OPENROUTER_API_KEY 提供密钥")

    # 数据目录
    if args.data_dir:
        data_dir = args.data_dir
    else:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        data_dir = str(Path(__file__).parent / "data-auto-search" / args.app_name / timestamp)
    os.makedirs(data_dir, exist_ok=True)

    # 设备
    if args.device == "Android":
        device = AndroidDevice()
    else:
        device = HarmonyDevice()

    # 客户端
    decider_client = init_decider_client(args.service_ip, args.decider_port)
    explorer_client = init_explorer_client(args.openrouter_base_url, args.openrouter_api_key)
    use_qwen3 = args.use_qwen3 == "on"
    allow_hierarchy_text_decider = args.allow_hierarchy_text_decider == "on"

    # 启动应用
    logging.info(f"Starting app: {args.app_name}")
    device.start_app(args.app_name)
    time.sleep(1.5)

    actions: List[Dict[str, Any]] = []
    reacts: List[Dict[str, Any]] = []
    step_counter = [0]  # 可变计数器，保证全局递增
    path_counter = [0]

    steps_dir = os.path.join(data_dir, "steps")
    paths_dir = os.path.join(data_dir, "paths")
    os.makedirs(steps_dir, exist_ok=True)
    os.makedirs(paths_dir, exist_ok=True)

    try:
        decider_model = DECIDER_MODEL_PLACEHOLDER

        explore_dfs(
            app_name=args.app_name,
            depth_limit=args.depth,
            breadth=args.breadth,
            current_depth=0,
            decider_client=decider_client,
            decider_model=decider_model,
            explorer_client=explorer_client,
            explorer_model=args.explorer_model,
            device=device,
            device_type=args.device,
            use_qwen3=use_qwen3,
            allow_hierarchy_text_decider=allow_hierarchy_text_decider,
            data_dir=data_dir,
            actions=actions,
            reacts=reacts,
            step_counter=step_counter,
            path_counter=path_counter,
            steps_dir=steps_dir,
            paths_dir=paths_dir,
        )
    finally:
        logging.info(f"Auto-search finished. data_dir={data_dir}")


if __name__ == "__main__":
    main()
