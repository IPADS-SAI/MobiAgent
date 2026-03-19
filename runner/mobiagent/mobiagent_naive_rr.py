#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# """
# MobiAgent Naïve Record & Replay Version
# 最小化依赖的版本 - record+replay
# """

from openai import OpenAI
import base64
from PIL import Image, ImageDraw, ImageFont
import json
import io
import logging
import time
import re
import ast
import os
import argparse
import sys
import importlib.util
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import hashlib
import math
from collections import Counter
import xml.etree.ElementTree as ET
# from xml_box_matcher import (
#     match_pred_box_to_xml,
#     validate_match,
#     match_result_to_dict,
#     validation_result_to_dict,
#     node_center,
# )

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger(__name__)

MAX_STEPS = 15  # 最大步骤数
SCREENSHOT_FACTOR = 0.5  # 截图压缩比例
DECIDER_PARSE_MAX_RETRIES = 3  # Decider解析失败后重试次数
DEVICE_WAIT_TIME = 0.5  # 设备动作后的基础等待时间
TOCTOU_IMAGE_SIM_MIN = 0.95  # strict image similarity guard before execute
TOCTOU_MAX_RETRY_PER_STEP = 2  # retry decider in same step when TOCTOU fails
TOCTOU_HASH_SIZE = 16  # aHash size used by TOCTOU image comparison

# Replay decision thresholds (auto-inspect whether past trace should be reused)
REPLAY_USE_MIN_SCORE_BY_STAGE = {
    # Keep these aligned with stage match thresholds to avoid "matched but not replayed".
    "edit_or_keyword": 0.70,
    "bm25": 0.45,
    "vector": 0.58,
}
REPLAY_USE_MAX_ACTIONS_WITHOUT_DONE = 8
REPLAY_USE_MAX_REPEAT_RATIO = 0.72
REPLAY_XML_OVERLAP_MIN = 0.55
GRAPH_REPLAY_SEARCH_MODE = "conservative"
GRAPH_REPLAY_MAX_CANDIDATES_PER_STEP = 6

# No-progress guard (if page XML does not change for 3 rounds, stop)
NO_PROGRESS_XML_STREAK = 3

# 全局客户端
decider_client = None
grounder_client = None
planner_client = None


def _init_template_graph_builder(
    graph_enabled: bool,
    graph_dir: Optional[str],
    app_name: str,
) -> Tuple[Optional[Any], Optional[str], str]:
    """
    Lazy init template graph builder.
    Returns: (builder, graph_file, error_message)
    """
    if not graph_enabled:
        return None, None, ""
    if not graph_dir:
        return None, None, "graph_dir is empty"
    try:
        from template_graph_builder import TemplateGraphBuilder  # local module
    except Exception as e:
        return None, None, f"import failed: {e}"
    try:
        builder = TemplateGraphBuilder(graph_dir=graph_dir, app_name=app_name)
        return builder, getattr(builder, "graph_file", None), ""
    except Exception as e:
        return None, None, f"init failed: {e}"


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


def stable_task_hash(task_desc: str) -> str:
    """为任务描述生成跨进程稳定的短哈希"""
    return hashlib.md5(task_desc.encode("utf-8")).hexdigest()[:8]



# --- Matching helpers (for ExperienceRR fuzzy replay) ---

def normalize_task_text(text: str) -> str:
    """Normalize task description for fuzzy matching."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_tokens(text: str) -> List[str]:
    """Tokenize text into mixed alnum and CJK tokens (lightweight)."""
    if not text:
        return []
    text = text.lower()
    tokens = re.findall(r"[0-9a-z]+|[\u4e00-\u9fff]", text)
    # add CJK bigrams for a bit more context
    cjk = "".join(re.findall(r"[\u4e00-\u9fff]", text))
    if len(cjk) > 1:
        tokens.extend([cjk[i:i+2] for i in range(len(cjk) - 1)])
    return tokens


def levenshtein_distance(a: str, b: str) -> int:
    """Compute Levenshtein edit distance (iterative DP)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (0 if ca == cb else 1)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def edit_similarity(a: str, b: str) -> float:
    """Edit-distance similarity in [0,1]."""
    max_len = max(len(a), len(b), 1)
    return 1.0 - (levenshtein_distance(a, b) / max_len)


def jaccard_similarity(tokens_a: List[str], tokens_b: List[str]) -> float:
    if not tokens_a or not tokens_b:
        return 0.0
    set_a, set_b = set(tokens_a), set(tokens_b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def bm25_scores(query_tokens: List[str], doc_tokens_list: List[List[str]], k1: float = 1.5, b: float = 0.75) -> List[float]:
    """Compute BM25 scores for a query against multiple docs."""
    if not doc_tokens_list:
        return []
    query_terms = list(dict.fromkeys(query_tokens))
    N = len(doc_tokens_list)
    doc_lens = [len(doc) for doc in doc_tokens_list]
    avgdl = (sum(doc_lens) / N) if N else 0.0
    df = Counter()
    for doc in doc_tokens_list:
        df.update(set(doc))
    idf = {}
    for term, freq in df.items():
        idf[term] = math.log((N - freq + 0.5) / (freq + 0.5) + 1.0)
    scores = []
    for doc, doc_len in zip(doc_tokens_list, doc_lens):
        if doc_len == 0:
            scores.append(0.0)
            continue
        tf = Counter(doc)
        score = 0.0
        for term in query_terms:
            if term not in tf:
                continue
            freq = tf[term]
            denom = freq + k1 * (1.0 - b + b * (doc_len / (avgdl or 1.0)))
            score += idf.get(term, 0.0) * (freq * (k1 + 1.0)) / (denom or 1.0)
        scores.append(score)
    return scores


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


_EMBED_MODEL = None
_EMBED_TOKENIZER = None

def _load_embedder(model_path: Optional[str] = None) -> bool:
    """Lazy-load embedding model for vector matching."""
    global _EMBED_MODEL, _EMBED_TOKENIZER
    if _EMBED_MODEL is not None and _EMBED_TOKENIZER is not None:
        return True
    try:
        from transformers import AutoTokenizer, AutoModel  # type: ignore
        import torch  # type: ignore
    except Exception as e:
        logger.warning(f"[MATCH] Embedding backend unavailable: {e}")
        return False
    if model_path is None:
        repo_root = Path(__file__).resolve().parents[2]
        model_path = str(repo_root / "bge-small-zh")
    if not os.path.exists(model_path):
        logger.warning(f"[MATCH] Embedding model not found: {model_path}")
        return False
    _EMBED_TOKENIZER = AutoTokenizer.from_pretrained(model_path)
    _EMBED_MODEL = AutoModel.from_pretrained(model_path)
    _EMBED_MODEL.eval()
    return True


def get_text_embedding(text: str) -> Optional[List[float]]:
    """Compute a sentence embedding (mean pooling, L2-normalized)."""
    if not text:
        return None
    if not _load_embedder():
        return None
    try:
        import torch  # type: ignore
        inputs = _EMBED_TOKENIZER(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            model_output = _EMBED_MODEL(**inputs)
        token_embeddings = model_output[0]
        attention_mask = inputs["attention_mask"]
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = (token_embeddings * mask).sum(1)
        counts = mask.sum(1).clamp(min=1e-9)
        emb = summed / counts
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.squeeze(0).cpu().tolist()
    except Exception as e:
        logger.warning(f"[MATCH] Embedding failed: {e}")
        return None
def convert_qwen3_coordinates_to_absolute(bbox_or_coords, img_width, img_height, is_bbox=True):
    """
    将 Qwen3 模型返回的相对坐标（0-1000范围）转换为绝对坐标
    """
    if is_bbox:
        x1, y1, x2, y2 = bbox_or_coords
        x1 = int(x1 / 1000 * img_width)
        x2 = int(x2 / 1000 * img_width)
        y1 = int(y1 / 1000 * img_height)
        y2 = int(y2 / 1000 * img_height)
        return [x1, y1, x2, y2]
    x, y = bbox_or_coords
    x = int(x / 1000 * img_width)
    y = int(y / 1000 * img_height)
    return [x, y]


def _parse_android_bounds(bounds_str: str) -> Optional[List[int]]:
    """Parse Android bounds string: [x1,y1][x2,y2]."""
    if not bounds_str:
        return None
    m = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", bounds_str)
    if not m:
        return None
    return [int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))]


def _bbox_center(bbox: List[int]) -> Tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) // 2, (y1 + y2) // 2


def _bbox_area(bbox: List[int]) -> int:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def _point_in_bbox(x: int, y: int, bbox: List[int]) -> bool:
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]


def _extract_android_node_meta(node_attrib: Dict[str, str]) -> Dict[str, str]:
    """Keep a stable subset of XML node attributes for replay matching."""
    keys = [
        "resource-id", "text", "content-desc", "class", "package",
        "clickable", "enabled", "selected", "checked", "focused",
        "scrollable", "long-clickable", "bounds",
    ]
    return {k: node_attrib.get(k, "") for k in keys}


def extract_android_ui_meta_by_bbox(hierarchy_xml: str, bbox: List[int]) -> Optional[Dict[str, str]]:
    """Find the smallest XML node containing bbox center and return stable metadata."""
    try:
        root = ET.fromstring(hierarchy_xml)
    except Exception as e:
        logger.warning(f"[XML] Failed to parse hierarchy XML: {e}")
        return None

    cx, cy = _bbox_center(bbox)
    best = None
    best_area = None
    for elem in root.iter():
        b = _parse_android_bounds(elem.attrib.get("bounds", ""))
        if not b:
            continue
        if _point_in_bbox(cx, cy, b):
            area = _bbox_area(b)
            if best is None or area < best_area:
                best = elem
                best_area = area
    if best is None:
        return None
    return _extract_android_node_meta(best.attrib)


def extract_android_ui_meta_by_point(hierarchy_xml: str, x: int, y: int) -> Optional[Dict[str, str]]:
    bbox = [x, y, x, y]
    return extract_android_ui_meta_by_bbox(hierarchy_xml, bbox)


def strict_ui_meta_equal(expected: Optional[Dict[str, str]], current: Optional[Dict[str, str]]) -> bool:
    if not expected or not current:
        return False
    return expected == current


def _ui_meta_match_keys() -> List[str]:
    return [
        "resource-id",
        "class",
        "package",
        "text",
        "content-desc",
        "clickable",
        "enabled",
        "selected",
        "checked",
        "focused",
        "scrollable",
        "long-clickable",
    ]


def find_android_point_by_ui_meta(
    hierarchy_xml: Optional[str],
    expected_ui_meta: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Resolve click point by exact UI metadata match on current XML."""
    if not hierarchy_xml or not isinstance(expected_ui_meta, dict):
        return None
    keys = _ui_meta_match_keys()
    constrained = {
        key: str(expected_ui_meta.get(key, "")).strip()
        for key in keys
        if str(expected_ui_meta.get(key, "")).strip()
    }
    if not constrained:
        return None

    try:
        root = ET.fromstring(hierarchy_xml)
    except Exception as e:
        logger.warning(f"[XML] Failed to parse hierarchy XML for ui_meta match: {e}")
        return None

    best: Optional[Dict[str, Any]] = None
    for elem in root.iter():
        attrib = elem.attrib or {}
        bounds = _parse_android_bounds(attrib.get("bounds", ""))
        if not bounds:
            continue
        ok = True
        matched_keys = 0
        for key, expected in constrained.items():
            if str(attrib.get(key, "")).strip() != expected:
                ok = False
                break
            matched_keys += 1
        if not ok:
            continue
        area = _bbox_area(bounds)
        cx, cy = _bbox_center(bounds)
        candidate = {
            "position": [int(cx), int(cy)],
            "ui_meta": _extract_android_node_meta(attrib),
            "matched_keys": matched_keys,
            "area": area,
        }
        if best is None:
            best = candidate
            continue
        if candidate["matched_keys"] > best["matched_keys"]:
            best = candidate
            continue
        if candidate["matched_keys"] == best["matched_keys"] and candidate["area"] < best["area"]:
            best = candidate
    return best


def _xml_tokens(hierarchy_xml: str) -> List[str]:
    """Extract coarse page tokens for lightweight XML page matching."""
    try:
        root = ET.fromstring(hierarchy_xml)
    except Exception:
        return []
    tokens: List[str] = []
    for elem in root.iter():
        attrib = elem.attrib
        if not attrib:
            continue
        token = "|".join([
            attrib.get("class", ""),
            attrib.get("resource-id", ""),
            attrib.get("text", ""),
            attrib.get("content-desc", ""),
            attrib.get("bounds", ""),
        ])
        if token.strip("|"):
            tokens.append(token)
    return tokens


def build_xml_page_signature(hierarchy_xml: Optional[str]) -> str:
    """Build a stable hash for no-progress detection."""
    if not hierarchy_xml:
        return ""
    tokens = _xml_tokens(hierarchy_xml)
    payload = "\n".join(tokens) if tokens else hierarchy_xml.strip()
    return hashlib.md5(payload.encode("utf-8", errors="ignore")).hexdigest()


def evaluate_replay_xml_match(
    current_hierarchy_xml: Optional[str],
    recorded_xml_path: Optional[str],
) -> Dict[str, Any]:
    """Check whether current XML roughly matches recorded XML for this replay step."""
    result: Dict[str, Any] = {
        "allow_replay": False,
        "reason": "",
        "metrics": {},
        "recorded_xml": recorded_xml_path,
    }
    if not current_hierarchy_xml:
        result["reason"] = "current xml missing"
        return result
    if not recorded_xml_path or not os.path.exists(recorded_xml_path):
        result["reason"] = "recorded xml missing"
        return result
    try:
        with open(recorded_xml_path, "r", encoding="utf-8") as f:
            recorded_xml = f.read()
    except Exception as e:
        result["reason"] = f"read recorded xml failed: {e}"
        return result

    current_tokens = set(_xml_tokens(current_hierarchy_xml))
    recorded_tokens = set(_xml_tokens(recorded_xml))
    if not current_tokens or not recorded_tokens:
        result["reason"] = "xml tokenization failed"
        return result

    overlap = len(current_tokens & recorded_tokens) / max(len(recorded_tokens), 1)
    result["metrics"]["xml_overlap"] = round(float(overlap), 3)
    if overlap < REPLAY_XML_OVERLAP_MIN:
        result["reason"] = f"xml overlap too low ({overlap:.3f} < {REPLAY_XML_OVERLAP_MIN:.3f})"
        return result

    result["allow_replay"] = True
    result["reason"] = "xml matched"
    return result


# def _resolve_click_with_xml_match(
#     hierarchy_xml: Optional[str],
#     pred_bbox: List[int],
#     action_type: str,
#     expected_text: Optional[str],
#     target_package: Optional[str],
# ) -> Dict[str, Any]:
#     """Resolve predicted bbox to XML node center before executing click."""
#     if not hierarchy_xml:
#         return {
#             "ok": False,
#             "error": {
#                 "ok": False,
#                 "status": "VALIDATION_FAIL",
#                 "error_code": "XML_MISSING",
#                 "message": "Hierarchy XML is missing before click validation",
#             },
#         }

#     match_result = match_pred_box_to_xml(
#         xml_str=hierarchy_xml,
#         pred_box=pred_bbox,
#         action_type=action_type,
#         expected_text=expected_text,
#         target_package=target_package,
#     )
#     validation_result = validate_match(
#         match_result=match_result,
#         expected_text=expected_text,
#         action_type=action_type,
#         text_threshold=0.45,
#         require_clickable=True,
#     )

#     match_payload = match_result_to_dict(match_result, include_candidates=5)
#     validation_payload = validation_result_to_dict(validation_result)
#     if not validation_result.ok or validation_result.click_node is None:
#         return {
#             "ok": False,
#             "error": {
#                 **validation_payload,
#                 "match": match_payload,
#             },
#         }

#     click_node = validation_result.click_node
#     x, y = node_center(click_node)
#     return {
#         "ok": True,
#         "position": [int(x), int(y)],
#         "bbox": [int(v) for v in click_node.bbox],
#         "match": match_payload,
#         "validation": validation_payload,
#     }


def _build_validation_error_record(
    step: int,
    source_action: str,
    target: str,
    pred_bbox: List[int],
    screenshot_file: str,
    xml_file: Optional[str],
    error_payload: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "step": step,
        "action": "validation_error",
        "source_action": source_action,
        "target": target,
        "pred_bbox": [int(v) for v in pred_bbox],
        "screenshot_file": screenshot_file,
        "xml_file": xml_file,
        "validation_error": error_payload,
    }


def dump_hierarchy_for_step(device: Any, data_dir: str, step: int) -> Tuple[Optional[str], Optional[str]]:
    """Dump current hierarchy XML and save as step file."""
    xml_file = f"{step:02d}.xml"
    xml_path = os.path.join(data_dir, xml_file)
    try:
        hierarchy = device.dump_hierarchy()
        if hierarchy is None:
            return None, None
        if not isinstance(hierarchy, str):
            hierarchy = str(hierarchy)
        with open(xml_path, "w", encoding="utf-8") as f:
            f.write(hierarchy)
        return hierarchy, xml_file
    except Exception as e:
        logger.warning(f"[XML] Failed to dump hierarchy: {e}")
        return None, None


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
            "浏览器": "com.microsoft.emmx",
            "知乎": "com.zhihu.android",
            "设置": "com.android.settings"
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

    def dump_hierarchy(self):
        """导出当前页面层级XML。"""
        try:
            return self.d.dump_hierarchy()
        except Exception as e:
            logger.error(f"[ERROR] 导出层级失败: {e}")
            raise

    def click(self, x, y):
        """点击屏幕"""
        try:
            self.d.click(x, y)
            time.sleep(0.5)
            logger.info(f"[OK] 已点击: ({x}, {y})")
        except Exception as e:
            logger.error(f"[ERROR] 点击失败: {e}")

    def clear_input(self):
        """清空当前输入框内容（假设输入框已获得焦点）。"""
        try:
            # 与 mobiagent.py 一致：移动光标并删除已有内容。
            self.d.shell(['input', 'keyevent', 'KEYCODE_MOVE_END'])
            self.d.shell(['input', 'keyevent', 'KEYCODE_MOVE_HOME'])
            self.d.shell(['input', 'keyevent', 'KEYCODE_DEL'])
            logger.info("[INPUT] 已尝试清空输入框默认内容")
            return True
        except Exception as e:
            logger.warning(f"[INPUT] 清空输入框失败: {e}")
            return False

    def input_text(self, text):
        """输入文本（与 mobiagent.py 对齐：ADBKeyboard + ADB_CLEAR_TEXT + B64 + Enter）"""
        current_ime = None
        try:
            logger.info(f"[INPUT] 开始输入文本: {text}")
            current_ime = self.d.current_ime()
            logger.info(f"[INPUT] 当前输入法: {current_ime}")

            self.d.shell([
                'settings', 'put', 'secure', 'default_input_method',
                'com.android.adbkeyboard/.AdbIME'
            ])
            time.sleep(DEVICE_WAIT_TIME)

            # 依赖 ADB Keyboard 的清空输入框命令
            self.d.shell(['am', 'broadcast', '-a', 'ADB_CLEAR_TEXT'])
            time.sleep(0.2)

            charsb64 = base64.b64encode(text.encode('utf-8')).decode('utf-8')
            self.d.shell(['am', 'broadcast', '-a', 'ADB_INPUT_B64', '--es', 'msg', charsb64])
            time.sleep(DEVICE_WAIT_TIME)

            self.d.shell(['settings', 'put', 'secure', 'default_input_method', current_ime])
            # 按下回车确认输入，与 mobiagent.py 一致
            self.d.shell(['input', 'keyevent', 'KEYCODE_ENTER'])
            logger.info(f"[OK] 已输入文本(ADB IME): {text}")
            return True
        except Exception as e:
            logger.error(f"[ERROR] 输入失败: {e}")
            # 出错时尽力恢复原输入法，避免污染后续动作
            if current_ime:
                try:
                    self.d.shell(['settings', 'put', 'secure', 'default_input_method', current_ime])
                except Exception:
                    pass
            return False

    def swipe(self, direction):
        """滑动屏幕"""
        try:
            # Normalize direction (support Chinese and English)
            dir_map = {
                '上': 'up',
                '向上': 'up',
                '上滑': 'up',
                '下': 'down',
                '向下': 'down',
                '下滑': 'down',
                '左': 'left',
                '向左': 'left',
                '左滑': 'left',
                '右': 'right',
                '向右': 'right',
                '右滑': 'right',
            }
            if isinstance(direction, str):
                norm = direction.strip().lower()
                direction = dir_map.get(norm, norm)
            self.d.swipe_ext(direction=direction, scale=0.6)
            logger.info(f"[OK] 已滑动: {direction}")
        except Exception as e:
            logger.error(f"[ERROR] 滑动失败: {e}")


def get_screenshot_b64(device=None, compress=True, screenshot_path: Optional[str] = None):
    """
    获取截图并转换为Base64，返回原始分辨率信息
    
    Args:
        device: 设备对象
        compress: 是否压缩（True=50%, False=原始分辨率）
    """
    try:
        path = screenshot_path
        if not path or (not os.path.exists(path)):
            if device is None:
                logger.error("[ERROR] screenshot path missing and device is None")
                return None, None, None
            path = device.screenshot()
        img = Image.open(path)
        original_width, original_height = img.width, img.height
        logger.info(f"[DEBUG] 原始截图分辨率: {original_width}x{original_height}")
        
        if compress:
            # 压缩截图
            compressed_width = int(img.width * SCREENSHOT_FACTOR)
            compressed_height = int(img.height * SCREENSHOT_FACTOR)
            img = img.resize((compressed_width, compressed_height), 
                            Image.Resampling.LANCZOS)
            logger.info(f"[DEBUG] 压缩后截图分辨率: {compressed_width}x{compressed_height}")
        else:
            logger.info(f"[DEBUG] 使用原始分辨率（未压缩）")
        
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8"), original_width, original_height
    except Exception as e:
        logger.error(f"[ERROR] Base64转换失败: {e}")
        return None, None, None


def _image_ahash_bits(image_path: str, hash_size: int = TOCTOU_HASH_SIZE) -> Optional[List[int]]:
    """Compute aHash bits for one image."""
    if not image_path or not os.path.exists(image_path):
        return None
    try:
        with Image.open(image_path) as img:
            gray = img.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
            pixels = list(gray.getdata())
    except Exception as e:
        logger.warning(f"[TOCTOU] Failed to load image for hash ({image_path}): {e}")
        return None
    if not pixels:
        return None
    avg = float(sum(int(p) for p in pixels)) / max(len(pixels), 1)
    return [1 if int(p) >= avg else 0 for p in pixels]


def compute_image_ahash_similarity(
    image_path_a: str,
    image_path_b: str,
    hash_size: int = TOCTOU_HASH_SIZE,
) -> Optional[float]:
    """Compute aHash similarity in [0,1], higher means more similar."""
    bits_a = _image_ahash_bits(image_path_a, hash_size=hash_size)
    bits_b = _image_ahash_bits(image_path_b, hash_size=hash_size)
    if bits_a is None or bits_b is None:
        return None
    if len(bits_a) != len(bits_b) or not bits_a:
        return None
    hamming = sum(1 for a, b in zip(bits_a, bits_b) if a != b)
    return 1.0 - (hamming / len(bits_a))


def evaluate_toctou_pre_action(
    device: Any,
    data_dir: str,
    step: int,
    attempt: int,
    reference_screenshot_path: Optional[str],
) -> Dict[str, Any]:
    """
    Before executing action, capture one fresh screenshot and compare with model reference.
    """
    result: Dict[str, Any] = {
        "allow": False,
        "score": 0.0,
        "threshold": TOCTOU_IMAGE_SIM_MIN,
        "reason": "",
        "attempt": int(attempt),
        "check_image_file": "",
        "check_image_path": None,
    }
    if not reference_screenshot_path or not os.path.exists(reference_screenshot_path):
        result["reason"] = "reference screenshot missing"
        return result

    check_image_file = f"{step:02d}_toctou_check_a{int(attempt)}.jpg"
    check_image_path = os.path.join(data_dir, check_image_file)
    result["check_image_file"] = check_image_file
    result["check_image_path"] = check_image_path

    try:
        raw_path = device.screenshot()
        if raw_path and os.path.exists(raw_path):
            if os.path.abspath(raw_path) != os.path.abspath(check_image_path):
                shutil.copy(raw_path, check_image_path)
        elif not os.path.exists(check_image_path):
            result["reason"] = "capture check screenshot failed"
            return result
    except Exception as e:
        result["reason"] = f"capture check screenshot failed: {e}"
        return result

    if not os.path.exists(check_image_path):
        result["reason"] = "check screenshot missing"
        return result

    score = compute_image_ahash_similarity(
        reference_screenshot_path,
        check_image_path,
        hash_size=TOCTOU_HASH_SIZE,
    )
    if score is None:
        result["reason"] = "image similarity unavailable"
        return result

    result["score"] = round(float(score), 4)
    if score >= TOCTOU_IMAGE_SIM_MIN:
        result["allow"] = True
        result["reason"] = "image matched"
    else:
        result["reason"] = (
            f"image drift ({score:.4f} < {TOCTOU_IMAGE_SIM_MIN:.4f})"
        )
    return result


def _try_load_json_obj(text: str) -> Optional[Dict[str, Any]]:
    """尝试把文本解析为JSON对象，兼容少量模型输出格式噪声。"""
    if text is None:
        return None

    candidate = text.strip()
    if not candidate:
        return None

    for normalized in (
        candidate,
        candidate.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'"),
    ):
        try:
            parsed = json.loads(normalized)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        # 兼容单引号风格字典（如 {'action': 'click'}）
        try:
            parsed = ast.literal_eval(normalized)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    return None


def _extract_balanced_json_obj(text: str) -> Optional[Dict[str, Any]]:
    """从混合文本中提取首个花括号平衡的JSON对象（支持嵌套JSON）。"""
    if not text:
        return None

    n = len(text)
    for start in range(n):
        if text[start] != "{":
            continue
        depth = 0
        in_str = False
        escape = False
        for i in range(start, n):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == "\"":
                    in_str = False
                continue

            if ch == "\"":
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    parsed = _try_load_json_obj(text[start:i + 1])
                    if parsed is not None:
                        return parsed
                    break
    return None


def parse_json_response(response_str):
    """解析JSON响应（支持前置文本、代码块、嵌套JSON对象）。"""
    if response_str is None:
        logger.warning("[WARNING] 无法解析JSON: <None>")
        return None

    raw = str(response_str).strip()
    if not raw:
        logger.warning("[WARNING] 无法解析JSON: <empty>")
        return None

    # 1) 直接解析
    parsed = _try_load_json_obj(raw)
    if parsed is not None:
        return parsed

    # 2) 从```json ...```/```...```代码块解析
    for pattern in (r"```json\s*([\s\S]*?)\s*```", r"```\s*([\s\S]*?)\s*```"):
        for match in re.finditer(pattern, raw, re.MULTILINE):
            parsed = _try_load_json_obj(match.group(1))
            if parsed is not None:
                return parsed
            parsed = _extract_balanced_json_obj(match.group(1))
            if parsed is not None:
                return parsed

    # 3) 从前置文本中提取平衡JSON对象
    parsed = _extract_balanced_json_obj(raw)
    if parsed is not None:
        return parsed

    logger.warning(f"[WARNING] 无法解析JSON: {raw[:200]}")
    return None


def visualize_click(screenshot_path: str, x: int, y: int, target: str, output_path: str):
    """
    在截图上标注点击位置
    
    Args:
        screenshot_path: 原始截图路径
        x: 点击的X坐标
        y: 点击的Y坐标
        target: 点击目标的描述
        output_path: 输出的标注图路径
    """
    try:
        img = Image.open(screenshot_path)
        draw = ImageDraw.Draw(img)
        
        # 绘制十字
        cross_size = 30
        draw.line([(x - cross_size, y), (x + cross_size, y)], fill=(0, 255, 0), width=3)
        draw.line([(x, y - cross_size), (x, y + cross_size)], fill=(0, 255, 0), width=3)
        
        # 绘制圆圈
        circle_radius = 50
        draw.ellipse(
            [(x - circle_radius, y - circle_radius), 
             (x + circle_radius, y + circle_radius)], 
            outline=(0, 255, 0), width=3
        )
        
        # 添加文字标签
        try:
            font = ImageFont.truetype("arial.ttf", 30)
        except:
            font = ImageFont.load_default()
        
        text = f"CLICK: {target}\n({x}, {y})"
        text_x, text_y = max(10, x - 100), max(10, y - 100)
        draw.text((text_x, text_y), text, fill=(0, 255, 0), font=font)
        
        # 保存标注图
        img.save(output_path)
        logger.info(f"[VISUALIZE] 已保存标注图: {output_path}")
        
    except Exception as e:
        logger.warning(f"[WARNING] 可视化失败: {e}")


def visualize_model_bbox(screenshot_path: str, bbox: List[int], output_path: str, label: str = ""):
    """
    在截图上绘制模型输出范围（红框）

    Args:
        screenshot_path: 原始截图路径
        bbox: [x1, y1, x2, y2]
        output_path: 输出路径（如 xx_bounds.jpg）
        label: 可选标签文本
    """
    try:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        img = Image.open(screenshot_path)
        draw = ImageDraw.Draw(img)

        # 红框（与 mobiagent.py 的 bounds 图一致）
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=5)

        if label:
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except Exception:
                font = ImageFont.load_default()
            text_y = max(5, y1 - 30)
            draw.text((x1, text_y), label, fill=(255, 0, 0), font=font)

        img.save(output_path)
        logger.info(f"[VISUALIZE] 已保存模型输出范围: {output_path}")
    except Exception as e:
        logger.warning(f"[WARNING] 模型范围可视化失败: {e}")


def _load_decider_prompt_constants():
    """Load decider prompt constants with robust fallback across different CWDs."""
    try:
        from prompts.decider_qwen3_e2e import (
            DECIDER_SYSTEM_PROMPT,
            DECIDER_USER_PROMPT,
            DECIDER_CURRENT_STEP_PROMPT,
        )
        return DECIDER_SYSTEM_PROMPT, DECIDER_USER_PROMPT, DECIDER_CURRENT_STEP_PROMPT
    except ModuleNotFoundError:
        pass

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        from prompts.decider_qwen3_e2e import (
            DECIDER_SYSTEM_PROMPT,
            DECIDER_USER_PROMPT,
            DECIDER_CURRENT_STEP_PROMPT,
        )
        return DECIDER_SYSTEM_PROMPT, DECIDER_USER_PROMPT, DECIDER_CURRENT_STEP_PROMPT
    except ModuleNotFoundError:
        prompt_py = repo_root / "prompts" / "decider_qwen3_e2e.py"
        if not prompt_py.exists():
            raise

        spec = importlib.util.spec_from_file_location(
            "decider_qwen3_e2e_fallback",
            str(prompt_py),
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load prompt module: {prompt_py}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return (
            module.DECIDER_SYSTEM_PROMPT,
            module.DECIDER_USER_PROMPT,
            module.DECIDER_CURRENT_STEP_PROMPT,
        )


def build_decider_messages(task: str, history: List[str], screenshot_b64: str):
    (
        DECIDER_SYSTEM_PROMPT,
        DECIDER_USER_PROMPT,
        DECIDER_CURRENT_STEP_PROMPT,
    ) = _load_decider_prompt_constants()

    if len(history) == 0:
        history_str = "(No history)"
    else:
        history_str = "\n".join(f"{idx}. {h}" for idx, h in enumerate(history, 1))

    context_text = DECIDER_USER_PROMPT.format(task=task, history=history_str)
    instruction_text = DECIDER_CURRENT_STEP_PROMPT

    return [
        {
            "role": "system",
            "content": DECIDER_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": context_text,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{screenshot_b64}",
                    },
                },
                {
                    "type": "text",
                    "text": instruction_text,
                },
            ],
        },
    ]


def get_decider_action(task: str, history: List[str], screenshot_b64: str, model: str = ""):
    """调用Decider模型获取动作"""
    try:
        messages = build_decider_messages(task, history, screenshot_b64)
        response = decider_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"[ERROR] Decider调用失败: {e}")
        return None


def get_grounder_coords(target_element, screenshot_b64, model="", use_qwen3=True):
    """调用Grounder模型获取坐标"""
    try:
        if use_qwen3:
            prompt = f"""在给定的UI截图中，找到以下元素的坐标:
{target_element}

请返回JSON:
{{
    "bbox": [x1, y1, x2, y2]
}}

其中[x1, y1, x2, y2]是元素的边界框坐标，坐标为0-1000的相对归一化数值（相对于图像宽高）。"""
        else:
            prompt = f"""在给定的UI截图中，找到以下元素的坐标:
{target_element}

请返回JSON:
{{
    "bbox": [x1, y1, x2, y2]
}}

其中[x1, y1, x2, y2]是元素在截图中的像素坐标。"""
        
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


class ExperienceRR:
    """Simple Record & Replay manager with fuzzy matching."""

    EDIT_SIM_THRESHOLD = 0.70
    KEYWORD_JACCARD_THRESHOLD = 0.30
    BM25_MIN_SCORE = 0.45
    VECTOR_SIM_THRESHOLD = 0.58

    def __init__(
        self,
        data_dir: str,
        task_name: str,
        task_desc: Optional[str] = None,
        search_root: Optional[str] = None,
        enable_load: bool = True,
    ):
        """
        Initialize Experience RR.

        Args:
            data_dir: data directory for current task
            task_name: task name for experience file naming
            task_desc: current task description
            search_root: root directory to search past experiences
        """
        self.data_dir = data_dir
        self.task_name = task_name
        self.task_desc = task_desc or ""
        self.search_root = search_root or os.path.dirname(data_dir)
        self.experience_file = os.path.join(data_dir, f"{task_name}_experience.json")
        self.actions: List[Dict[str, Any]] = []
        self.match_info: Optional[Dict[str, Any]] = None
        self.loaded_from_dir: Optional[str] = None
        self.loaded_experience_path: Optional[str] = None

        # Optional historical load. For graph-first runtime we disable it and keep record-only mode.
        if enable_load:
            self.load_experience()

    def _list_experience_files(self) -> List[str]:
        if not self.search_root or not os.path.isdir(self.search_root):
            return []
        paths = []
        for root, _, files in os.walk(self.search_root):
            for name in files:
                if name.startswith("task_") and name.endswith("_experience.json"):
                    paths.append(os.path.join(root, name))
        return paths

    def _load_task_description(self, experience_path: str) -> Optional[str]:
        # 1) preferred: task_description in experience file
        try:
            with open(experience_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            desc = data.get("task_description")
            if desc:
                return desc
        except Exception:
            pass
        # 2) fallback: actions_record.json in same folder
        actions_path = os.path.join(os.path.dirname(experience_path), "actions_record.json")
        if os.path.exists(actions_path):
            try:
                with open(actions_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data.get("task_description")
            except Exception:
                return None
        return None

    def _load_experience_file(self, experience_path: str):
        try:
            with open(experience_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.actions = self._normalize_actions(data.get("actions", []))
            self.loaded_from_dir = os.path.dirname(experience_path)
            self.loaded_experience_path = experience_path
            logger.info(f"[REPLAY] Loaded {len(self.actions)} actions from: {experience_path}")
        except Exception as e:
            logger.warning(f"[WARNING] Failed to load experience file: {e}")
            self.actions = []
            self.loaded_from_dir = None
            self.loaded_experience_path = None

    def _normalize_actions(self, actions: Any) -> List[Dict[str, Any]]:
        """Normalize and sort actions by step for deterministic replay order."""
        if not isinstance(actions, list):
            return []
        normalized: List[Dict[str, Any]] = []
        for idx, action in enumerate(actions):
            if not isinstance(action, dict):
                continue
            item = dict(action)
            step = item.get("step")
            if isinstance(step, str) and step.isdigit():
                item["step"] = int(step)
            item["_order"] = idx
            normalized.append(item)
        normalized.sort(
            key=lambda x: (
                x["step"] if isinstance(x.get("step"), int) else 10**9,
                x.get("_order", 10**9),
            )
        )
        for item in normalized:
            item.pop("_order", None)
        return normalized

    def _action_signature(self, action: Dict[str, Any]) -> str:
        """Build a lightweight signature for repetition analysis."""
        name = str(action.get("action", "")).lower()
        if name == "click":
            target = normalize_task_text(str(action.get("target", "")))
            position = action.get("position")
            if (
                isinstance(position, list)
                and len(position) == 2
                and all(isinstance(v, (int, float)) for v in position)
            ):
                x_bin = int(float(position[0]) // 60)
                y_bin = int(float(position[1]) // 60)
                return f"click:{target}:{x_bin}:{y_bin}"
            return f"click:{target}"
        if name == "click_input":
            target = normalize_task_text(str(action.get("target", "")))
            text = normalize_task_text(str(action.get("text", "")))
            position = action.get("position")
            if (
                isinstance(position, list)
                and len(position) == 2
                and all(isinstance(v, (int, float)) for v in position)
            ):
                x_bin = int(float(position[0]) // 60)
                y_bin = int(float(position[1]) // 60)
                return f"click_input:{target}:{text}:{x_bin}:{y_bin}"
            return f"click_input:{target}:{text}"
        if name == "input":
            text = normalize_task_text(str(action.get("text", "")))
            return f"input:{text}"
        if name == "swipe":
            direction = normalize_task_text(str(action.get("direction", "")))
            return f"swipe:{direction}"
        return name

    def inspect_replay_usability(self) -> Dict[str, Any]:
        """
        Decide whether to use loaded replay actions.
        This is the key gate that turns replay on/off automatically in auto mode.
        """
        info: Dict[str, Any] = {
            "use_replay": False,
            "reason": "",
            "stage": "",
            "score": 0.0,
            "replay_action_count": 0,
            "has_done": False,
        }

        if not self.actions:
            info["reason"] = "no actions loaded"
            return info
        if not self.match_info:
            info["reason"] = "no matched experience"
            return info

        stage = str(self.match_info.get("stage", ""))
        score = float(self.match_info.get("score", 0.0) or 0.0)
        info["stage"] = stage
        info["score"] = score

        min_score = REPLAY_USE_MIN_SCORE_BY_STAGE.get(stage, 0.85)
        if score < min_score:
            info["reason"] = f"match score too low ({score:.3f} < {min_score:.3f})"
            return info

        # Replay should end naturally if done exists; otherwise keep only short traces.
        done_idx = -1
        for i, action in enumerate(self.actions):
            if str(action.get("action", "")).lower() == "done":
                done_idx = i
                break
        if done_idx >= 0:
            usable_actions = self.actions[:done_idx + 1]
            info["has_done"] = True
        else:
            usable_actions = list(self.actions)
            if len(usable_actions) > REPLAY_USE_MAX_ACTIONS_WITHOUT_DONE:
                info["reason"] = (
                    f"no done in trace and too long ({len(usable_actions)} > {REPLAY_USE_MAX_ACTIONS_WITHOUT_DONE})"
                )
                return info

        signatures = [self._action_signature(a) for a in usable_actions]
        if signatures:
            top_count = Counter(signatures).most_common(1)[0][1]
            repeat_ratio = top_count / max(len(signatures), 1)
            info["repeat_ratio"] = round(float(repeat_ratio), 3)
            if repeat_ratio > REPLAY_USE_MAX_REPEAT_RATIO:
                info["reason"] = f"trace too repetitive ({repeat_ratio:.3f} > {REPLAY_USE_MAX_REPEAT_RATIO:.3f})"
                return info

        click_actions = [
            a for a in usable_actions
            if str(a.get("action", "")).lower() in {"click", "click_input"}
        ]
        if click_actions:
            xml_ready = sum(1 for a in click_actions if self.get_action_xml_path(a))
            xml_ready_ratio = xml_ready / max(len(click_actions), 1)
            info["xml_ready_ratio"] = round(float(xml_ready_ratio), 3)
            if xml_ready_ratio < 1.0:
                info["reason"] = "replay xml metadata incomplete"
                return info

        self.actions = usable_actions
        info["use_replay"] = True
        info["reason"] = "accepted"
        info["replay_action_count"] = len(self.actions)
        return info

    def match_experience(self) -> Optional[Dict[str, Any]]:
        if not self.task_desc:
            logger.info("[MATCH] Empty task description; skip matching")
            return None

        candidates = []
        for path in self._list_experience_files():
            desc = self._load_task_description(path)
            if not desc:
                continue
            candidates.append((path, desc))

        if not candidates:
            return None

        query_norm = normalize_task_text(self.task_desc)
        query_tokens = extract_tokens(self.task_desc)

        # MATCHING_STAGE_1: normalized edit distance / keyword overlap
        best = None
        stage1_probe = None
        for path, desc in candidates:
            cand_norm = normalize_task_text(desc)
            cand_tokens = extract_tokens(desc)
            edit_sim = edit_similarity(query_norm, cand_norm)
            kw_sim = jaccard_similarity(query_tokens, cand_tokens)
            probe_score = max(edit_sim, kw_sim)
            if stage1_probe is None or probe_score > stage1_probe["score"]:
                stage1_probe = {
                    "path": path,
                    "desc": desc,
                    "score": probe_score,
                    "edit_sim": edit_sim,
                    "kw_sim": kw_sim,
                }
            if edit_sim >= self.EDIT_SIM_THRESHOLD or kw_sim >= self.KEYWORD_JACCARD_THRESHOLD:
                score = max(edit_sim, kw_sim)
                if (best is None) or (score > best["score"]):
                    best = {"path": path, "desc": desc, "score": score, "stage": "edit_or_keyword"}
        if stage1_probe:
            logger.info(
                f"[MATCH-S1] best score={stage1_probe['score']:.3f} "
                f"(edit={stage1_probe['edit_sim']:.3f}, kw={stage1_probe['kw_sim']:.3f})"
            )
        if best:
            return best

        # MATCHING_STAGE_2: BM25 (low-cost lexical retrieval)
        doc_tokens_list = [extract_tokens(d) for _, d in candidates]
        scores = bm25_scores(query_tokens, doc_tokens_list)
        if scores:
            best_idx = max(range(len(scores)), key=scores.__getitem__)
            best_score = scores[best_idx]
            logger.info(f"[MATCH-S2] best bm25={best_score:.3f}")
            if best_score >= self.BM25_MIN_SCORE:
                path, desc = candidates[best_idx]
                return {"path": path, "desc": desc, "score": best_score, "stage": "bm25"}

        # MATCHING_STAGE_3: vector similarity (only if stage1/2 miss)
        query_vec = get_text_embedding(self.task_desc)
        if query_vec is None:
            logger.info("[MATCH] Vector backend unavailable; skip stage3")
            return None
        best_score = -1.0
        best_idx = -1
        for i, (_, desc) in enumerate(candidates):
            cand_vec = get_text_embedding(desc)
            if cand_vec is None:
                continue
            sim = cosine_similarity(query_vec, cand_vec)
            if sim > best_score:
                best_score = sim
                best_idx = i
        if best_idx >= 0:
            logger.info(f"[MATCH-S3] best vector={best_score:.3f}")
        if best_idx >= 0 and best_score >= self.VECTOR_SIM_THRESHOLD:
            path, desc = candidates[best_idx]
            return {"path": path, "desc": desc, "score": best_score, "stage": "vector"}

        return None

    def load_experience(self):
        """Load recorded experience via fuzzy matching."""
        matched = self.match_experience()
        if matched:
            self._load_experience_file(matched["path"])
            self.match_info = matched
            logger.info(
                f"[MATCH] Hit stage={matched['stage']} score={matched['score']:.3f} desc={matched['desc']}"
            )
        else:
            self.actions = []

    def save_experience(self):
        """Save recorded actions to experience file."""
        try:
            data = {
                "task_name": self.task_name,
                "task_description": self.task_desc,
                "task_description_norm": normalize_task_text(self.task_desc),
                "task_keywords": extract_tokens(self.task_desc),
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "action_count": len(self.actions),
                "actions": self.actions
            }
            with open(self.experience_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"[RECORD] Saved {len(self.actions)} actions to: {self.experience_file}")
        except Exception as e:
            logger.error(f"[ERROR] Failed to save experience file: {e}")

    def record_action(self, action: Dict[str, Any]):
        """Record a single action."""
        self.actions.append(action)
        logger.info(f"[RECORD] Record action {len(self.actions)}: {action['action']}")

    def get_next_action(self, action_idx: int) -> Optional[Dict[str, Any]]:
        if action_idx < len(self.actions):
            return self.actions[action_idx]
        return None

    def get_action_screenshot_path(self, action: Dict[str, Any]) -> Optional[str]:
        """Resolve screenshot path used for replay guard."""
        base_dir = self.loaded_from_dir
        if not base_dir:
            return None

        candidate = action.get("screenshot_file")
        if isinstance(candidate, str) and candidate:
            if os.path.isabs(candidate):
                return candidate if os.path.exists(candidate) else None
            rel_path = os.path.join(base_dir, candidate)
            if os.path.exists(rel_path):
                return rel_path

        step = action.get("step")
        if isinstance(step, int):
            for name in (f"{step:02d}_screenshot.jpg", f"{step}_screenshot.jpg"):
                path = os.path.join(base_dir, name)
                if os.path.exists(path):
                    return path
        return None

    def get_action_xml_path(self, action: Dict[str, Any]) -> Optional[str]:
        """Resolve XML path used for replay XML guard."""
        base_dir = self.loaded_from_dir
        if not base_dir:
            return None

        candidate = action.get("xml_file")
        if isinstance(candidate, str) and candidate:
            if os.path.isabs(candidate):
                return candidate if os.path.exists(candidate) else None
            rel_path = os.path.join(base_dir, candidate)
            if os.path.exists(rel_path):
                return rel_path

        step = action.get("step")
        if isinstance(step, int):
            for name in (f"{step:02d}.xml", f"{step}.xml"):
                path = os.path.join(base_dir, name)
                if os.path.exists(path):
                    return path
        return None

    def has_experience(self) -> bool:
        return len(self.actions) > 0

    def clone_replay_seed(self, replay_actions: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Copy the matched historical trajectory into current task dir for traceability.
        This does not affect runtime replay behavior.
        """
        info: Dict[str, Any] = {
            "copied": False,
            "seed_dir": None,
            "source_dir": self.loaded_from_dir,
            "file_count": 0,
            "error": "",
        }
        if not self.loaded_from_dir:
            info["error"] = "no matched source dir"
            return info

        try:
            seed_dir = os.path.join(self.data_dir, "_replay_seed")
            os.makedirs(seed_dir, exist_ok=True)

            files_to_copy = set()
            if self.loaded_experience_path and os.path.exists(self.loaded_experience_path):
                files_to_copy.add(self.loaded_experience_path)

            actions_record_path = os.path.join(self.loaded_from_dir, "actions_record.json")
            if os.path.exists(actions_record_path):
                files_to_copy.add(actions_record_path)

            source_actions = replay_actions if replay_actions is not None else self.actions
            for action in source_actions:
                screenshot_path = self.get_action_screenshot_path(action)
                xml_path = self.get_action_xml_path(action)
                if screenshot_path and os.path.exists(screenshot_path):
                    files_to_copy.add(screenshot_path)
                if xml_path and os.path.exists(xml_path):
                    files_to_copy.add(xml_path)

            copied = 0
            for src in sorted(files_to_copy):
                dst = os.path.join(seed_dir, os.path.basename(src))
                if os.path.abspath(src) == os.path.abspath(dst):
                    continue
                shutil.copy2(src, dst)
                copied += 1

            info["copied"] = True
            info["seed_dir"] = seed_dir
            info["file_count"] = copied
            return info
        except Exception as e:
            info["error"] = str(e)
            return info



def try_execute_graph_replay_candidate(
    device: Any,
    candidate: Dict[str, Any],
    step: int,
    screenshot_file: str,
    current_xml_file: Optional[str],
    current_hierarchy_xml: Optional[str],
) -> Dict[str, Any]:
    """Execute one graph replay candidate with deterministic XML guard."""
    payload = candidate.get("action_payload") if isinstance(candidate.get("action_payload"), dict) else {}
    action_name = str(payload.get("action_type", "")).lower()
    base_record: Dict[str, Any] = {
        "step": step,
        "action": action_name or "unknown",
        "screenshot_file": screenshot_file,
        "xml_file": current_xml_file,
        "replayed": True,
        "replay_source": "template_graph",
        "graph_replay": {
            "edge_id": candidate.get("edge_id"),
            "from_node_id": candidate.get("node_id"),
            "to_node_id": candidate.get("to_node_id"),
            "action_sig": candidate.get("action_sig", ""),
            "search_source": candidate.get("source", ""),
            "mode": candidate.get("mode", GRAPH_REPLAY_SEARCH_MODE),
            "replay_score": candidate.get("replay_score", 0.0),
            "edge_confidence": candidate.get("edge_confidence", 0.0),
            "edge_probability": candidate.get("edge_probability", 0.0),
        },
    }

    if action_name == "done":
        return {
            "ok": True,
            "used_done": True,
            "action_record": base_record,
            "history_payload": {
                "reasoning": "graph replay accepted",
                "action": "done",
                "parameters": {},
            },
        }

    if action_name in {"click", "click_input"}:
        ui_meta = payload.get("ui_meta") if isinstance(payload.get("ui_meta"), dict) else None
        position = payload.get("position")
        pending_text = str(payload.get("text", "")) if action_name == "click_input" else ""
        if action_name == "click_input" and not pending_text:
            return {"ok": False, "reject_reason": "missing_click_input_text"}
        click_x: Optional[int] = None
        click_y: Optional[int] = None
        current_ui_meta = None
        guard_info: Dict[str, Any] = {"used_ui_meta": False, "used_position_fallback": False}

        if ui_meta:
            matched = find_android_point_by_ui_meta(current_hierarchy_xml, ui_meta)
            if matched and isinstance(matched.get("position"), list) and len(matched["position"]) == 2:
                click_x = int(matched["position"][0])
                click_y = int(matched["position"][1])
                current_ui_meta = matched.get("ui_meta")
                guard_info["used_ui_meta"] = True
                guard_info["matched_keys"] = int(matched.get("matched_keys", 0) or 0)
            else:
                return {"ok": False, "reject_reason": "ui_meta_not_matched"}

        if click_x is None or click_y is None:
            if (
                isinstance(position, list)
                and len(position) == 2
                and all(isinstance(v, (int, float)) for v in position)
            ):
                click_x = int(round(float(position[0])))
                click_y = int(round(float(position[1])))
                guard_info["used_position_fallback"] = True
                if current_hierarchy_xml:
                    current_ui_meta = extract_android_ui_meta_by_point(current_hierarchy_xml, click_x, click_y)
            else:
                return {"ok": False, "reject_reason": "invalid_click_position"}

        device.click(click_x, click_y)
        target = str(payload.get("target", ""))
        base_record.update(
            {
                "action": action_name,
                "target": target,
                "position": [click_x, click_y],
                "replay_guard": guard_info,
            }
        )
        if current_ui_meta is not None:
            base_record["ui_meta"] = current_ui_meta

        if action_name == "click_input":
            input_ok = device.input_text(pending_text)
            base_record["text"] = pending_text
            base_record["input_ok"] = bool(input_ok)
            history_payload = {
                "reasoning": "graph replay accepted",
                "action": "click_input",
                "parameters": {
                    "target_element": target,
                    "text": pending_text,
                },
            }
        else:
            history_payload = {
                "reasoning": "graph replay accepted",
                "action": "click",
                "parameters": {"target_element": target},
            }
        return {
            "ok": True,
            "used_done": False,
            "action_record": base_record,
            "history_payload": history_payload,
        }

    if action_name == "input":
        text = str(payload.get("text", ""))
        if not text:
            return {"ok": False, "reject_reason": "missing_input_text"}
        device.input_text(text)
        base_record["text"] = text
        return {
            "ok": True,
            "used_done": False,
            "action_record": base_record,
            "history_payload": {
                "reasoning": "graph replay accepted",
                "action": "input",
                "parameters": {"text": text},
            },
        }

    if action_name == "swipe":
        direction = str(payload.get("direction", "down")).lower() or "down"
        device.swipe(direction)
        base_record["direction"] = direction
        return {
            "ok": True,
            "used_done": False,
            "action_record": base_record,
            "history_payload": {
                "reasoning": "graph replay accepted",
                "action": "swipe",
                "parameters": {"direction": direction},
            },
        }

    return {"ok": False, "reject_reason": f"unsupported_action:{action_name}"}


def execute_task_with_rr(app_name, task_desc, device, data_dir,
                        decider_model="", grounder_model="",
                        use_qwen3=True, no_compress=False,
                        template_graph="on", template_graph_dir: Optional[str] = None):
    """Execute task with graph-first replay (step-wise search) and record current trace."""
    os.makedirs(data_dir, exist_ok=True)

    history: List[str] = []
    actions: List[Dict[str, Any]] = []
    step = 0

    graph_enabled = str(template_graph).strip().lower() != "off"
    graph_runtime_hints: List[Dict[str, Any]] = []
    graph_error = ""
    graph_update_summary: Dict[str, Any] = {}
    resolved_graph_dir = template_graph_dir or os.path.join(os.path.dirname(data_dir), "_template_graph")
    graph_builder, graph_file, graph_init_error = _init_template_graph_builder(
        graph_enabled=graph_enabled,
        graph_dir=resolved_graph_dir,
        app_name=app_name,
    )
    if graph_enabled and graph_builder is None:
        graph_error = graph_init_error or "unknown init error"
        logger.warning(f"[GRAPH] Disabled due to initialization failure: {graph_error}")
    elif graph_builder is not None:
        logger.info(f"[GRAPH] Enabled incremental template graph: {graph_file}")

    task_hash = stable_task_hash(task_desc)
    experience_rr = ExperienceRR(
        data_dir,
        f"task_{task_hash}",
        task_desc=task_desc,
        search_root=os.path.dirname(data_dir),
        enable_load=False,
    )
    graph_replay_stats: Dict[str, Any] = {
        "enabled": bool(graph_builder is not None),
        "mode": GRAPH_REPLAY_SEARCH_MODE,
        "max_candidates_per_step": GRAPH_REPLAY_MAX_CANDIDATES_PER_STEP,
        "steps_attempted": 0,
        "candidate_count": 0,
        "candidates_tried": 0,
        "steps_replayed": 0,
        "actions_replayed": 0,
        "step_fallback_to_decider": 0,
        "rejected_reasons": {},
        "executed_edges": {},
        "executed_from_nodes": {},
        "executed_sources": {},
        "done_by_graph_replay": False,
    }
    stop_reason: Optional[str] = None
    prev_xml_signature = ""
    no_change_xml_streak = 0

    logger.info(f"[START] Execute task: {task_desc}")
    logger.info(
        f"[MODE] Graph replay enabled={graph_replay_stats['enabled']} "
        f"mode={graph_replay_stats['mode']}"
    )
    # target_package = device.app_package_names.get(app_name, app_name)

    while step < MAX_STEPS:
        step += 1
        logger.info(f"\n[STEP] Step {step}")

        logger.info("[THINK] Capture screenshot...")
        screenshot_file = f"{step:02d}_screenshot.jpg"
        screenshot_save_path = os.path.join(data_dir, screenshot_file)
        current_screenshot_path: Optional[str] = None

        try:
            raw_screenshot_path = device.screenshot()
            if raw_screenshot_path and os.path.exists(raw_screenshot_path):
                if os.path.abspath(raw_screenshot_path) != os.path.abspath(screenshot_save_path):
                    shutil.copy(raw_screenshot_path, screenshot_save_path)
                current_screenshot_path = screenshot_save_path
                logger.info(f"[OK] Screenshot saved: {current_screenshot_path}")
        except Exception as e:
            logger.warning(f"[WARNING] Failed to save screenshot: {e}")

        if not current_screenshot_path or not os.path.exists(current_screenshot_path):
            logger.error("[ERROR] Screenshot unavailable, abort task")
            break

        current_hierarchy_xml, current_xml_file = dump_hierarchy_for_step(device, data_dir, step)
        current_xml_signature = build_xml_page_signature(current_hierarchy_xml)
        if current_xml_signature and prev_xml_signature and current_xml_signature == prev_xml_signature:
            no_change_xml_streak += 1
        else:
            no_change_xml_streak = 0
        prev_xml_signature = current_xml_signature

        if no_change_xml_streak >= NO_PROGRESS_XML_STREAK:
            stop_reason = "no_progress_xml"
            logger.warning(
                f"[GUARD] Stop by XML no-change guard at step {step} (streak={no_change_xml_streak})"
            )
            action_record = {
                "step": step,
                "action": "done",
                "screenshot_file": screenshot_file,
                "xml_file": current_xml_file,
                "forced_stop_reason": stop_reason,
            }
            actions.append(action_record)
            experience_rr.record_action(action_record)
            break

        if graph_builder is not None and current_hierarchy_xml:
            try:
                current_candidates = graph_builder.infer_current_node(current_hierarchy_xml, top_k=3)
                best_node_id = current_candidates[0]["node_id"] if current_candidates else None
                next_candidates = graph_builder.predict_next(best_node_id, top_k=3) if best_node_id else []
                interruption_info = graph_builder.detect_interruption(current_hierarchy_xml)
                frontier = graph_builder.suggest_low_confidence_frontier(limit=5)
                step_hint = {
                    "step": step,
                    "xml_file": current_xml_file,
                    "current_candidates": current_candidates,
                    "next_candidates": next_candidates,
                    "interruption": interruption_info,
                    "low_confidence_frontier": frontier,
                }
                graph_runtime_hints.append(step_hint)
                logger.info(
                    f"[GRAPH] Step {step} hints: candidates={len(current_candidates)} "
                    f"next={len(next_candidates)} interruption={interruption_info.get('is_interruption')}"
                )
            except Exception as e:
                logger.warning(f"[GRAPH] Runtime hint failed at step {step}: {e}")

        graph_replay_hit = False
        if graph_builder is not None and current_hierarchy_xml:
            graph_replay_stats["steps_attempted"] += 1
            try:
                replay_candidates = graph_builder.search_replay_actions(
                    current_hierarchy_xml=current_hierarchy_xml,
                    mode=GRAPH_REPLAY_SEARCH_MODE,
                )
            except Exception as e:
                replay_candidates = []
                logger.warning(f"[GRAPH-REPLAY] candidate search failed at step {step}: {e}")

            if replay_candidates:
                candidate_limit = min(
                    len(replay_candidates),
                    int(GRAPH_REPLAY_MAX_CANDIDATES_PER_STEP),
                )
                graph_replay_stats["candidate_count"] += candidate_limit
                for candidate in replay_candidates[:candidate_limit]:
                    graph_replay_stats["candidates_tried"] += 1
                    exec_res = try_execute_graph_replay_candidate(
                        device=device,
                        candidate=candidate,
                        step=step,
                        screenshot_file=screenshot_file,
                        current_xml_file=current_xml_file,
                        current_hierarchy_xml=current_hierarchy_xml,
                    )
                    if not exec_res.get("ok"):
                        reason = str(exec_res.get("reject_reason", "unknown_reject"))
                        graph_replay_stats["rejected_reasons"][reason] = (
                            int(graph_replay_stats["rejected_reasons"].get(reason, 0) or 0) + 1
                        )
                        continue

                    action_record = exec_res["action_record"]
                    actions.append(action_record)
                    experience_rr.record_action(action_record)
                    graph_replay_stats["steps_replayed"] += 1
                    graph_replay_stats["actions_replayed"] += 1
                    edge_id = str(candidate.get("edge_id", ""))
                    from_node = str(candidate.get("node_id", ""))
                    src = str(candidate.get("source", ""))
                    if edge_id:
                        graph_replay_stats["executed_edges"][edge_id] = (
                            int(graph_replay_stats["executed_edges"].get(edge_id, 0) or 0) + 1
                        )
                    if from_node:
                        graph_replay_stats["executed_from_nodes"][from_node] = (
                            int(graph_replay_stats["executed_from_nodes"].get(from_node, 0) or 0) + 1
                        )
                    if src:
                        graph_replay_stats["executed_sources"][src] = (
                            int(graph_replay_stats["executed_sources"].get(src, 0) or 0) + 1
                        )
                    history_payload = exec_res.get("history_payload")
                    if history_payload:
                        history.append(json.dumps(history_payload, ensure_ascii=False))

                    if exec_res.get("used_done"):
                        stop_reason = "graph_replay_done"
                        graph_replay_stats["done_by_graph_replay"] = True
                        logger.info(f"[OK] Task done by graph replay at step {step}")
                        graph_replay_hit = True
                        break

                    graph_replay_hit = True
                    logger.info(
                        f"[GRAPH-REPLAY] accepted step={step} action={action_record.get('action')} "
                        f"edge={candidate.get('edge_id')} from={candidate.get('node_id')}"
                    )
                    time.sleep(1)
                    break

        if graph_replay_hit:
            if stop_reason == "graph_replay_done":
                break
            continue
        if graph_builder is not None:
            graph_replay_stats["step_fallback_to_decider"] += 1
            logger.info(f"[GRAPH-REPLAY] no accepted candidate at step {step}, fallback to decider")

        decider_screenshot_file = screenshot_file
        retry_in_step = 0
        decider_step_completed = False
        abort_current_task = False

        while retry_in_step <= TOCTOU_MAX_RETRY_PER_STEP:
            screenshot_b64, orig_width, orig_height = get_screenshot_b64(
                device=None,
                compress=(not no_compress),
                screenshot_path=current_screenshot_path,
            )
            if not screenshot_b64:
                logger.error("[ERROR] Screenshot encode failed, abort task")
                abort_current_task = True
                break

            if no_compress:
                scale_factor = 1.0
            else:
                scale_factor = 1.0 / SCREENSHOT_FACTOR

            logger.info("[THINK] Calling decider model...")
            decider_response = None
            decider_response_str = None
            for parse_try in range(1, DECIDER_PARSE_MAX_RETRIES + 1):
                decider_response_str = get_decider_action(task_desc, history, screenshot_b64, decider_model)
                if not decider_response_str:
                    logger.warning(
                        f"[WARNING] Decider returned empty response (attempt {parse_try}/{DECIDER_PARSE_MAX_RETRIES})"
                    )
                    continue

                logger.info(
                    f"[RESPONSE] Decider (attempt {parse_try}/{DECIDER_PARSE_MAX_RETRIES}): {decider_response_str[:300]}"
                )
                decider_response = parse_json_response(decider_response_str)
                if decider_response:
                    break
                logger.warning(
                    f"[WARNING] Failed to parse decider response (attempt {parse_try}/{DECIDER_PARSE_MAX_RETRIES})"
                )

            if not decider_response:
                logger.error(f"[ERROR] Failed to parse decider response after {DECIDER_PARSE_MAX_RETRIES} attempts")
                abort_current_task = True
                break

            action_name = decider_response.get("action", "done")
            reasoning = decider_response.get("reasoning", "")
            parameters = decider_response.get("parameters", {})

            logger.info(f"[REASON] {reasoning}")
            logger.info(f"[ACTION] {action_name}")

            toctou_guard: Optional[Dict[str, Any]] = None
            need_toctou_retry = False

            if action_name == "done":
                action_record = {
                    "step": step,
                    "action": "done",
                    "screenshot_file": decider_screenshot_file,
                    "xml_file": current_xml_file,
                }
                actions.append(action_record)
                experience_rr.record_action(action_record)
                stop_reason = "done"
                logger.info("[OK] Task completed")
                decider_step_completed = True
                break

            if action_name == "click":
                target = parameters.get("target_element", "")
                logger.info(f"[TARGET] {target}")
                grounder_response_str = get_grounder_coords(
                    target,
                    screenshot_b64,
                    grounder_model,
                    use_qwen3=use_qwen3,
                )
                logger.info(
                    f"[DEBUG] Grounder raw: {grounder_response_str[:200] if grounder_response_str else 'None'}"
                )

                if grounder_response_str:
                    grounder_response = parse_json_response(grounder_response_str)
                    if grounder_response and "bbox" in grounder_response:
                        bbox = grounder_response["bbox"]
                        if use_qwen3:
                            bbox_abs = convert_qwen3_coordinates_to_absolute(
                                bbox,
                                orig_width,
                                orig_height,
                                is_bbox=True,
                            )
                            x1, y1, x2, y2 = [int(v) for v in bbox_abs]
                        else:
                            x1 = int(float(bbox[0]) * scale_factor)
                            y1 = int(float(bbox[1]) * scale_factor)
                            x2 = int(float(bbox[2]) * scale_factor)
                            y2 = int(float(bbox[3]) * scale_factor)

                        x = int((x1 + x2) / 2)
                        y = int((y1 + y2) / 2)

                        toctou_guard = evaluate_toctou_pre_action(
                            device=device,
                            data_dir=data_dir,
                            step=step,
                            attempt=retry_in_step + 1,
                            reference_screenshot_path=current_screenshot_path,
                        )
                        guard_score = float(toctou_guard.get("score", 0.0) or 0.0)
                        logger.info(
                            f"[TOCTOU] step={step} attempt={toctou_guard.get('attempt')} "
                            f"score={guard_score:.4f} threshold={TOCTOU_IMAGE_SIM_MIN:.4f} "
                            f"allow={bool(toctou_guard.get('allow'))} reason={toctou_guard.get('reason')}"
                        )
                        if not toctou_guard.get("allow"):
                            need_toctou_retry = True
                        else:
                            device.click(x, y)

                            try:
                                visualize_output = os.path.join(data_dir, f"{step:02d}_click_visualization.jpg")
                                visualize_click(current_screenshot_path, x, y, target, visualize_output)
                                bounds_path = os.path.join(data_dir, f"{step}_bounds.jpg")
                                visualize_model_bbox(
                                    current_screenshot_path,
                                    [x1, y1, x2, y2],
                                    bounds_path,
                                    label=target,
                                )
                            except Exception as e:
                                logger.warning(f"[WARNING] Click visualization failed: {e}")

                            current_ui_meta = None
                            if current_hierarchy_xml:
                                current_ui_meta = extract_android_ui_meta_by_bbox(
                                    current_hierarchy_xml,
                                    [x1, y1, x2, y2],
                                )

                            action_record = {
                                "step": step,
                                "action": "click",
                                "position": [x, y],
                                "target": target,
                                "bbox": [x1, y1, x2, y2],
                                "bbox_compressed": bbox,
                                "scale_factor": scale_factor,
                                "screenshot_file": decider_screenshot_file,
                                "xml_file": current_xml_file,
                                "toctou_check": {
                                    "score": guard_score,
                                    "threshold": float(toctou_guard.get("threshold", TOCTOU_IMAGE_SIM_MIN) or TOCTOU_IMAGE_SIM_MIN),
                                    "attempt": int(toctou_guard.get("attempt", retry_in_step + 1) or (retry_in_step + 1)),
                                    "check_image_file": toctou_guard.get("check_image_file", ""),
                                },
                            }
                            if use_qwen3:
                                action_record["bbox_norm_0_1000"] = bbox
                            if current_ui_meta is not None:
                                action_record["ui_meta"] = current_ui_meta
                            actions.append(action_record)
                            experience_rr.record_action(action_record)
                    else:
                        logger.error(f"[ERROR] Invalid grounder response: {grounder_response}")
                else:
                    logger.error("[ERROR] Grounder returned empty response")

            elif action_name == "click_input":
                target = parameters.get("target_element", "")
                text = str(parameters.get("text", ""))
                logger.info(f"[TARGET] {target}")
                logger.info(f"[INPUT-TEXT] {text}")

                if not target:
                    logger.error("[ERROR] click_input 缺少 target_element")
                else:
                    grounder_response_str = get_grounder_coords(
                        target,
                        screenshot_b64,
                        grounder_model,
                        use_qwen3=use_qwen3,
                    )
                    logger.info(
                        f"[DEBUG] Grounder raw: {grounder_response_str[:200] if grounder_response_str else 'None'}"
                    )

                    if grounder_response_str:
                        grounder_response = parse_json_response(grounder_response_str)
                        if grounder_response and "bbox" in grounder_response:
                            bbox = grounder_response["bbox"]
                            if use_qwen3:
                                bbox_abs = convert_qwen3_coordinates_to_absolute(
                                    bbox,
                                    orig_width,
                                    orig_height,
                                    is_bbox=True,
                                )
                                x1, y1, x2, y2 = [int(v) for v in bbox_abs]
                            else:
                                x1 = int(float(bbox[0]) * scale_factor)
                                y1 = int(float(bbox[1]) * scale_factor)
                                x2 = int(float(bbox[2]) * scale_factor)
                                y2 = int(float(bbox[3]) * scale_factor)

                            x = int((x1 + x2) / 2)
                            y = int((y1 + y2) / 2)

                            toctou_guard = evaluate_toctou_pre_action(
                                device=device,
                                data_dir=data_dir,
                                step=step,
                                attempt=retry_in_step + 1,
                                reference_screenshot_path=current_screenshot_path,
                            )
                            guard_score = float(toctou_guard.get("score", 0.0) or 0.0)
                            logger.info(
                                f"[TOCTOU] step={step} attempt={toctou_guard.get('attempt')} "
                                f"score={guard_score:.4f} threshold={TOCTOU_IMAGE_SIM_MIN:.4f} "
                                f"allow={bool(toctou_guard.get('allow'))} reason={toctou_guard.get('reason')}"
                            )
                            if not toctou_guard.get("allow"):
                                need_toctou_retry = True
                            else:
                                device.click(x, y)
                                input_ok = device.input_text(text)
                                if not input_ok:
                                    logger.warning("[WARNING] click_input: 点击后输入失败")

                                try:
                                    visualize_output = os.path.join(data_dir, f"{step:02d}_click_visualization.jpg")
                                    visualize_click(current_screenshot_path, x, y, target, visualize_output)
                                    bounds_path = os.path.join(data_dir, f"{step}_bounds.jpg")
                                    visualize_model_bbox(
                                        current_screenshot_path,
                                        [x1, y1, x2, y2],
                                        bounds_path,
                                        label=target,
                                    )
                                except Exception as e:
                                    logger.warning(f"[WARNING] Click visualization failed: {e}")

                                current_ui_meta = None
                                if current_hierarchy_xml:
                                    current_ui_meta = extract_android_ui_meta_by_bbox(
                                        current_hierarchy_xml,
                                        [x1, y1, x2, y2],
                                    )

                                action_record = {
                                    "step": step,
                                    "action": "click_input",
                                    "position": [x, y],
                                    "target": target,
                                    "text": text,
                                    "input_ok": bool(input_ok),
                                    "bbox": [x1, y1, x2, y2],
                                    "bbox_compressed": bbox,
                                    "scale_factor": scale_factor,
                                    "screenshot_file": decider_screenshot_file,
                                    "xml_file": current_xml_file,
                                    "toctou_check": {
                                        "score": guard_score,
                                        "threshold": float(toctou_guard.get("threshold", TOCTOU_IMAGE_SIM_MIN) or TOCTOU_IMAGE_SIM_MIN),
                                        "attempt": int(toctou_guard.get("attempt", retry_in_step + 1) or (retry_in_step + 1)),
                                        "check_image_file": toctou_guard.get("check_image_file", ""),
                                    },
                                }
                                if use_qwen3:
                                    action_record["bbox_norm_0_1000"] = bbox
                                if current_ui_meta is not None:
                                    action_record["ui_meta"] = current_ui_meta
                                actions.append(action_record)
                                experience_rr.record_action(action_record)
                        else:
                            logger.error(f"[ERROR] Invalid grounder response: {grounder_response}")
                    else:
                        logger.error("[ERROR] Grounder returned empty response")

            elif action_name == "input":
                text = parameters.get("text", "")
                toctou_guard = evaluate_toctou_pre_action(
                    device=device,
                    data_dir=data_dir,
                    step=step,
                    attempt=retry_in_step + 1,
                    reference_screenshot_path=current_screenshot_path,
                )
                guard_score = float(toctou_guard.get("score", 0.0) or 0.0)
                logger.info(
                    f"[TOCTOU] step={step} attempt={toctou_guard.get('attempt')} "
                    f"score={guard_score:.4f} threshold={TOCTOU_IMAGE_SIM_MIN:.4f} "
                    f"allow={bool(toctou_guard.get('allow'))} reason={toctou_guard.get('reason')}"
                )
                if not toctou_guard.get("allow"):
                    need_toctou_retry = True
                else:
                    device.input_text(text)
                    action_record = {
                        "step": step,
                        "action": "input",
                        "text": text,
                        "screenshot_file": decider_screenshot_file,
                        "xml_file": current_xml_file,
                        "toctou_check": {
                            "score": guard_score,
                            "threshold": float(toctou_guard.get("threshold", TOCTOU_IMAGE_SIM_MIN) or TOCTOU_IMAGE_SIM_MIN),
                            "attempt": int(toctou_guard.get("attempt", retry_in_step + 1) or (retry_in_step + 1)),
                            "check_image_file": toctou_guard.get("check_image_file", ""),
                        },
                    }
                    actions.append(action_record)
                    experience_rr.record_action(action_record)

            elif action_name == "swipe":
                direction = str(parameters.get("direction", "down")).lower()
                toctou_guard = evaluate_toctou_pre_action(
                    device=device,
                    data_dir=data_dir,
                    step=step,
                    attempt=retry_in_step + 1,
                    reference_screenshot_path=current_screenshot_path,
                )
                guard_score = float(toctou_guard.get("score", 0.0) or 0.0)
                logger.info(
                    f"[TOCTOU] step={step} attempt={toctou_guard.get('attempt')} "
                    f"score={guard_score:.4f} threshold={TOCTOU_IMAGE_SIM_MIN:.4f} "
                    f"allow={bool(toctou_guard.get('allow'))} reason={toctou_guard.get('reason')}"
                )
                if not toctou_guard.get("allow"):
                    need_toctou_retry = True
                else:
                    device.swipe(direction)
                    action_record = {
                        "step": step,
                        "action": "swipe",
                        "direction": direction,
                        "screenshot_file": decider_screenshot_file,
                        "xml_file": current_xml_file,
                        "toctou_check": {
                            "score": guard_score,
                            "threshold": float(toctou_guard.get("threshold", TOCTOU_IMAGE_SIM_MIN) or TOCTOU_IMAGE_SIM_MIN),
                            "attempt": int(toctou_guard.get("attempt", retry_in_step + 1) or (retry_in_step + 1)),
                            "check_image_file": toctou_guard.get("check_image_file", ""),
                        },
                    }
                    actions.append(action_record)
                    experience_rr.record_action(action_record)

            if need_toctou_retry:
                if retry_in_step >= TOCTOU_MAX_RETRY_PER_STEP:
                    logger.warning(
                        f"[TOCTOU] Step {step} exceeded retry budget ({TOCTOU_MAX_RETRY_PER_STEP}); skip action"
                    )
                    decider_step_completed = True
                    break

                retry_in_step += 1
                check_path = None
                if isinstance(toctou_guard, dict):
                    candidate_path = toctou_guard.get("check_image_path")
                    if isinstance(candidate_path, str) and os.path.exists(candidate_path):
                        check_path = candidate_path
                if check_path:
                    current_screenshot_path = check_path
                    decider_screenshot_file = os.path.basename(check_path)
                current_hierarchy_xml, current_xml_file = dump_hierarchy_for_step(device, data_dir, step)
                logger.info(
                    f"[TOCTOU] Retry decider in step {step}: "
                    f"{retry_in_step}/{TOCTOU_MAX_RETRY_PER_STEP}"
                )
                continue

            history.append(json.dumps(decider_response, ensure_ascii=False))
            time.sleep(1)
            decider_step_completed = True
            break

        if abort_current_task:
            break
        if stop_reason == "done":
            break
        if not decider_step_completed:
            logger.warning(f"[TOCTOU] Step {step} ended without completion marker; continue next step")
            continue

    if stop_reason is None and step >= MAX_STEPS:
        stop_reason = "max_steps_reached"

    if graph_builder is not None:
        try:
            graph_update_summary = graph_builder.update_from_run(
                data_dir=data_dir,
                actions=actions,
                task_desc=task_desc,
                task_hash=task_hash,
                stop_reason=stop_reason,
            )
            graph_file = graph_update_summary.get("graph_file", graph_file)
            logger.info(
                f"[GRAPH] Updated graph: {graph_file} "
                f"(nodes+={graph_update_summary.get('nodes_created', 0)}, "
                f"edges+={graph_update_summary.get('edges_created', 0)})"
            )
        except Exception as e:
            graph_error = f"update failed: {e}"
            logger.warning(f"[GRAPH] Failed to update graph: {graph_error}")

    template_graph_payload: Dict[str, Any] = {
        "enabled": bool(graph_builder is not None),
        "graph_file": graph_file,
        "update_summary": graph_update_summary,
        "runtime_hints": graph_runtime_hints,
    }
    if graph_error:
        template_graph_payload["error"] = graph_error

    result = {
        "app_name": app_name,
        "task_description": task_desc,
        "mode": "auto",
        "graph_replay": graph_replay_stats,
        "stop_reason": stop_reason,
        "action_count": len(actions),
        "actions": actions,
        "template_graph": template_graph_payload,
    }

    result_path = os.path.join(data_dir, "actions_record.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info(f"[OK] Saved result: {result_path}")
    experience_rr.save_experience()
    return result

def extract_app_from_task(task_desc, app_package_names):
    """从任务描述中提取应用名称"""
    for app_name in app_package_names.keys():
        if app_name in task_desc:
            return app_name
    return None


def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(description="MobiAgent Naïve Record & Replay版本")
    
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
    parser.add_argument("--task", type=str, default=None,
                       help="要执行的任务描述（单条任务）")
    parser.add_argument("--task_file", type=str, default=None,
                       help="任务列表JSON文件路径（内容为字符串数组）")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="数据保存目录（必需）")
    parser.add_argument("--use_qwen3", action="store_true", default=True,
                       help="是否使用Qwen3坐标体系（默认开启）")
    parser.add_argument("--no_compress", action="store_true", default=False,
                       help="调试选项：不压缩截图（发送原始分辨率给Grounder）")
    parser.add_argument("--decider_model", type=str, default="",
                       help="Decider模型名称")
    parser.add_argument("--grounder_model", type=str, default="",
                       help="Grounder模型名称")
    
    parser.add_argument("--template_graph", choices=["on", "off"], default="on",
                       help="Enable incremental template graph builder (default: on)")
    parser.add_argument("--template_graph_dir", type=str, default=None,
                       help="Template graph storage directory (default: <data_dir>/_template_graph)")

    args = parser.parse_args()
    resolved_template_graph_dir = args.template_graph_dir or os.path.join(args.data_dir, "_template_graph")
    
    # 应用包名映射
    app_package_names = {
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
            "浏览器": "com.microsoft.emmx",
            "知乎": "com.zhihu.android",
            "设置": "com.android.settings",
    }
    
    # 读取任务列表（支持单条--task或--task_file）
    task_list: List[str] = []
    if args.task_file:
        try:
            with open(args.task_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                task_list.extend([str(t) for t in loaded])
            else:
                logger.error("[ERROR] task_file内容必须是字符串数组(JSON list)")
                return
        except Exception as e:
            logger.error(f"[ERROR] 读取task_file失败: {e}")
            return
    if args.task:
        task_list.append(args.task)
    if not task_list:
        logger.error("[ERROR] 必须提供--task或--task_file")
        return
    
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
        
        for task_desc in task_list:
            # 如果未指定应用，则从任务描述中提取
            app_name = args.app
            if not app_name:
                detected_app = extract_app_from_task(task_desc, app_package_names)
                if detected_app:
                    app_name = detected_app
                    logger.info(f"[DETECT] 从任务描述中提取应用: {app_name}")
                else:
                    logger.error("[ERROR] 无法从任务中提取应用名称，请使用--app指定")
                    continue

            # 为每个任务创建独立目录，避免覆盖截图/日志
            task_hash = stable_task_hash(task_desc)
            task_dir = os.path.join(args.data_dir, f"task_{task_hash}")

            # 启动应用
            logger.info(f"[APP] 正在启动应用: {app_name}")
            device.start_app(app_name)
            time.sleep(2)
            
            # 执行任务
            result = execute_task_with_rr(
                app_name,
                task_desc,
                device,
                task_dir,
                args.decider_model,
                args.grounder_model,
                use_qwen3=args.use_qwen3,
                no_compress=args.no_compress,
                template_graph=args.template_graph,
                template_graph_dir=resolved_template_graph_dir,
            )
            
            # 停止应用
            logger.info(f"[STOP] 正在停止应用: {app_name}")
            device.stop_app(app_name)
        
        logger.info("[DONE] 执行完成")
    
    except KeyboardInterrupt:
        logger.info("[WARNING] 用户中断")
    except Exception as e:
        logger.error(f"[ERROR] 发生错误: {e}", exc_info=True)


if __name__ == "__main__":
    main()
