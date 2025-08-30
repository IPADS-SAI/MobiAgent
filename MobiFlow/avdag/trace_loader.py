from __future__ import annotations
import json
import os
from typing import Any, Dict, List
import re

from .types import Frame


def _read_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def load_frames_from_dir(folder: str) -> List[Frame]:
    """将包含 images/xml/actions/react 的目录转换为帧序列。

    规则：
    - 帧索引从 1 开始（按文件名 N.xml/N.jpg 推断），依次排序。
    - 每帧字段：
      - image: `<folder>/<i>.jpg`（存在时）
      - xml_text: `<folder>/<i>.xml` 的原始文本（存在时）
      - reasoning: react.json[i-1].reasoning（存在时）
      - action: actions.json.actions[i-1]（存在时）
      - text: 合并 reasoning + action 文本，用于简易匹配
    """
    frames: List[Frame] = []
    # 添加一个空的帧，作为起始点
    frames.append({
        "image": None,
        "xml_text": "",
        "reasoning": None,
        "react_action": None,
        "action": None,
        "text": "",
        "ui": {},
        "task_description": "",
        "app_name": ""
    })
    if not os.path.isdir(folder):
        raise FileNotFoundError(folder)

    actions_path = os.path.join(folder, "actions.json")
    react_path = os.path.join(folder, "react.json")
    actions = []
    reacts = []
    act_meta: Dict[str, Any] = {}
    if os.path.exists(actions_path):
        with open(actions_path, "r", encoding="utf-8") as f:
            act_meta = json.load(f) or {}
            actions = (act_meta or {}).get("actions", [])
    if os.path.exists(react_path):
        with open(react_path, "r", encoding="utf-8") as f:
            reacts = json.load(f) or []

    # 找到所有形如 N.xml 或 N.jpg 的索引
    indices: List[int] = []
    for name in os.listdir(folder):
        if name.endswith(".xml"):
            try:
                idx = int(os.path.splitext(name)[0])
                indices.append(idx)
            except ValueError:
                pass
        elif name.endswith(".jpg"):
            try:
                idx = int(os.path.splitext(name)[0])
                indices.append(idx)
            except ValueError:
                pass
    indices = sorted(sorted(set(indices)))

    for i in indices:
        fr: Frame = {}
        xml_path = os.path.join(folder, f"{i}.xml")
        img_path = os.path.join(folder, f"{i}.jpg")
        fr["image"] = img_path if os.path.exists(img_path) else None
        fr["xml_text"] = _read_file(xml_path) if os.path.exists(xml_path) else ""
        # 从 xml 中提取包名等元信息，可以按需要补充，构建更完善的UI上下文
        # 例如：fr["ui"] = {"package": "com.example.app"} 等
        # 这里仅示例提取 package 名称
        ui: Dict[str, Any] = {}
        if fr["xml_text"]:
            m = re.search(r'package="([^"]+)"', fr["xml_text"])  # 简单提取第一个 package
            if m:
                ui["package"] = m.group(1)
        if ui:
            fr["ui"] = ui

        r = reacts[i - 1] if 0 <= (i - 1) < len(reacts) else None
        a = actions[i - 1] if 0 <= (i - 1) < len(actions) else None
        fr["reasoning"] = r.get("reasoning") if isinstance(r, dict) else None
        fr["react_action"] = r.get("action") if isinstance(r, dict) else None
        fr["action"] = a if isinstance(a, dict) else None
        # 添加顶层任务元信息，便于 LLM 判断相关性
        if act_meta:
            fr["task_description"] = act_meta.get("task_description") or act_meta.get("old_task_description")
            fr["app_name"] = act_meta.get("app_name")

        # 组装便于简单文本匹配的 text 字段
        pieces: List[str] = []
        if fr.get("reasoning"):
            pieces.append(str(fr["reasoning"]))
        if a:
            if a.get("type"):
                pieces.append(str(a["type"]))
            if a.get("text"):
                pieces.append(str(a["text"]))
        if r and isinstance(r, dict):
            params = r.get("parameters") or {}
            for v in params.values():
                try:
                    pieces.append(str(v))
                except Exception:
                    pass
        fr["text"] = " \n".join(pieces)

        frames.append(fr)

    # 增加邻接上下文引用（只读）
    for idx, fr in enumerate(frames):
        fr["_index"] = idx
        fr["_prev"] = frames[idx - 1] if idx > 0 else None
        fr["_next"] = frames[idx + 1] if idx + 1 < len(frames) else None

    return frames


__all__ = ["load_frames_from_dir"]
