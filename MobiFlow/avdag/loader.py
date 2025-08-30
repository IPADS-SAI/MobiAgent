from __future__ import annotations
import json
from typing import Any, Dict

import yaml

from .types import ConditionSpec, NodeSpec, SuccessSpec, TaskSpec


def _parse_node(d: Dict[str, Any]) -> NodeSpec:
    cond = d.get("condition")
    condition = None
    if cond:
        condition = ConditionSpec(type=cond.get("type"), params=cond.get("params", {}))
    return NodeSpec(
        id=d["id"],
        name=d.get("name"),
        deps=d.get("deps"),
        next=d.get("next"),
        condition=condition,
        score=d.get("score", 10),  # 默认分数为10分
    )


def load_task(path: str) -> TaskSpec:
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    else:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    nodes = [_parse_node(n) for n in raw.get("nodes", [])]
    succ_raw = raw.get("success") or {}
    success = SuccessSpec(any_of=succ_raw.get("any_of"), all_of=succ_raw.get("all_of")) if succ_raw else None
    return TaskSpec(task_id=raw.get("task_id", "task"), nodes=nodes, success=success)

__all__ = ["load_task"]
