#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Map an imprecise predicted box to the best Android UIAutomator XML node."""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

BBox = Tuple[int, int, int, int]

CONTAINER_CLASS_KEYWORDS = (
    "framelayout",
    "linearlayout",
    "relativelayout",
    "viewgroup",
    "recyclerview",
)


@dataclass
class UiNode:
    """Flattened UI node parsed from XML."""

    node_id: int
    parent_id: Optional[int]
    bbox: BBox
    area: int
    depth: int
    child_count: int
    text_norm: str
    resource_id: str
    class_name: str
    package: str
    clickable: bool
    enabled: bool
    scrollable: bool
    selected: bool
    checkable: bool
    checked: bool
    text: str = ""
    content_desc: str = ""
    hint: str = ""


@dataclass
class CandidateScore:
    node_id: int
    bbox: BBox
    package: str
    class_name: str
    resource_id: str
    clickable: bool
    enabled: bool
    depth: int
    child_count: int
    cov: float
    cont: float
    iou: float
    dist: float
    s_geom: float
    s_attr: float
    s_text: float
    total_score: float
    retrieval_reasons: List[str] = field(default_factory=list)


@dataclass
class MatchResult:
    status: str
    message: str
    pred_box: BBox
    action_type: str
    best_node: Optional[UiNode]
    candidates: List[CandidateScore] = field(default_factory=list)
    nodes: List[UiNode] = field(default_factory=list)
    thresholds: Dict[str, float] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)


@dataclass
class ValidationResult:
    ok: bool
    status: str
    message: str
    error_code: str = ""
    best_node: Optional[UiNode] = None
    click_node: Optional[UiNode] = None
    details: Dict[str, Any] = field(default_factory=dict)


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _to_bool(v: str) -> bool:
    return str(v).strip().lower() == "true"


def _parse_bounds(bounds_str: str) -> Optional[BBox]:
    if not bounds_str:
        return None
    m = re.match(r"\[(\-?\d+),(\-?\d+)\]\[(\-?\d+),(\-?\d+)\]", bounds_str.strip())
    if not m:
        return None
    x1, y1, x2, y2 = (int(m.group(i)) for i in range(1, 5))
    return _normalize_bbox((x1, y1, x2, y2))


def _normalize_bbox(bbox: Sequence[int]) -> BBox:
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def bbox_center(bbox: BBox) -> Tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def node_center(node: UiNode) -> Tuple[int, int]:
    cx, cy = bbox_center(node.bbox)
    return int(round(cx)), int(round(cy))


def bbox_area(bbox: BBox) -> int:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def intersection_area(a: BBox, b: BBox) -> int:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0
    return (ix2 - ix1) * (iy2 - iy1)


def bbox_iou(a: BBox, b: BBox) -> float:
    inter = intersection_area(a, b)
    if inter <= 0:
        return 0.0
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def _point_in_bbox(x: float, y: float, bbox: BBox) -> bool:
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]


def _infer_screen_size(nodes: Sequence[UiNode], pred_box: BBox) -> Tuple[int, int]:
    max_x = max([pred_box[2]] + [n.bbox[2] for n in nodes]) if nodes else pred_box[2]
    max_y = max([pred_box[3]] + [n.bbox[3] for n in nodes]) if nodes else pred_box[3]
    return max(1, max_x), max(1, max_y)


def _center_distance_norm(a: BBox, b: BBox, screen_diag: float) -> float:
    ax, ay = bbox_center(a)
    bx, by = bbox_center(b)
    d = math.hypot(ax - bx, ay - by)
    return min(1.0, d / max(screen_diag, 1.0))


def parse_xml_to_nodes(xml_str: str) -> List[UiNode]:
    root = ET.fromstring(xml_str)
    nodes: List[UiNode] = []

    def dfs(elem: ET.Element, parent_idx: Optional[int], depth: int) -> None:
        attrib = elem.attrib or {}
        bbox = _parse_bounds(attrib.get("bounds", ""))
        current_parent = parent_idx
        if bbox is not None:
            text = attrib.get("text", "")
            content_desc = attrib.get("content-desc", "")
            hint = attrib.get("hint", "")
            text_norm = normalize_text(" ".join([v for v in (text, content_desc, hint) if v]))

            node = UiNode(
                node_id=len(nodes),
                parent_id=parent_idx,
                bbox=bbox,
                area=bbox_area(bbox),
                depth=depth,
                child_count=0,
                text_norm=text_norm,
                resource_id=attrib.get("resource-id", ""),
                class_name=attrib.get("class", ""),
                package=attrib.get("package", ""),
                clickable=_to_bool(attrib.get("clickable", "false")),
                enabled=_to_bool(attrib.get("enabled", "true")),
                scrollable=_to_bool(attrib.get("scrollable", "false")),
                selected=_to_bool(attrib.get("selected", "false")),
                checkable=_to_bool(attrib.get("checkable", "false")),
                checked=_to_bool(attrib.get("checked", "false")),
                text=text,
                content_desc=content_desc,
                hint=hint,
            )
            nodes.append(node)
            current_parent = node.node_id

        for child in list(elem):
            dfs(child, current_parent, depth + 1)

    dfs(root, None, 0)

    child_counts = [0 for _ in nodes]
    for node in nodes:
        if node.parent_id is not None and 0 <= node.parent_id < len(nodes):
            child_counts[node.parent_id] += 1
    for idx, count in enumerate(child_counts):
        nodes[idx].child_count = count
    return nodes


def _tokenize_text(text: str) -> List[str]:
    if not text:
        return []
    return re.findall(r"[0-9a-z]+|[\u4e00-\u9fff]", text.lower())


def _token_jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _text_similarity(expected_norm: str, candidate_norm: str) -> float:
    if not expected_norm or not candidate_norm:
        return 0.0
    seq = SequenceMatcher(None, expected_norm, candidate_norm).ratio()
    tok = _token_jaccard(_tokenize_text(expected_norm), _tokenize_text(candidate_norm))
    contain = 1.0 if (expected_norm in candidate_norm or candidate_norm in expected_norm) else 0.0
    return min(1.0, 0.55 * seq + 0.35 * tok + 0.10 * contain)


def _aggregate_parent_text(node: UiNode, nodes: Sequence[UiNode], max_hops: int = 2) -> str:
    parts: List[str] = []
    cur = node
    hops = 0
    while cur.parent_id is not None and hops < max_hops:
        parent_idx = cur.parent_id
        if parent_idx < 0 or parent_idx >= len(nodes):
            break
        parent = nodes[parent_idx]
        if parent.text_norm:
            parts.append(parent.text_norm)
        if parent.resource_id:
            parts.append(normalize_text(parent.resource_id))
        cur = parent
        hops += 1
    return normalize_text(" ".join(parts))


def _score_text(node: UiNode, nodes: Sequence[UiNode], expected_text: str) -> float:
    expected_norm = normalize_text(expected_text)
    if not expected_norm:
        return 0.0
    node_text_only = normalize_text(node.text_norm)
    node_id_text = normalize_text(node.resource_id)
    node_text = normalize_text(" ".join([node_text_only, node_id_text]))
    parent_text = _aggregate_parent_text(node, nodes)
    return max(
        _text_similarity(expected_norm, node_text_only),
        _text_similarity(expected_norm, node_id_text),
        _text_similarity(expected_norm, node_text),
        _text_similarity(expected_norm, normalize_text(" ".join([node_text, parent_text]))),
    )


def _is_container_like(node: UiNode, screen_area: int) -> bool:
    cls = node.class_name.lower()
    is_container = any(key in cls for key in CONTAINER_CLASS_KEYWORDS)
    if not is_container:
        return False
    area_ratio = node.area / max(screen_area, 1)
    text_empty = not node.text_norm
    id_empty = not normalize_text(node.resource_id)
    return text_empty and id_empty and area_ratio >= 0.15


def _score_attr(
    node: UiNode,
    action_type: str,
    target_package: Optional[str],
    screen_area: int,
) -> float:
    action = (action_type or "").strip().upper()
    raw = 0.0
    if node.clickable:
        raw += 0.22
    if node.enabled:
        raw += 0.10
    else:
        raw -= 0.15
    if node.child_count == 0:
        raw += 0.05
    if node.scrollable:
        raw -= 0.03
    if _is_container_like(node, screen_area):
        raw -= 0.32

    norm_target_pkg = (target_package or "").strip()
    if norm_target_pkg:
        if node.package == norm_target_pkg:
            raw += 0.06
        elif node.package and node.package != norm_target_pkg:
            raw -= 0.12

    if node.package == "com.android.systemui" and action not in {"BACK", "HOME"}:
        raw -= 0.30

    return min(1.0, max(0.0, 0.5 + raw))


def _dynamic_weights(action_type: str, has_expected_text: bool) -> Tuple[float, float, float]:
    action = (action_type or "").strip().upper()
    if action in {"BACK", "HOME"}:
        return 0.70, 0.30, 0.0
    if has_expected_text:
        if action in {"CLICK_INPUT", "INPUT", "TYPE"}:
            return 0.45, 0.20, 0.35
        return 0.50, 0.25, 0.25
    return 0.72, 0.28, 0.0


def _dynamic_thresholds(action_type: str, has_expected_text: bool) -> Tuple[float, float]:
    action = (action_type or "").strip().upper()
    if action in {"BACK", "HOME"}:
        return 0.48, 0.05
    if has_expected_text:
        return 0.54, 0.06
    return 0.58, 0.08


def _retrieve_candidates(
    nodes: Sequence[UiNode],
    pred_box: BBox,
    screen_diag: float,
    max_candidates: int = 80,
    center_dist_threshold: float = 0.35,
) -> List[Tuple[UiNode, Dict[str, Any]]]:
    pred_area = max(bbox_area(pred_box), 1)
    candidate_pool: List[Tuple[UiNode, Dict[str, Any]]] = []

    for node in nodes:
        if node.area <= 0:
            continue
        inter = intersection_area(pred_box, node.bbox)
        cov = inter / node.area
        cont = inter / pred_area
        iou = bbox_iou(pred_box, node.bbox)
        dist = _center_distance_norm(pred_box, node.bbox, screen_diag)
        center_inside = _point_in_bbox(*bbox_center(node.bbox), pred_box)

        reasons: List[str] = []
        if center_inside:
            reasons.append("center_inside_pred")
        if cov >= 0.30:
            reasons.append("coverage>=0.3")
        if iou >= 0.01:
            reasons.append("iou>=0.01")
        if dist <= center_dist_threshold:
            reasons.append("center_dist<=thr")

        if reasons:
            candidate_pool.append(
                (
                    node,
                    {
                        "cov": cov,
                        "cont": cont,
                        "iou": iou,
                        "dist": dist,
                        "reasons": reasons,
                    },
                )
            )

    candidate_pool.sort(
        key=lambda item: (
            1 if item[0].clickable else 0,
            1 if item[0].child_count == 0 else 0,
            item[1]["cov"],
            item[1]["iou"],
            -item[1]["dist"],
        ),
        reverse=True,
    )
    return candidate_pool[:max_candidates]


def match_pred_box_to_xml(
    xml_str: str,
    pred_box: Sequence[int],
    action_type: str,
    expected_text: Optional[str] = None,
    target_package: Optional[str] = None,
) -> MatchResult:
    """Match a predicted box to the best XML node."""

    norm_pred = _normalize_bbox(pred_box)
    if bbox_area(norm_pred) <= 0:
        return MatchResult(
            status="PARSE_ERROR",
            message="Invalid predicted box area",
            pred_box=norm_pred,
            action_type=action_type,
            best_node=None,
        )

    if not xml_str:
        return MatchResult(
            status="PARSE_ERROR",
            message="Hierarchy XML is empty",
            pred_box=norm_pred,
            action_type=action_type,
            best_node=None,
        )

    try:
        nodes = parse_xml_to_nodes(xml_str)
    except Exception as exc:
        return MatchResult(
            status="PARSE_ERROR",
            message=f"XML parse failed: {exc}",
            pred_box=norm_pred,
            action_type=action_type,
            best_node=None,
        )

    if not nodes:
        return MatchResult(
            status="NO_CANDIDATE",
            message="No valid nodes with bounds found",
            pred_box=norm_pred,
            action_type=action_type,
            best_node=None,
            nodes=[],
        )

    screen_w, screen_h = _infer_screen_size(nodes, norm_pred)
    screen_diag = max(1.0, math.hypot(screen_w, screen_h))
    screen_area = max(1, screen_w * screen_h)

    retrieved = _retrieve_candidates(
        nodes=nodes,
        pred_box=norm_pred,
        screen_diag=screen_diag,
        max_candidates=80,
        center_dist_threshold=0.35,
    )
    if not retrieved:
        return MatchResult(
            status="NO_CANDIDATE",
            message="No candidates passed retrieval rules",
            pred_box=norm_pred,
            action_type=action_type,
            best_node=None,
            nodes=nodes,
        )

    has_expected_text = bool(normalize_text(expected_text or ""))
    w_geom, w_attr, w_text = _dynamic_weights(action_type, has_expected_text)
    t_score, t_margin = _dynamic_thresholds(action_type, has_expected_text)

    scored: List[CandidateScore] = []
    for node, metrics in retrieved:
        cov = float(metrics["cov"])
        cont = float(metrics["cont"])
        dist = float(metrics["dist"])
        iou = float(metrics["iou"])
        s_geom = 0.55 * cov + 0.15 * cont + 0.30 * (1.0 - dist)
        s_attr = _score_attr(node, action_type, target_package, screen_area)
        s_text = _score_text(node, nodes, expected_text or "") if has_expected_text else 0.0
        total = w_geom * s_geom + w_attr * s_attr + w_text * s_text

        scored.append(
            CandidateScore(
                node_id=node.node_id,
                bbox=node.bbox,
                package=node.package,
                class_name=node.class_name,
                resource_id=node.resource_id,
                clickable=node.clickable,
                enabled=node.enabled,
                depth=node.depth,
                child_count=node.child_count,
                cov=cov,
                cont=cont,
                iou=iou,
                dist=dist,
                s_geom=s_geom,
                s_attr=s_attr,
                s_text=s_text,
                total_score=total,
                retrieval_reasons=list(metrics["reasons"]),
            )
        )

    scored.sort(key=lambda c: c.total_score, reverse=True)
    best = scored[0]
    second = scored[1] if len(scored) > 1 else None
    margin = best.total_score - second.total_score if second else best.total_score

    is_match = best.total_score >= t_score and margin >= t_margin
    status = "MATCHED" if is_match else "AMBIGUOUS"
    message = (
        f"best={best.total_score:.3f}, margin={margin:.3f}"
        if is_match
        else f"ambiguous best={best.total_score:.3f}, margin={margin:.3f}"
    )

    return MatchResult(
        status=status,
        message=message,
        pred_box=norm_pred,
        action_type=action_type,
        best_node=nodes[best.node_id] if 0 <= best.node_id < len(nodes) else None,
        candidates=scored,
        nodes=nodes,
        thresholds={"score": t_score, "margin": t_margin},
        weights={"geom": w_geom, "attr": w_attr, "text": w_text},
    )


def check_clickable(
    best_node: UiNode,
    nodes: Sequence[UiNode],
    action_type: str,
) -> Tuple[Optional[UiNode], Dict[str, Any]]:
    action = (action_type or "").strip().upper()
    if action in {"BACK", "HOME"}:
        return best_node, {"ancestor_hops": 0, "reason": "navigation_action"}

    cur = best_node
    hops = 0
    while True:
        if cur.clickable and cur.enabled:
            return cur, {"ancestor_hops": hops}
        if cur.parent_id is None:
            break
        parent_idx = cur.parent_id
        if parent_idx < 0 or parent_idx >= len(nodes):
            break
        cur = nodes[parent_idx]
        hops += 1
        if hops > 20:
            break
    return None, {"ancestor_hops": hops}


def check_text(
    best_node: UiNode,
    nodes: Sequence[UiNode],
    expected_text: Optional[str],
    threshold: float = 0.45,
) -> Tuple[bool, float]:
    expected_norm = normalize_text(expected_text or "")
    if not expected_norm:
        return True, 1.0
    score = _score_text(best_node, nodes, expected_norm)
    return score >= threshold, score


def validate_match(
    match_result: MatchResult,
    expected_text: Optional[str] = None,
    action_type: Optional[str] = None,
    text_threshold: float = 0.45,
    require_clickable: bool = True,
) -> ValidationResult:
    """Validate a match result and return structured execution gate result."""

    act = action_type or match_result.action_type
    if match_result.status != "MATCHED" or match_result.best_node is None:
        return ValidationResult(
            ok=False,
            status="VALIDATION_FAIL",
            message=f"match status is {match_result.status}",
            error_code=match_result.status,
            best_node=match_result.best_node,
            click_node=None,
            details={"match_status": match_result.status},
        )

    best_node = match_result.best_node
    click_node = best_node

    if require_clickable and (act or "").strip().upper() not in {"BACK", "HOME"}:
        clickable_node, click_info = check_clickable(best_node, match_result.nodes, act)
        if clickable_node is None:
            return ValidationResult(
                ok=False,
                status="VALIDATION_FAIL",
                message="No clickable enabled node in ancestor chain",
                error_code="NOT_CLICKABLE",
                best_node=best_node,
                click_node=None,
                details={"clickable_check": click_info},
            )
        click_node = clickable_node

    ok_text, text_score = check_text(click_node, match_result.nodes, expected_text, threshold=text_threshold)
    if not ok_text:
        return ValidationResult(
            ok=False,
            status="VALIDATION_FAIL",
            message=f"Expected text mismatch (score={text_score:.3f})",
            error_code="TEXT_MISMATCH",
            best_node=best_node,
            click_node=click_node,
            details={"text_score": round(float(text_score), 3), "threshold": text_threshold},
        )

    return ValidationResult(
        ok=True,
        status="VALID",
        message="XML match validated",
        error_code="",
        best_node=best_node,
        click_node=click_node,
        details={},
    )


def node_to_dict(node: Optional[UiNode]) -> Optional[Dict[str, Any]]:
    if node is None:
        return None
    return {
        "node_id": node.node_id,
        "parent_id": node.parent_id,
        "bbox": list(node.bbox),
        "area": node.area,
        "depth": node.depth,
        "child_count": node.child_count,
        "text_norm": node.text_norm,
        "resource_id": node.resource_id,
        "class_name": node.class_name,
        "package": node.package,
        "clickable": node.clickable,
        "enabled": node.enabled,
        "scrollable": node.scrollable,
        "selected": node.selected,
        "checkable": node.checkable,
        "checked": node.checked,
    }


def candidate_to_dict(candidate: CandidateScore) -> Dict[str, Any]:
    return {
        "node_id": candidate.node_id,
        "bbox": list(candidate.bbox),
        "package": candidate.package,
        "class_name": candidate.class_name,
        "resource_id": candidate.resource_id,
        "clickable": candidate.clickable,
        "enabled": candidate.enabled,
        "depth": candidate.depth,
        "child_count": candidate.child_count,
        "cov": round(float(candidate.cov), 4),
        "cont": round(float(candidate.cont), 4),
        "iou": round(float(candidate.iou), 4),
        "dist": round(float(candidate.dist), 4),
        "s_geom": round(float(candidate.s_geom), 4),
        "s_attr": round(float(candidate.s_attr), 4),
        "s_text": round(float(candidate.s_text), 4),
        "total_score": round(float(candidate.total_score), 4),
        "retrieval_reasons": list(candidate.retrieval_reasons),
    }


def match_result_to_dict(match_result: MatchResult, include_candidates: int = 5) -> Dict[str, Any]:
    k = max(1, int(include_candidates))
    return {
        "status": match_result.status,
        "message": match_result.message,
        "pred_box": list(match_result.pred_box),
        "action_type": match_result.action_type,
        "best_node": node_to_dict(match_result.best_node),
        "thresholds": dict(match_result.thresholds),
        "weights": dict(match_result.weights),
        "candidate_count": len(match_result.candidates),
        "top_candidates": [candidate_to_dict(c) for c in match_result.candidates[:k]],
    }


def validation_result_to_dict(validation_result: ValidationResult) -> Dict[str, Any]:
    return {
        "ok": validation_result.ok,
        "status": validation_result.status,
        "message": validation_result.message,
        "error_code": validation_result.error_code,
        "best_node": node_to_dict(validation_result.best_node),
        "click_node": node_to_dict(validation_result.click_node),
        "details": dict(validation_result.details),
    }


__all__ = [
    "UiNode",
    "CandidateScore",
    "MatchResult",
    "ValidationResult",
    "bbox_center",
    "node_center",
    "parse_xml_to_nodes",
    "match_pred_box_to_xml",
    "check_clickable",
    "check_text",
    "validate_match",
    "node_to_dict",
    "candidate_to_dict",
    "match_result_to_dict",
    "validation_result_to_dict",
]
