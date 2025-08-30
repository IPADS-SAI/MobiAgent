from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable

Frame = Dict[str, Any]  # 简化：每帧就是一个字典

@dataclass
class ConditionSpec:
    type: str
    params: Dict[str, Any]

@dataclass
class NodeSpec:
    id: str
    name: Optional[str] = None
    deps: Optional[List[str]] = None
    # next: 声明后继节点（可选）。用于定义可选路径（OR语义）。
    # 注意：当同一节点同时定义了 deps 与 next 指向它的父节点时，deps 仍然采用 AND 语义优先；
    # 未定义 deps 时，来自 next 的父子边在验证阶段按 OR 语义处理。
    next: Optional[List[str]] = None
    condition: Optional[ConditionSpec] = None
    # score: 节点分数，默认为10分
    score: int = 10

@dataclass
class SuccessSpec:
    any_of: Optional[List[str]] = None
    all_of: Optional[List[str]] = None

@dataclass
class TaskSpec:
    task_id: str
    nodes: List[NodeSpec]
    success: Optional[SuccessSpec] = None

@dataclass
class NodeMatch:
    node_id: str
    frame_index: int

@dataclass
class VerifyResult:
    ok: bool
    matched: List[NodeMatch]
    reason: Optional[str] = None
    # 新增：判定过程日志与人工复核标记（兼容旧测试）
    logs: List["DecisionLog"] = field(default_factory=list)
    manual_review_needed: bool = False
    # 新增：任务总分（成功匹配的节点分数之和）
    total_score: int = 0


@dataclass
class DecisionLog:
    frame_index: int
    node_id: str
    strategy: str
    decision: str  # hit | miss | inconclusive
    details: Optional[str] = None
    # 新增字段：记录检查器的详细结果
    checker_type: Optional[str] = None  # ocr | llm | text | regex | ui | action | dynamic_match
    checker_result: Optional[str] = None  # 检查器的详细结果或原因
    matched_keywords: Optional[List[str]] = None  # 匹配成功的关键词
    unmatched_keywords: Optional[List[str]] = None  # 未匹配的关键词


@dataclass
class VerifierOptions:
    """可注入的判定能力与策略顺序。

    - ocr: 函数接受 Frame，返回识别出的文本（字符串）或 None 表示不支持/不确定。
    - llm: 函数接受上下文字典，返回 True/False/None（None 表示不确定）。
    - escalation_order: 策略升级顺序，默认 [text, regex, ui, action, dynamic_match, icons, ocr, llm]
    - log_decisions: 是否记录详细日志。
    - force_llm_verification: 是否强制使用LLM验证，即使其他策略已经匹配。
    - prevent_frame_backtrack: 是否防止帧回退（默认True），一旦某个帧被OCR/LLM使用，之前的帧也标记为已使用。
    - ocr_frame_exclusive: OCR验证时是否独占使用帧（默认True），防止同一帧被多个OCR节点重复使用。
    - llm_frame_exclusive: LLM验证时是否独占使用帧（默认True），防止同一帧被多个LLM节点重复使用。
    """
    ocr: Optional[Callable[[Frame], Optional[str]]] = None
    llm: Optional[Callable[[Dict[str, Any]], Optional[bool]]] = None
    escalation_order: List[str] = field(default_factory=lambda: [
        "text", "regex", "ui", "action", "icons", "ocr", "llm"
    ])
    log_decisions: bool = True
    force_llm_verification: bool = False
    prevent_frame_backtrack: bool = True
    ocr_frame_exclusive: bool = True
    llm_frame_exclusive: bool = True
    max_llm_retries: int = 3  # LLM请求的最大重试次数
    llm_retry_delay: float = 1.0  # LLM重试之间的延迟（秒）

__all__ = [
    "Frame",
    "ConditionSpec",
    "NodeSpec",
    "SuccessSpec",
    "TaskSpec",
    "NodeMatch",
    "VerifyResult",
    "DecisionLog",
    "VerifierOptions",
]
