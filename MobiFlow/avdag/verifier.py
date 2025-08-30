from __future__ import annotations
import json
import os
import base64
import time
import re
from typing import Dict, List, Tuple, Optional

from .types import Frame, NodeMatch, TaskSpec, VerifyResult, VerifierOptions, DecisionLog
from .dag import DAG
from .conditions import get_checker
from .loader import load_task
from .trace_loader import load_frames_from_dir
from .logger import get_logger, get_frame_logger, get_verifier_logger, get_llm_logger


def _collect_candidates_path_aware(frames: List[Frame], task: TaskSpec, dag: DAG, options: Optional[VerifierOptions], logs: List[DecisionLog]) -> Dict[str, List[int]]:
    """路径感知的候选帧收集：按拓扑顺序检查节点，支持路径剪枝和动态帧分配。

    - 按拓扑顺序逐节点检查；
    - 路径剪枝：只为当前可达路径上的节点分配帧资源；
    - 动态帧分配：根据已成功的路径动态决定下一个要检查的节点；
    - 帧独占优化：针对每个路径分支独立管理帧使用，避免跨路径冲突。
    """
    cand: Dict[str, List[int]] = {}
    # 路径级别的帧使用追踪：key为路径标识，value为该路径已使用的帧集合
    path_used_frames: Dict[str, set] = {}
    # 节点可达性：记录每个节点当前是否可达（基于其依赖/前驱的满足情况）
    node_reachable: Dict[str, bool] = {}
    # 节点的最小可行帧索引：用于路径约束
    min_frame_idx: Dict[str, int] = {}
    # 节点实际匹配的帧索引：用于正确计算后继节点的时序约束
    matched_frame_idx: Dict[str, int] = {}
    
    frame_logger = get_frame_logger()
    frame_logger.debug("开始路径感知的候选帧收集")
    
    # 按拓扑顺序处理节点
    topo_order = dag.topo_order()
    
    # 初始化：根节点（无依赖）总是可达
    for nid in dag.nodes:
        node = dag.nodes[nid]
        deps = node.deps or []
        or_parents = dag.parents_from_next.get(nid, [])
        
        if not deps and not or_parents:
            # 根节点
            node_reachable[nid] = True
            min_frame_idx[nid] = 0
        else:
            node_reachable[nid] = False
            min_frame_idx[nid] = 0
    
    for nid in topo_order:
        node = dag.nodes[nid]
        
        # 路径剪枝：只处理可达节点
        if not node_reachable.get(nid, False):
            cand[nid] = []
            frame_logger.debug(f"节点 {nid}: 不可达，跳过检查")
            continue
            
        hits: List[int] = []
        if not node.condition:
            cand[nid] = hits
            # 无条件节点视为总是成功，更新后继节点可达性
            _update_successor_reachability(nid, 0, dag, node_reachable, min_frame_idx, matched_frame_idx, frame_logger)
            continue
            
        checker = get_checker(node.condition.type)
        params = node.condition.params or {}
        
        # 判断当前节点是否需要OCR或LLM验证
        needs_ocr = _node_needs_ocr(node, params)
        needs_llm = _node_needs_llm(node, params)
        needs_frame_exclusive = needs_ocr or needs_llm
        
        # 确定该节点的搜索起始帧（基于路径约束）
        start_frame = min_frame_idx.get(nid, 0)
        
        # 获取当前路径上已使用的帧（仅针对帧独占节点）
        current_path_used = set()
        if needs_frame_exclusive:
            # 通过回溯当前节点的路径来确定已使用的帧
            current_path_used = _get_path_used_frames(nid, dag, cand, needs_frame_exclusive)
        
        frame_logger.debug(f"节点 {nid}: 需要独占={needs_frame_exclusive}, 起始帧={start_frame}, 路径已使用帧={sorted(current_path_used)}")
        
        for i in range(start_frame, len(frames)):
            # 对于需要独占帧的节点，跳过当前路径上已使用的帧
            if needs_frame_exclusive and i in current_path_used:
                frame_logger.trace(f"节点 {nid}: 跳过路径已使用帧 {i}")
                continue
                
            fr = frames[i]
            try:
                ok = checker.check(fr, params, options)
            except Exception as e:
                ok = False
                if options and options.log_decisions:
                    logs.append(DecisionLog(
                        frame_index=i, 
                        node_id=nid, 
                        strategy=node.condition.type, 
                        decision="inconclusive", 
                        details=str(e),
                        checker_type="exception",
                        checker_result=str(e)
                    ))
                frame_logger.error(f"节点 {nid}: 帧 {i} 检查异常 - {e}")
            else:
                if options and options.log_decisions:
                    # 获取检查器的详细结果
                    checker_result = ""
                    matched_keywords = []
                    unmatched_keywords = []
                    checker_type = node.condition.type
                    
                    # 从frame中获取OCR或LLM的详细结果
                    if "_last_ocr_result" in fr:
                        ocr_result = fr["_last_ocr_result"]
                        checker_type = "ocr"
                        checker_result = ocr_result.get("reason", "")
                        matched_keywords = ocr_result.get("matched_keywords", [])
                        unmatched_keywords = ocr_result.get("unmatched_keywords", [])
                    elif "_last_llm_result" in fr:
                        llm_result = fr["_last_llm_result"]
                        checker_type = "llm"
                        checker_result = llm_result.get("reason", "")
                    
                    logs.append(DecisionLog(
                        frame_index=i, 
                        node_id=nid, 
                        strategy=node.condition.type, 
                        decision=("hit" if ok else "miss"),
                        checker_type=checker_type,
                        checker_result=checker_result,
                        matched_keywords=matched_keywords if matched_keywords else None,
                        unmatched_keywords=unmatched_keywords if unmatched_keywords else None
                    ))
                frame_logger.trace(f"节点 {nid}: 帧 {i} 检查结果 {'成功' if ok else '失败'}")
            
            if ok:
                hits.append(i)
                frame_logger.debug(f"节点 {nid} 在帧 {i} 匹配成功")
                
                # 动态更新后继节点的可达性
                _update_successor_reachability(nid, i, dag, node_reachable, min_frame_idx, matched_frame_idx, frame_logger)
                
                # 对于帧独占节点，找到匹配后立即停止（早停机制）
                if needs_frame_exclusive:
                    frame_logger.debug(f"节点 {nid}: 找到匹配帧 {i}，启用早停机制")
                    break
                else:
                    # 对于不需要独占帧的节点，继续搜索更多匹配
                    frame_logger.trace(f"节点 {nid}: 帧 {i} 匹配成功，继续搜索更多候选帧")
                
        frame_logger.debug(f"节点 {nid}: 找到 {len(hits)} 个候选帧 {hits}")
        cand[nid] = hits
    
    return cand


def _get_path_used_frames(node_id: str, dag: DAG, cand: Dict[str, List[int]], is_exclusive: bool) -> set:
    """获取到达当前节点的路径上已使用的帧集合（用于帧独占逻辑）
    
    重要：当一个帧被使用时，该帧之前的所有帧也被标记为已使用，确保时序的线性关系
    """
    used_frames = set()
    
    # 回溯路径上的前驱节点
    def backtrack_path(nid: str, visited: set):
        if nid in visited:
            return
        visited.add(nid)
        
        node = dag.nodes.get(nid)
        if not node:
            return
            
        # 检查该节点是否使用了帧独占
        if node.condition:
            params = node.condition.params or {}
            node_needs_exclusive = _node_needs_ocr(node, params) or _node_needs_llm(node, params)
            
            # 如果该节点需要帧独占且有匹配的帧，将其及之前的所有帧加入已使用集合
            if node_needs_exclusive and cand.get(nid):
                for matched_frame in cand[nid]:
                    # 当某个帧被使用时，该帧及之前的所有帧都被标记为已使用
                    for frame_idx in range(matched_frame + 1):
                        used_frames.add(frame_idx)
        
        # 递归检查依赖节点
        for dep in (node.deps or []):
            backtrack_path(dep, visited)
        
        # 递归检查next路径的父节点
        for parent in dag.parents_from_next.get(nid, []):
            backtrack_path(parent, visited)
    
    # 从当前节点开始回溯（不包含当前节点本身）
    visited = {node_id}
    node = dag.nodes.get(node_id)
    if node:
        for dep in (node.deps or []):
            backtrack_path(dep, visited)
        for parent in dag.parents_from_next.get(node_id, []):
            backtrack_path(parent, visited)
    
    return used_frames


def _update_successor_reachability(node_id: str, frame_idx: int, dag: DAG, node_reachable: Dict[str, bool], min_frame_idx: Dict[str, int], matched_frame_idx: Dict[str, int], logger):
    """动态更新后继节点的可达性"""
    # 记录当前节点的实际匹配帧索引
    matched_frame_idx[node_id] = frame_idx
    
    # 更新通过deps依赖当前节点的后继节点
    for child_id in dag.children.get(node_id, []):
        child_node = dag.nodes[child_id]
        
        # 检查deps依赖
        if child_node.deps and node_id in child_node.deps:
            # 检查该子节点的所有deps依赖是否都已满足
            all_deps_satisfied = True
            max_dep_frame = -1
            
            for dep in child_node.deps:
                if not node_reachable.get(dep, False):
                    all_deps_satisfied = False
                    break
                # 使用实际匹配的帧索引而不是最小帧索引
                dep_matched_frame = matched_frame_idx.get(dep, -1)
                if dep_matched_frame >= 0:
                    max_dep_frame = max(max_dep_frame, dep_matched_frame)
            
            if all_deps_satisfied and max_dep_frame >= 0:
                node_reachable[child_id] = True
                min_frame_idx[child_id] = max_dep_frame + 1  # 严格时序：后继必须发生在依赖之后
                logger.debug(f"节点 {child_id}: 通过deps依赖变为可达，最小帧索引={min_frame_idx[child_id]} (基于依赖节点 {node_id} 的匹配帧 {frame_idx})")
        
        # 检查next路径依赖（OR语义）
        elif not child_node.deps and node_id in dag.parents_from_next.get(child_id, []):
            node_reachable[child_id] = True
            min_frame_idx[child_id] = frame_idx + 1  # 后继节点必须在当前节点之后
            logger.debug(f"节点 {child_id}: 通过next路径变为可达，最小帧索引={min_frame_idx[child_id]} (基于父节点 {node_id} 的匹配帧 {frame_idx})")


def _collect_candidates(frames: List[Frame], task: TaskSpec, options: Optional[VerifierOptions], logs: List[DecisionLog]) -> Dict[str, List[int]]:
    """兼容性包装：调用路径感知的候选帧收集函数"""
    dag = DAG(task.nodes)
    return _collect_candidates_path_aware(frames, task, dag, options, logs)


def _node_needs_ocr(node, params: Dict[str, any]) -> bool:
    """判断节点是否需要OCR验证"""
    if node.condition and node.condition.type in ("escalate", "juxtaposition"):
        checker_params = params or {}
        return "ocr" in checker_params
    return False


def _node_needs_llm(node, params: Dict[str, any]) -> bool:
    """判断节点是否需要LLM验证"""
    if node.condition and node.condition.type in ("escalate", "juxtaposition"):
        checker_params = params or {}
        return "llm" in checker_params
    return False


def _min_feasible_index(cands: List[int], min_required: int) -> Optional[int]:
    """在 cands 中找出第一个 >= min_required 的索引。若无则返回 None。"""
    # 二分也可，这里线性即可（帧通常不多）
    for x in cands:
        if x >= min_required:
            return x
    return None


def _calculate_total_score(matched_nodes: List[str], task: TaskSpec) -> int:
    """计算匹配节点的总分数"""
    total_score = 0
    node_dict = {node.id: node for node in task.nodes}
    
    for node_id in matched_nodes:
        if node_id in node_dict:
            total_score += node_dict[node_id].score
    
    return total_score


def verify(frames: List[Frame], task: TaskSpec, options: Optional[VerifierOptions] = None) -> VerifyResult:
    """给定帧与任务，判断是否存在符合拓扑依赖的满足路径。

    算法：拓扑序 DP，记录每个节点的最小可行帧索引（满足依赖且该节点匹配）。
    """
    dag = DAG(task.nodes)
    
    # 确定成功节点集合（用于路径分析）
    succ_nodes: List[str]
    if task.success:
        if task.success.any_of:
            succ_nodes = task.success.any_of
        elif task.success.all_of:
            succ_nodes = task.success.all_of
        else:
            # 空 success 定义：视作 sinks
            succ_nodes = dag.sinks()
    else:
        succ_nodes = dag.sinks()
    
    # 输出可能的路径到日志
    verifier_logger = get_verifier_logger()
    dag.log_possible_paths(succ_nodes, verifier_logger)
    
    logs: List[DecisionLog] = []
    cands = _collect_candidates(frames, task, options, logs)

    topo = dag.topo_order()
    min_idx: Dict[str, Optional[int]] = {nid: None for nid in dag.nodes}
    prev: Dict[str, Optional[str]] = {nid: None for nid in dag.nodes}  # 用于回溯路径（记录选择的父节点）

    for nid in topo:
        node = dag.nodes[nid]
        deps = node.deps or []
        # 依赖与路径父节点：
        # - 若声明了 deps，则采用 AND 语义（保持兼容）；
        # - 否则，若存在由 next 形成的父节点集合，则采用 OR 语义（任一父成功即可）。
        dep_idx = 0
        chosen_parent: Optional[str] = None
        if deps:
            latest_dep = -1
            latest_src = None
            for d in deps:
                if min_idx[d] is None:
                    latest_dep = None
                    break
                if min_idx[d] > latest_dep:
                    latest_dep = min_idx[d]  # type: ignore
                    latest_src = d
            if latest_dep is None:
                min_idx[nid] = None
                continue
            dep_idx = latest_dep
            chosen_parent = latest_src
        else:
            # OR 语义父节点（来自 next）
            or_parents = dag.parents_from_next.get(nid, [])
            if or_parents:
                # 取最早完成的父节点作为起点
                available = [(p, min_idx[p]) for p in or_parents if min_idx.get(p) is not None]
                if not available:
                    min_idx[nid] = None
                    continue
                chosen_parent, parent_idx = min(available, key=lambda x: x[1])  # type: ignore
                dep_idx = int(parent_idx)  # type: ignore
            else:
                # 无依赖与无路径父节点，视作根节点
                dep_idx = 0
                chosen_parent = None

        # 从候选中找第一个满足顺序的帧
        hit = _min_feasible_index(cands.get(nid, []), dep_idx)
        if hit is not None:
            min_idx[nid] = hit
            # 记录采用的父节点（用于路径回溯）
            prev[nid] = chosen_parent

    # 决定成功节点集合
    succ_nodes: List[str]
    if task.success:
        if task.success.any_of:
            succ_nodes = task.success.any_of
            ok = any(min_idx.get(n) is not None for n in succ_nodes)
        elif task.success.all_of:
            succ_nodes = task.success.all_of
            ok = all(min_idx.get(n) is not None for n in succ_nodes)
        else:
            # 空 success 定义：视作 sinks
            succ_nodes = dag.sinks()
            ok = any(min_idx.get(n) is not None for n in succ_nodes)
    else:
        succ_nodes = dag.sinks()
        ok = any(min_idx.get(n) is not None for n in succ_nodes)

    if not ok:
        # 即使最终成功条件不满足，也回溯已经匹配的节点
        matched: List[Tuple[int, str]] = []  # (frame_idx, node_id)
        
        # 找出所有成功匹配的节点（不论是否达到最终成功条件）
        successful_nodes = []
        for nid in dag.nodes:
            if min_idx.get(nid) is not None:
                matched.append((min_idx[nid], nid))  # type: ignore
                successful_nodes.append(nid)
        
        verifier_logger = get_verifier_logger()
        verifier_logger.debug(f"成功匹配的节点: {successful_nodes}")
        verifier_logger.debug(f"成功节点要求: {succ_nodes}")
        verifier_logger.debug(f"min_idx状态: {min_idx}")
        
        matched.sort(key=lambda x: x[0])
        
        # 计算已匹配节点的总分
        matched_node_ids = [nid for idx, nid in matched]
        total_score = _calculate_total_score(matched_node_ids, task)
        
        # 构建详细的失败原因，包含最后一个检查的节点信息
        detailed_reason = "no feasible success path"
        if logs:
            # 找到最后一次检查的日志
            last_log = logs[-1]
            if last_log.checker_result:
                detailed_reason += f" (最后检查节点 {last_log.node_id}: {last_log.checker_result})"
            elif last_log.unmatched_keywords:
                detailed_reason += f" (最后检查节点 {last_log.node_id}: 未匹配关键词 {last_log.unmatched_keywords})"
        
        # 若存在某些节点完全无法判断（无命中且存在 escalation/高阶策略未配置），标记人工复核
        manual = any(
            (n.condition and n.condition.type in ("escalate",)) for n in task.nodes
        ) and (options is None or (options.llm is None and options.ocr is None))
        
        return VerifyResult(
            ok=False, 
            matched=[NodeMatch(node_id=nid, frame_index=idx) for idx, nid in matched],
            reason=detailed_reason, 
            logs=logs, 
            manual_review_needed=manual,
            total_score=total_score
        )

    # 回溯出一条可行路径：
    # 若 any_of，则取 min_idx 最小的那个；若 all_of，则回溯每个并合并。
    matched: List[Tuple[int, str]] = []  # (frame_idx, node_id)

    def backtrack(start: str):
        chain = []
        cur = start
        while cur is not None and min_idx.get(cur) is not None:
            chain.append((min_idx[cur], cur))  # type: ignore
            cur = prev.get(cur)
        # 依赖方向回溯得到自底向上链条，反转
        chain.reverse()
        return chain

    if task.success and task.success.all_of:
        added = set()
        for n in succ_nodes:
            if min_idx.get(n) is not None:
                for item in backtrack(n):
                    if item[1] not in added:
                        matched.append(item)
                        added.add(item[1])
    else:
        # any_of 或默认 sinks：选择最早完成的一个成功节点
        candidate_succ = [n for n in succ_nodes if min_idx.get(n) is not None]
        target = min(candidate_succ, key=lambda n: min_idx[n])
        matched = backtrack(target)

    matched.sort(key=lambda x: x[0])
    
    # 计算匹配节点的总分
    matched_node_ids = [nid for idx, nid in matched]
    total_score = _calculate_total_score(matched_node_ids, task)
    
    # 构建成功的详细原因，包含最后一个成功节点的信息
    detailed_reason = None
    if logs:
        # 找到最后一次成功的检查日志
        success_logs = [log for log in logs if log.decision == "hit"]
        if success_logs:
            last_success_log = success_logs[-1]
            if last_success_log.checker_result:
                detailed_reason = f"任务验证成功 (最后成功节点 {last_success_log.node_id}: {last_success_log.checker_result})"
            elif last_success_log.matched_keywords:
                detailed_reason = f"任务验证成功 (最后成功节点 {last_success_log.node_id}: 匹配关键词 {last_success_log.matched_keywords})"
    
    return VerifyResult(
        ok=True,
        matched=[NodeMatch(node_id=nid, frame_index=idx) for idx, nid in matched],
        reason=detailed_reason,
        logs=logs,
        manual_review_needed=False,
        total_score=total_score
    )


def verify_task(task_path: str, trace_path: str) -> VerifyResult:
    task = load_task(task_path)
    with open(trace_path, "r", encoding="utf-8") as f:
        frames = json.load(f)
        assert isinstance(frames, list)
    return verify(frames, task)


def verify_task_folder(task_path: str, trace_folder: str, options: Optional[VerifierOptions] = None) -> VerifyResult:
    task = load_task(task_path)
    frames = load_frames_from_dir(trace_folder)
    return verify(frames, task, options)


def make_llm_options(api_key: str, base_url: str, model: str = "google/gemini-2.5-flash", force_llm: bool = False, max_retries: int = 3, retry_delay: float = 1.0) -> VerifierOptions:
    """构造带 LLM 回调的 VerifierOptions，使用 LangChain OpenAI 兼容接口。

    注意：不在库内硬编码 key；由调用方传入。
    
    Args:
        api_key: API密钥
        base_url: API基础URL 
        model: 模型名称
        force_llm: 是否强制使用LLM验证
        max_retries: LLM请求的最大重试次数
        retry_delay: 重试间隔（秒）
    """
    try:
        from langchain_openai import ChatOpenAI  # type: ignore
    except Exception:  # pragma: no cover - 可选依赖
        def _llm(_ctx):
            return None
        return VerifierOptions(llm=_llm, force_llm_verification=force_llm)

    # 创建 LangChain LLM 客户端
    client = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.2,
        max_tokens=3000,
        timeout=40
    )

    def _llm(ctx: Dict[str, any]) -> Optional[bool]:  # type: ignore
        params = (ctx.get("params") or {})
        prompt = params.get("prompt") or "请判断该步骤是否达成预期。"
        frame = ctx.get("frame") or {}
        reasoning = frame.get("reasoning") or ""
        action = frame.get("action") or {}
        task_desc = frame.get("task_description") or ""
        
        # 获取当前帧和下一帧的图片
        current_image = frame.get("image")
        next_frame = frame.get("_next") or {}
        next_image = next_frame.get("image")

        # TODO: 必要时考虑增加上一帧图片
        prev_frame = frame.get("_prev") or {}
        prev_image = prev_frame.get("image")

        # 构建图片内容列表
        image_contents = []
        current_and_next = False
        prev_and_current = False
        
        if current_image and next_image:
            current_and_next = True
            if current_image and os.path.exists(current_image):
                with open(current_image, "rb") as f:
                    import base64
                    current_image_b64 = base64.b64encode(f.read()).decode()
                    image_contents.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{current_image_b64}"
                        }
                    })
            
            if next_image and os.path.exists(next_image):
                with open(next_image, "rb") as f:
                    import base64
                    next_image_b64 = base64.b64encode(f.read()).decode()
                    image_contents.append({
                        "type": "image_url", 
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{next_image_b64}"
                        }
                    })
        elif prev_image and current_image:
            if prev_image and os.path.exists(prev_image):
                prev_and_current = True
                with open(prev_image, "rb") as f:
                    import base64
                    prev_image_b64 = base64.b64encode(f.read()).decode()
                    image_contents.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{prev_image_b64}"
                        }
                    })
            if current_image and os.path.exists(current_image):
                with open(current_image, "rb") as f:
                    import base64
                    current_image_b64 = base64.b64encode(f.read()).decode()
                    image_contents.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{current_image_b64}"
                        }
                    })
        else:
            llm_logger.error("状态截图均不可用，无法进行判断。")
            return None

        # sys = "你是一个针对移动端操作的自动化验证助手，请主要基于提供的截图和上下文辅助谨慎判断任务是否正确完成，并返回JSON格式的结果。"
        sys = """
        作为移动端操作自动化验证专家，您需要严格基于视觉证据分析截图状态，结合上下文判断任务节点是否达成。
        核心原则：
        1. 截图是主要判定依据，文本信息仅作辅助参考（可能包含错误）
        2. 需识别界面元素的视觉变化及状态转移
        3. 对模糊场景保持保守判断
        4. 如无必要，不要展开冗长推理
        """
        
        # 构建文本内容
        if current_and_next:
            text_content = (
                 "## 关键节点验证任务\n"
                f"**全局任务task_description**: {task_desc}\n"
                f"**当前节点要求**: {prompt}\n\n"
                "移动端执行参考（注意：可能包含错误）\n"
                f"- 操作意图: {reasoning}\n"
                f"- 执行动作: {action}\n"
                "## 分析指令\n"
                "1. **视觉对比分析**：\n"
                "   - 当前帧截图：识别准备状态是否满足任务节点要求（按钮/文本/图标等）\n"
                "   - 下一帧截图：检测操作后界面变化（状态变更/新元素/提示消失等）是否满足任务节点要求\n"
                "   - 对比两帧差异\n"
                "2. **容错机制**：\n"
                "   - 若文本描述与截图状态冲突 → 以截图为准\n"
                "   - 若未观测到满足节点要求的状态或变化 → 返回no\n"
                "3. **判定标准**：\n"
                "   ❌ 以下情况视为失败：\n"
                "   - 关键元素未变化（如按钮仍可点击）\n"
                "   - 出现错误提示/异常状态\n"
                "   - 界面变化不符合任务逻辑链条\n\n"
                "## 输出要求\n"
                "请严格按照以下JSON格式返回结果：\n"
                '{"result": "yes", "reason": "简要说明判断原因"} 或 {"result": "no", "reason": "简要说明失败原因"}'
            )
        elif prev_and_current:
            text_content = (
                 "## 关键节点验证任务\n"
                f"**全局任务**: {task_desc}\n"
                f"**当前节点要求**: {prompt}\n\n"
                "移动端执行参考（注意：可能包含错误）\n"
                f"- 操作意图: {reasoning}\n"
                f"- 执行动作: {action}\n"
                "## 分析指令\n"
                "1. **视觉对比分析**：\n"
                "   - 上一帧截图：上一步操作前识别准备状态内容状态（按钮/文本/图标等）\n"
                "   - 当前帧截图：检测上步操作后、当前界面变化（状态变更/新元素/提示消失等）\n"
                "   - 对比两帧差异是否满足任务节点要求\n"
                "2. **容错机制**：\n"
                "   - 若文本描述与截图状态冲突 → 以截图为准\n"
                "   - 若未观测到满足节点要求的状态或变化 → 返回no\n"
                "3. **判定标准**：\n"
                "   ❌ 以下情况视为失败：\n"
                "   - 关键元素未变化（如按钮仍可点击）\n"
                "   - 出现错误提示/异常状态\n"
                "   - 界面变化不符合任务逻辑链条\n\n"
                "## 输出要求\n"
                "请严格按照以下JSON格式返回结果：\n"
                '{"result": "yes", "reason": "简要说明判断原因"} 或 {"result": "no", "reason": "简要说明失败原因"}'
            )
        ##  精简版
                # f"当前节点判断任务: {prompt}\n"
                # f"总任务描述: {task_desc}\n"
                # f"移动端推理(仅参考): {reasoning}\n"
                # f"移动端动作(仅参考): {action}\n"
                # "请主要基于提供的前一帧和当前帧状态截图，判断该步骤是否达成预期。\n"
                # "“推理”和“动作”由移动端操作提供，仅参考，不一定正确，需实际按截图进一步分析。\n"
                # "请严格按照以下JSON格式返回结果：\n"
                # '{"result": "yes", "reason": "简要说明判断原因"} 或 {"result": "no", "reason": "简要说明失败原因"}'
        ###

        # 构建消息内容
        message_content = [{"type": "text", "text": text_content}]
        message_content.extend(image_contents)
        
        # LangChain 消息格式
        from langchain_core.messages import SystemMessage, HumanMessage
        
        messages = [
            SystemMessage(content=sys),
            HumanMessage(content=message_content)
        ]
        
        llm_logger = get_llm_logger()
        llm_logger.debug(f"prompt with {len(image_contents)} images: {text_content}")
        
        # 重试配置
        max_retries = 3  # 默认值
        retry_delay = 1.0  # 默认值（秒）
        
        # 从上下文中获取options配置（如果有的话）
        if ctx and "options" in ctx:
            options = ctx["options"]
            if hasattr(options, 'max_llm_retries'):
                max_retries = options.max_llm_retries
            if hasattr(options, 'llm_retry_delay'):
                retry_delay = options.llm_retry_delay
        
        for attempt in range(max_retries):
            response_text = None
            try:
                # 使用 LangChain 客户端调用
                resp = client.invoke(messages)
                
                # 检查响应是否有效
                if not resp or not resp.content:
                    if attempt < max_retries - 1:
                        llm_logger.warning(f"received empty or invalid response from LLM (attempt {attempt + 1}/{max_retries}), retrying...")
                        import time
                        time.sleep(retry_delay)
                        continue
                    else:
                        llm_logger.error(f"received empty or invalid response from LLM after {max_retries} attempts")
                        return None
                
                response_text = resp.content.strip()
                
                # 检查响应内容是否为空
                if not response_text:
                    if attempt < max_retries - 1:
                        llm_logger.warning(f"received empty response content (attempt {attempt + 1}/{max_retries}), retrying...")
                        import time
                        time.sleep(retry_delay)
                        continue
                    else:
                        llm_logger.error(f"received empty response content after {max_retries} attempts")
                        return None
                
                llm_logger.debug(f"raw response (attempt {attempt + 1}): {response_text}")
                
                # 尝试解析JSON响应
                try:
                    import json
                    import re
                    
                    def extract_json_from_text(text):
                        """从文本中提取JSON内容，处理包含```json标记的情况"""
                        # 移除可能的markdown代码块标记
                        cleaned_text = text.strip()
                        
                        # 尝试匹配 ```json ... ``` 格式
                        json_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', cleaned_text, re.DOTALL)
                        if json_block_match:
                            return json_block_match.group(1).strip()
                        
                        # 尝试匹配纯JSON格式（查找第一个完整的JSON对象）
                        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned_text, re.DOTALL)
                        if json_match:
                            return json_match.group(0).strip()
                        
                        # 如果都没找到，返回原文本
                        return cleaned_text
                    
                    # 先尝试直接解析
                    try:
                        result_json = json.loads(response_text)
                    except json.JSONDecodeError:
                        # 如果直接解析失败，尝试提取JSON内容
                        extracted_json = extract_json_from_text(response_text)
                        llm_logger.debug(f"extracted json: {extracted_json}")
                        result_json = json.loads(extracted_json)
                    
                    result = result_json.get("result", "").lower()
                    reason = result_json.get("reason", "")
                    
                    llm_logger.debug(f"parsed result: {result}, reason: {reason}")
                    
                    # 在frame中记录LLM检查结果
                    llm_result_record = {
                        'success': result == "yes",
                        'reason': reason
                    }
                    frame['_last_llm_result'] = llm_result_record
                                    
                    if result == "yes":
                        return True
                    elif result == "no":
                        return False
                    elif result == "":
                        # 结果为空，可能需要重试
                        if attempt < max_retries - 1:
                            llm_logger.warning(f"received empty result field (attempt {attempt + 1}/{max_retries}), retrying...")
                            import time
                            time.sleep(retry_delay)
                            continue
                        else:
                            llm_logger.warning(f"unexpected empty result value after {max_retries} attempts")
                            frame['_last_llm_result'] = {
                                'success': False,
                                'reason': 'LLM返回空结果'
                            }
                            return None
                    else:
                        # 结果值不是预期的，但不重试，直接返回None
                        llm_logger.warning(f"unexpected result value: {result}")
                        frame['_last_llm_result'] = {
                            'success': False,
                            'reason': f'LLM返回异常结果: {result}'
                        }
                        return None
                        
                except (json.JSONDecodeError, AttributeError, TypeError) as e:
                    # JSON解析失败，尝试从文本中提取结果
                    llm_logger.warning(f"JSON parsing failed ({str(e)}), trying text extraction from: {response_text}")
                    text_lower = response_text.lower()
                    if "yes" in text_lower and "no" not in text_lower:
                        llm_logger.debug("extracted result: yes (from text)")
                        frame['_last_llm_result'] = {
                            'success': True,
                            'reason': 'LLM返回yes (从文本提取)'
                        }
                        return True
                    elif "no" in text_lower and "yes" not in text_lower:
                        llm_logger.debug("extracted result: no (from text)")
                        frame['_last_llm_result'] = {
                            'success': False,
                            'reason': 'LLM返回no (从文本提取)'
                        }
                        return False
                    else:
                        # 如果文本提取也失败，且还有重试机会，则重试
                        if attempt < max_retries - 1:
                            llm_logger.warning(f"unable to extract clear result from response (attempt {attempt + 1}/{max_retries}), retrying...")
                            import time
                            time.sleep(retry_delay)
                            continue
                        else:
                            llm_logger.warning(f"unable to extract clear result from response after {max_retries} attempts: {response_text}")
                            frame['_last_llm_result'] = {
                                'success': False,
                                'reason': 'LLM响应无法解析'
                            }
                            return None
                
            except Exception as e:
                if attempt < max_retries - 1:
                    llm_logger.warning(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}, retrying...")
                    import time
                    time.sleep(retry_delay)
                    continue
                else:
                    llm_logger.error(f"LLM call failed after {max_retries} attempts: {e}")
                    frame['_last_llm_result'] = {
                        'success': False,
                        'reason': f'LLM调用失败: {str(e)}'
                    }
                    return None
        
        # 如果到这里说明所有重试都失败了
        llm_logger.error("All retry attempts exhausted")
        frame['_last_llm_result'] = {
            'success': False,
            'reason': 'LLM所有重试尝试均失败'
        }
        return None

    return VerifierOptions(
        llm=_llm, 
        force_llm_verification=force_llm,
        max_llm_retries=max_retries,
        llm_retry_delay=retry_delay
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify LLM mobile agent task by DAG")
    parser.add_argument("task", help="path to task yaml/json")
    parser.add_argument("trace", help="path to trace json frames")
    args = parser.parse_args()

    res = verify_task(args.task, args.trace)
    print(json.dumps({
        "ok": res.ok,
        "matched": [{"node": m.node_id, "frame": m.frame_index} for m in res.matched],
        "reason": res.reason,
    }, ensure_ascii=False, indent=2))
