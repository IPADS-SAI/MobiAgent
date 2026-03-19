
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import hashlib
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


class TemplateGraphBuilder:
    """Incrementally build and query app-level template state graph."""

    SCHEMA_VERSION = 1

    HIGH_MATCH_THRESHOLD = 0.82
    MID_MATCH_THRESHOLD = 0.62

    NODE_LOW_CONFIDENCE = 0.68
    EDGE_LOW_CONFIDENCE = 0.58
    SEARCH_MODE_DEFAULT = "conservative"
    SEARCH_CONSERVATIVE_NODE_SCORE_MIN = 0.90
    SEARCH_CONSERVATIVE_NODE_CONFIDENCE_MIN = 0.60
    SEARCH_CONSERVATIVE_EDGE_CONFIDENCE_MIN = 0.60
    SEARCH_CONSERVATIVE_EDGE_PROBABILITY_MIN = 0.34
    SEARCH_CANDIDATE_NODE_TOP_K = 3
    SEARCH_CANDIDATE_EDGE_TOP_K = 3
    SEARCH_MAX_ACTIONS = 6

    MAX_EXAMPLES_PER_NODE = 6
    MAX_SIGNATURES_PER_KIND = 12
    MAX_TOKEN_FEATURES = 500

    POPUP_KEYWORDS = (   # 后续可以更新！
        "红包",
        "弹窗",
        "允许",
        "拒绝",
        "稍后",
        "以后再说",
        "广告",
        "跳过",
        "关闭",
        "升级",
        "更新",
        "订阅",
        "立即",
        "同意",
        "取消",
        "allow",
        "deny",
        "skip",
        "close",
        "update",
        "later",
        "ad",
    )
    INTERRUPTION_KEYWORDS = (
        "权限",
        "登录",
        "网络",
        "失败",
        "异常",
        "错误",
        "重试",
        "验证码",
        "授权",
        "permission",
        "login",
        "network",
        "failed",
        "error",
        "retry",
        "captcha",
    )

    def __init__(self, graph_dir: str, app_name: str):
        self.graph_dir = graph_dir
        self.app_name = app_name
        self.safe_app_name = self._safe_name(app_name)
        os.makedirs(self.graph_dir, exist_ok=True)
        self.graph_file = os.path.join(self.graph_dir, f"{self.safe_app_name}.graph.json")

        self.graph = self._load_or_init_graph()
        self._rebuild_indices()

    # ----------------------------- public APIs -----------------------------

    def update_from_run(
        self,
        data_dir: str,
        actions: List[Dict[str, Any]],
        task_desc: str,
        task_hash: str,
        stop_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update graph with one run trace (`actions` + step XML files)."""
        summary: Dict[str, Any] = {
            "transitions": 0,
            "nodes_created": 0,
            "nodes_updated": 0,
            "node_matched_strict": 0,
            "node_matched_high": 0,
            "node_matched_soft": 0,
            "edges_created": 0,
            "edges_updated": 0,
            "graph_file": self.graph_file,
        }
        if not isinstance(actions, list) or not actions:
            self._touch_meta(run_increment=1)
            self._save_graph()
            return summary

        ordered_actions = self._normalize_actions(actions)
        transitions = self._build_transitions(
            data_dir=data_dir,
            actions=ordered_actions,
            stop_reason=stop_reason,
        )
        if not transitions:
            self._touch_meta(run_increment=1)
            self._save_graph()
            return summary

        task_tokens = self._extract_task_tokens(task_desc)
        latest_main_node_id: Optional[str] = None

        for trans in transitions:
            action = trans["action"]
            from_obs = trans["from_obs"]
            to_obs = trans["to_obs"]

            from_ctx = self._build_context(
                task_hash=task_hash,
                task_tokens=task_tokens,
                action_name=str(action.get("action", "")),
                replayed=bool(action.get("replayed")),
                stop_reason=stop_reason,
                step_idx=trans.get("step"),
                xml_file=from_obs.get("xml_file"),
                screenshot_file=action.get("screenshot_file"),
            )
            from_ui_meta = action.get("ui_meta") if isinstance(action, dict) else None
            from_resolve = self._resolve_or_create_node(from_obs, from_ctx, ui_meta=from_ui_meta)
            from_node = self.node_by_id[from_resolve["node_id"]]
            summary["nodes_updated"] += 1
            if from_resolve["created"]:
                summary["nodes_created"] += 1
            else:
                summary[self._summary_key_for_match(from_resolve["match_type"])] += 1

            to_ctx = self._build_context(
                task_hash=task_hash,
                task_tokens=task_tokens,
                action_name=str(action.get("action", "")),
                replayed=bool(action.get("replayed")),
                stop_reason=stop_reason,
                step_idx=trans.get("step", 0) + 1 if isinstance(trans.get("step"), int) else None,
                xml_file=to_obs.get("xml_file"),
                screenshot_file=action.get("screenshot_file"),
            )
            to_resolve = self._resolve_or_create_node(to_obs, to_ctx, ui_meta=None)
            to_node = self.node_by_id[to_resolve["node_id"]]
            summary["nodes_updated"] += 1
            if to_resolve["created"]:
                summary["nodes_created"] += 1
            else:
                summary[self._summary_key_for_match(to_resolve["match_type"])] += 1

            if from_node.get("type") == "main":
                latest_main_node_id = from_node["node_id"]

            transition_type, to_type = self._classify_transition(
                from_obs=from_obs,
                from_node=from_node,
                to_obs=to_obs,
                to_node=to_node,
                action=action,
                latest_main_node_id=latest_main_node_id,
            )
            self._promote_node_type(to_node, to_type)

            edge_res = self._upsert_edge(
                from_node_id=from_node["node_id"],
                to_node_id=to_node["node_id"],
                action=action,
                transition_type=transition_type,
                context=from_ctx,
                ui_meta=from_ui_meta,
            )
            summary["transitions"] += 1
            if edge_res["created"]:
                summary["edges_created"] += 1
            else:
                summary["edges_updated"] += 1

        self._touch_meta(run_increment=1)
        self._save_graph()
        return summary

    def infer_current_node(self, current_hierarchy_xml: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Infer the most likely node candidates for current UI XML."""
        if not current_hierarchy_xml or not self.graph.get("nodes"):
            return []
        obs = self._build_state_observation(
            hierarchy_xml=current_hierarchy_xml,
            xml_file=None,
            screenshot_file=None,
            is_terminal=False,
            terminal_reason=None,
        )

        strict_ids = list(self.strict_index.get(obs["strict_signature"], []))
        if strict_ids:
            ranked = sorted(
                (self.node_by_id[node_id] for node_id in strict_ids if node_id in self.node_by_id),
                key=lambda n: n.get("visit_count", 0),
                reverse=True,
            )
            return [
                {
                    "node_id": n["node_id"],
                    "score": 1.0,
                    "match_type": "strict",
                    "node_type": n.get("type", "main"),
                    "visit_count": n.get("visit_count", 0),
                    "confidence": n.get("confidence", 0.0),
                }
                for n in ranked[: max(1, top_k)]
            ]

        candidates = self._candidate_nodes_for_observation(obs)
        scored = []
        for node in candidates:
            score = self._fuzzy_similarity(obs, node, ui_meta=None)
            scored.append((node, score))
        scored.sort(key=lambda x: x[1], reverse=True)

        out = []
        for node, score in scored[: max(1, top_k)]:
            out.append(
                {
                    "node_id": node["node_id"],
                    "score": round(float(score), 4),
                    "match_type": "fuzzy",
                    "node_type": node.get("type", "main"),
                    "visit_count": node.get("visit_count", 0),
                    "confidence": node.get("confidence", 0.0),
                }
            )
        return out

    def predict_next(
        self,
        node_id: Optional[str],
        action_sig: Optional[str] = None,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Predict next likely states from a known node."""
        if not node_id or node_id not in self.node_by_id:
            return []
        edges = []
        for edge in self.graph.get("edges", []):
            if edge.get("from") != node_id:
                continue
            if action_sig and edge.get("action_sig") != action_sig:
                continue
            to_node = self.node_by_id.get(edge.get("to"))
            payload = self._edge_primary_payload(edge)
            edges.append(
                {
                    "edge_id": edge.get("edge_id"),
                    "to_node_id": edge.get("to"),
                    "to_node_type": to_node.get("type", "main") if to_node else "main",
                    "action_sig": edge.get("action_sig", ""),
                    "action_type": edge.get("action_type", ""),
                    "transition_type": edge.get("transition_type", "normal"),
                    "probability": round(float(edge.get("probability", 0.0)), 4),
                    "count": int(edge.get("count", 0) or 0),
                    "confidence": round(float(edge.get("confidence", 0.0)), 4),
                    "action_payload": payload if isinstance(payload, dict) else None,
                }
            )
        edges.sort(key=lambda x: (x["probability"], x["count"]), reverse=True)
        return edges[: max(1, top_k)]

    def detect_interruption(self, current_hierarchy_xml: str) -> Dict[str, Any]:
        """Detect whether current page is likely a known popup/interruption."""
        obs = self._build_state_observation(
            hierarchy_xml=current_hierarchy_xml,
            xml_file=None,
            screenshot_file=None,
            is_terminal=False,
            terminal_reason=None,
        )
        infer = self.infer_current_node(current_hierarchy_xml, top_k=1)
        if infer:
            top = infer[0]
            if top["score"] >= 0.70:
                node = self.node_by_id.get(top["node_id"])
                node_type = (node or {}).get("type", "main")
                if node_type in {"overlay_popup", "interruption"}:
                    return {
                        "is_interruption": True,
                        "source": "graph",
                        "node_id": top["node_id"],
                        "node_type": node_type,
                        "score": top["score"],
                    }

        if self._looks_interruption_obs(obs):
            return {
                "is_interruption": True,
                "source": "heuristic",
                "node_id": None,
                "node_type": "interruption",
                "score": 0.6,
            }
        if self._looks_popup_obs(obs):
            return {
                "is_interruption": True,
                "source": "heuristic",
                "node_id": None,
                "node_type": "overlay_popup",
                "score": 0.55,
            }
        return {
            "is_interruption": False,
            "source": "none",
            "node_id": None,
            "node_type": "main",
            "score": 0.0,
        }

    def suggest_low_confidence_frontier(self, limit: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """Suggest low-confidence regions to prioritize exploration."""
        nodes = []
        for node in self.graph.get("nodes", []):
            conf = float(node.get("confidence", 0.0) or 0.0)
            if node.get("type") == "terminal":
                continue
            if conf < self.NODE_LOW_CONFIDENCE:
                nodes.append(
                    {
                        "node_id": node.get("node_id"),
                        "node_type": node.get("type", "main"),
                        "confidence": round(conf, 4),
                        "visit_count": int(node.get("visit_count", 0) or 0),
                    }
                )
        nodes.sort(key=lambda x: (x["confidence"], -x["visit_count"]))

        edges = []
        for edge in self.graph.get("edges", []):
            conf = float(edge.get("confidence", 0.0) or 0.0)
            prob = float(edge.get("probability", 0.0) or 0.0)
            branch_uncertain = 0.2 <= prob <= 0.8
            if conf < self.EDGE_LOW_CONFIDENCE or branch_uncertain:
                edges.append(
                    {
                        "edge_id": edge.get("edge_id"),
                        "from": edge.get("from"),
                        "to": edge.get("to"),
                        "action_sig": edge.get("action_sig", ""),
                        "transition_type": edge.get("transition_type", "normal"),
                        "confidence": round(conf, 4),
                        "probability": round(prob, 4),
                        "count": int(edge.get("count", 0) or 0),
                    }
                )
        edges.sort(key=lambda x: (x["confidence"], x["count"]))

        return {
            "nodes": nodes[: max(1, limit)],
            "edges": edges[: max(1, limit)],
        }

    def get_root_nodes(self) -> List[Dict[str, Any]]:
        """Return root nodes (in-degree = 0), excluding terminal nodes."""
        if not self.graph.get("nodes"):
            return []
        indegree: Dict[str, int] = {
            str(node.get("node_id")): 0 for node in self.graph.get("nodes", []) if node.get("node_id")
        }
        for edge in self.graph.get("edges", []):
            to_id = edge.get("to")
            if to_id in indegree:
                indegree[to_id] = int(indegree.get(to_id, 0) or 0) + 1
        roots = []
        for node in self.graph.get("nodes", []):
            node_id = node.get("node_id")
            if not node_id:
                continue
            if node.get("type") == "terminal":
                continue
            in_deg = int(indegree.get(node_id, 0) or 0)
            if in_deg != 0:
                continue
            roots.append(
                {
                    "node_id": node_id,
                    "node_type": node.get("type", "main"),
                    "visit_count": int(node.get("visit_count", 0) or 0),
                    "confidence": round(float(node.get("confidence", 0.0) or 0.0), 4),
                    "in_degree": in_deg,
                }
            )
        roots.sort(key=lambda x: (x["visit_count"], x["confidence"]), reverse=True)
        return roots

    def search_replay_actions(
        self,
        current_hierarchy_xml: str,
        mode: str = SEARCH_MODE_DEFAULT,
    ) -> List[Dict[str, Any]]:
        """
        Search replayable action candidates from current XML without models.
        Returns sorted candidates with action payload and edge metadata.
        """
        if not current_hierarchy_xml or not self.graph.get("nodes"):
            return []

        mode_norm = str(mode or self.SEARCH_MODE_DEFAULT).strip().lower()
        # Keep behavior conservative by default to reduce false replay.
        if mode_norm != "conservative":
            mode_norm = "conservative"

        node_score_min = self.SEARCH_CONSERVATIVE_NODE_SCORE_MIN
        node_conf_min = self.SEARCH_CONSERVATIVE_NODE_CONFIDENCE_MIN
        edge_conf_min = self.SEARCH_CONSERVATIVE_EDGE_CONFIDENCE_MIN
        edge_prob_min = self.SEARCH_CONSERVATIVE_EDGE_PROBABILITY_MIN

        node_hits = self.infer_current_node(
            current_hierarchy_xml,
            top_k=max(1, self.SEARCH_CANDIDATE_NODE_TOP_K),
        )

        selected_nodes: List[Tuple[Dict[str, Any], float, str]] = []
        for hit in node_hits:
            node_id = hit.get("node_id")
            if not node_id or node_id not in self.node_by_id:
                continue
            match_type = str(hit.get("match_type", "fuzzy"))
            hit_score = float(hit.get("score", 0.0) or 0.0)
            node_conf = float(hit.get("confidence", 0.0) or 0.0)
            if match_type == "strict":
                selected_nodes.append((hit, 1.0, "current"))
                continue
            if hit_score >= node_score_min and node_conf >= node_conf_min:
                selected_nodes.append((hit, hit_score, "current"))

        weak_match = len(selected_nodes) == 0
        if weak_match:
            roots = self.get_root_nodes()
            for root in roots[: max(1, self.SEARCH_CANDIDATE_NODE_TOP_K)]:
                selected_nodes.append((root, 0.58, "root_fallback"))

        candidates: List[Dict[str, Any]] = []
        seen: set = set()

        def _collect_from_node(hit: Dict[str, Any], node_score: float, source: str) -> None:
            node_id = str(hit.get("node_id", "")).strip()
            if not node_id:
                return
            for edge_info in self.predict_next(
                node_id=node_id,
                top_k=max(1, self.SEARCH_CANDIDATE_EDGE_TOP_K),
            ):
                edge_conf = float(edge_info.get("confidence", 0.0) or 0.0)
                edge_prob = float(edge_info.get("probability", 0.0) or 0.0)
                if edge_conf < edge_conf_min or edge_prob < edge_prob_min:
                    continue

                edge_id = edge_info.get("edge_id")
                full_edge = self._edge_by_id(edge_id)
                payload = self._edge_primary_payload(full_edge)
                if not payload:
                    continue
                action_type = str(payload.get("action_type", "")).lower()
                if action_type not in {"click", "click_input", "input", "swipe", "done"}:
                    continue

                dedup_key = (
                    str(edge_id or ""),
                    action_type,
                    str(payload.get("target", "")),
                    str(payload.get("text", "")),
                    str(payload.get("direction", "")),
                    source,
                )
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                # Conservative blend: prioritize node match certainty first, then edge stats.
                replay_score = 0.50 * float(node_score) + 0.30 * edge_conf + 0.20 * edge_prob
                candidates.append(
                    {
                        "source": source,
                        "mode": mode_norm,
                        "node_id": node_id,
                        "to_node_id": edge_info.get("to_node_id"),
                        "edge_id": edge_id,
                        "action_sig": edge_info.get("action_sig", ""),
                        "action_type": action_type,
                        "action_payload": payload,
                        "node_score": round(float(node_score), 4),
                        "edge_confidence": round(edge_conf, 4),
                        "edge_probability": round(edge_prob, 4),
                        "replay_score": round(float(replay_score), 4),
                    }
                )

        for hit, node_score, source in selected_nodes:
            _collect_from_node(hit, node_score=node_score, source=source)

        candidates.sort(
            key=lambda x: (
                float(x.get("replay_score", 0.0) or 0.0),
                float(x.get("edge_probability", 0.0) or 0.0),
                float(x.get("edge_confidence", 0.0) or 0.0),
            ),
            reverse=True,
        )
        return candidates[: max(1, self.SEARCH_MAX_ACTIONS)]

    # ----------------------------- core update logic -----------------------------

    def _build_transitions(
        self,
        data_dir: str,
        actions: List[Dict[str, Any]],
        stop_reason: Optional[str],
    ) -> List[Dict[str, Any]]:
        xml_cache: Dict[str, Optional[str]] = {}
        transitions: List[Dict[str, Any]] = []

        def _load_xml(xml_file: Optional[str]) -> Optional[str]:
            if not xml_file or not isinstance(xml_file, str):
                return None
            if xml_file in xml_cache:
                return xml_cache[xml_file]
            path = xml_file if os.path.isabs(xml_file) else os.path.join(data_dir, xml_file)
            if not os.path.exists(path):
                xml_cache[xml_file] = None
                return None
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception:
                text = None
            xml_cache[xml_file] = text
            return text

        for i, action in enumerate(actions):
            if not isinstance(action, dict):
                continue
            step = action.get("step")
            step_int = step if isinstance(step, int) else None
            from_xml_file = action.get("xml_file")
            from_xml = _load_xml(from_xml_file)
            from_obs = self._build_state_observation(
                hierarchy_xml=from_xml,
                xml_file=from_xml_file if isinstance(from_xml_file, str) else None,
                screenshot_file=action.get("screenshot_file"),
                is_terminal=False,
                terminal_reason=None,
                fallback_key=f"step:{step_int or i + 1}",
            )

            action_name = str(action.get("action", "")).lower()
            force_reason = action.get("forced_stop_reason")
            make_terminal = action_name == "done" or bool(force_reason)

            if make_terminal:
                terminal_reason = str(force_reason or stop_reason or action_name or "terminal")
                to_obs = self._build_state_observation(
                    hierarchy_xml=None,
                    xml_file=None,
                    screenshot_file=None,
                    is_terminal=True,
                    terminal_reason=terminal_reason,
                )
            elif i + 1 < len(actions):
                next_action = actions[i + 1]
                next_xml_file = next_action.get("xml_file")
                next_xml = _load_xml(next_xml_file)
                to_obs = self._build_state_observation(
                    hierarchy_xml=next_xml,
                    xml_file=next_xml_file if isinstance(next_xml_file, str) else None,
                    screenshot_file=next_action.get("screenshot_file"),
                    is_terminal=False,
                    terminal_reason=None,
                    fallback_key=f"step:{(step_int or i + 1) + 1}",
                )
            else:
                terminal_reason = str(stop_reason or "trace_end")
                to_obs = self._build_state_observation(
                    hierarchy_xml=None,
                    xml_file=None,
                    screenshot_file=None,
                    is_terminal=True,
                    terminal_reason=terminal_reason,
                )

            transitions.append(
                {
                    "step": step_int or (i + 1),
                    "action": action,
                    "from_obs": from_obs,
                    "to_obs": to_obs,
                }
            )
        return transitions

    def _resolve_or_create_node(
        self,
        obs: Dict[str, Any],
        context: Dict[str, Any],
        ui_meta: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        strict_sig = obs["strict_signature"]
        strict_hits = list(self.strict_index.get(strict_sig, []))
        if strict_hits:
            node_id = self._pick_best_node(strict_hits)
            node = self.node_by_id[node_id]
            self._update_node(node, obs, context, ui_meta=ui_meta, exact_match=True, uncertain=False)
            return {"node_id": node_id, "created": False, "match_type": "strict", "score": 1.0}

        best_node = None
        best_score = 0.0
        for node in self._candidate_nodes_for_observation(obs):
            score = self._fuzzy_similarity(obs, node, ui_meta=ui_meta)
            if score > best_score:
                best_score = score
                best_node = node

        if best_node is not None and best_score >= self.MID_MATCH_THRESHOLD:
            obs_popup = self._looks_popup_obs(obs)
            obs_interrupt = self._looks_interruption_obs(obs)
            best_type = best_node.get("type", "main")
            # Keep popup/interruption states as explicit side branches instead of
            # merging into a dominant main-chain node.
            if (
                obs_popup
                and best_type == "main"
                and best_score < 0.97
            ):
                best_node = None
            elif (
                obs_interrupt
                and best_type == "main"
                and best_score < 0.97
            ):
                best_node = None
            elif (
                (not obs_popup and not obs_interrupt)
                and best_type in {"overlay_popup", "interruption"}
                and best_score < 0.97
            ):
                best_node = None

        if best_node is not None and best_score >= self.MID_MATCH_THRESHOLD:
            uncertain = best_score < self.HIGH_MATCH_THRESHOLD
            self._update_node(
                best_node,
                obs,
                context,
                ui_meta=ui_meta,
                exact_match=False,
                uncertain=uncertain,
            )
            return {
                "node_id": best_node["node_id"],
                "created": False,
                "match_type": "soft" if uncertain else "high",
                "score": round(float(best_score), 4),
            }

        new_node = self._create_node(obs)
        self._update_node(new_node, obs, context, ui_meta=ui_meta, exact_match=True, uncertain=False)
        self.graph["nodes"].append(new_node)
        self.node_by_id[new_node["node_id"]] = new_node
        self._add_node_to_indices(new_node)
        return {"node_id": new_node["node_id"], "created": True, "match_type": "created", "score": 0.0}

    def _classify_transition(
        self,
        from_obs: Dict[str, Any],
        from_node: Dict[str, Any],
        to_obs: Dict[str, Any],
        to_node: Dict[str, Any],
        action: Dict[str, Any],
        latest_main_node_id: Optional[str],
    ) -> Tuple[str, str]:
        _ = action
        if bool(to_obs.get("is_terminal")):
            return "normal", "terminal"

        if self._looks_interruption_obs(to_obs):
            return "interrupt", "interruption"

        overlap = self._jaccard(
            set(from_obs.get("coarse_tokens", set())),
            set(to_obs.get("coarse_tokens", set())),
        )
        to_new_cnt = len(set(to_obs.get("coarse_tokens", set())) - set(from_obs.get("coarse_tokens", set())))
        popup_like = overlap >= 0.72 and to_new_cnt <= 30 and self._looks_popup_obs(to_obs)
        if popup_like:
            return "popup_enter", "overlay_popup"

        if from_node.get("type") == "overlay_popup":
            if latest_main_node_id and to_node.get("node_id") == latest_main_node_id:
                return "popup_exit", "main"
            if to_node.get("type") == "main":
                return "popup_exit", "main"
            return "recovery", to_node.get("type", "main")

        if from_node.get("type") == "interruption" and to_node.get("type", "main") == "main":
            return "recovery", "main"

        return "normal", to_node.get("type", "main")

    def _upsert_edge(
        self,
        from_node_id: str,
        to_node_id: str,
        action: Dict[str, Any],
        transition_type: str,
        context: Dict[str, Any],
        ui_meta: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        action_sig = self._action_signature(action)
        key = (from_node_id, action_sig, to_node_id)
        edge = self.edge_by_key.get(key)

        if edge is None:
            edge = {
                "edge_id": self._new_edge_id(),
                "from": from_node_id,
                "to": to_node_id,
                "action_sig": action_sig,
                "action_type": str(action.get("action", "")).lower(),
                "count": 0,
                "confidence": 0.0,
                "probability": 0.0,
                "transition_type": transition_type,
                "transition_type_counts": {},
                "context_counts": {
                    "task_hash": {},
                    "replayed": {},
                    "action_name": {},
                    "stop_reason": {},
                },
                # Replay payloads are incrementally learned from historical actions.
                "action_payload": {},
                "action_payload_counts": {},
            }
            self.graph["edges"].append(edge)
            self.edge_by_key[key] = edge
            created = True
        else:
            created = False

        edge["count"] = int(edge.get("count", 0) or 0) + 1
        self._bump_nested_counter(edge["transition_type_counts"], transition_type, 1)
        edge["transition_type"] = self._pick_top_label(edge["transition_type_counts"], default=transition_type)

        self._bump_nested_counter(edge["context_counts"]["task_hash"], context.get("task_hash"), 1)
        self._bump_nested_counter(edge["context_counts"]["replayed"], str(bool(context.get("replayed"))), 1)
        self._bump_nested_counter(edge["context_counts"]["action_name"], context.get("action_name"), 1)
        self._bump_nested_counter(edge["context_counts"]["stop_reason"], context.get("stop_reason"), 1)

        payload = self._build_replay_payload(action=action, ui_meta=ui_meta)
        self._update_edge_payload(edge, payload)

        self._refresh_edge_group_confidence(from_node_id, action_sig)
        return {"created": created, "edge_id": edge["edge_id"]}

    # ----------------------------- node feature logic -----------------------------

    def _create_node(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        node_type = "terminal" if obs.get("is_terminal") else "main"
        return {
            "node_id": self._new_node_id(),
            "type": node_type,
            "visit_count": 0,
            "confidence": 0.0,
            "exact_match_count": 0,
            "fuzzy_match_count": 0,
            "uncertain_match_count": 0,
            "signatures": {
                "strict": [],
                "coarse": [],
            },
            "context_counts": {
                "task_hash": {},
                "task_tokens": {},
                "action_name": {},
                "replayed": {},
                "stop_reason": {},
            },
            "feature_stats": {
                "coarse_token_counts": {},
                "package_counts": {},
                "keyword_counts": {},
                "ui_meta_counts": {},
            },
            "examples": [],
        }

    def _update_node(
        self,
        node: Dict[str, Any],
        obs: Dict[str, Any],
        context: Dict[str, Any],
        ui_meta: Optional[Dict[str, Any]],
        exact_match: bool,
        uncertain: bool,
    ) -> None:
        node["visit_count"] = int(node.get("visit_count", 0) or 0) + 1
        if exact_match:
            node["exact_match_count"] = int(node.get("exact_match_count", 0) or 0) + 1
        elif uncertain:
            node["uncertain_match_count"] = int(node.get("uncertain_match_count", 0) or 0) + 1
        else:
            node["fuzzy_match_count"] = int(node.get("fuzzy_match_count", 0) or 0) + 1

        self._append_unique_limited(node["signatures"]["strict"], obs["strict_signature"], self.MAX_SIGNATURES_PER_KIND)
        self._append_unique_limited(node["signatures"]["coarse"], obs["coarse_signature"], self.MAX_SIGNATURES_PER_KIND)

        self._bump_nested_counter(node["context_counts"]["task_hash"], context.get("task_hash"), 1)
        self._bump_nested_counter(node["context_counts"]["action_name"], context.get("action_name"), 1)
        self._bump_nested_counter(node["context_counts"]["replayed"], str(bool(context.get("replayed"))), 1)
        self._bump_nested_counter(node["context_counts"]["stop_reason"], context.get("stop_reason"), 1)
        for token in context.get("task_tokens", []):
            self._bump_nested_counter(node["context_counts"]["task_tokens"], token, 1)

        for token in obs.get("coarse_tokens", set()):
            self._bump_nested_counter(node["feature_stats"]["coarse_token_counts"], token, 1)
        for pkg, cnt in obs.get("package_counts", {}).items():
            self._bump_nested_counter(node["feature_stats"]["package_counts"], pkg, int(cnt))
        for kw, cnt in obs.get("keyword_counts", {}).items():
            self._bump_nested_counter(node["feature_stats"]["keyword_counts"], kw, int(cnt))

        if isinstance(ui_meta, dict):
            ui_key = self._ui_meta_key(ui_meta)
            if ui_key:
                self._bump_nested_counter(node["feature_stats"]["ui_meta_counts"], ui_key, 1)

        self._truncate_counter(node["feature_stats"]["coarse_token_counts"], self.MAX_TOKEN_FEATURES)
        self._truncate_counter(node["feature_stats"]["keyword_counts"], self.MAX_TOKEN_FEATURES)

        example = {
            "step_idx": context.get("step_idx"),
            "xml_file": context.get("xml_file"),
            "screenshot_file": context.get("screenshot_file"),
            "task_hash": context.get("task_hash"),
        }
        self._append_example(node["examples"], example, self.MAX_EXAMPLES_PER_NODE)

        if obs.get("is_terminal"):
            self._promote_node_type(node, "terminal")
        elif self._looks_interruption_obs(obs):
            self._promote_node_type(node, "interruption")
        elif self._looks_popup_obs(obs):
            if node.get("type") == "main":
                self._promote_node_type(node, "overlay_popup")

        node["confidence"] = round(self._calc_node_confidence(node), 4)

    def _calc_node_confidence(self, node: Dict[str, Any]) -> float:
        visit = float(node.get("visit_count", 0) or 0)
        exact = float(node.get("exact_match_count", 0) or 0)
        uncertain = float(node.get("uncertain_match_count", 0) or 0)
        exact_ratio = exact / max(visit, 1.0)
        uncertain_ratio = uncertain / max(visit, 1.0)

        score = 0.35 + 0.45 * min(visit / 10.0, 1.0) + 0.20 * exact_ratio - 0.15 * uncertain_ratio
        return max(0.05, min(0.99, score))

    # ----------------------------- matching helpers -----------------------------

    def _candidate_nodes_for_observation(self, obs: Dict[str, Any]) -> List[Dict[str, Any]]:
        coarse_sig = obs.get("coarse_signature")
        candidate_ids = list(self.coarse_index.get(coarse_sig, []))
        if candidate_ids:
            return [self.node_by_id[nid] for nid in candidate_ids if nid in self.node_by_id]
        return [n for n in self.graph.get("nodes", []) if n.get("type") != "terminal"]

    def _fuzzy_similarity(
        self,
        obs: Dict[str, Any],
        node: Dict[str, Any],
        ui_meta: Optional[Dict[str, Any]],
    ) -> float:
        obs_tokens = set(obs.get("coarse_tokens", set()))
        node_tokens = set((node.get("feature_stats", {}).get("coarse_token_counts", {}) or {}).keys())
        jacc = self._jaccard(obs_tokens, node_tokens)

        obs_pkg = obs.get("package_counts", {}) or {}
        node_pkg = node.get("feature_stats", {}).get("package_counts", {}) or {}
        pkg_overlap = self._counter_overlap(obs_pkg, node_pkg)

        ui_score = 0.0
        if isinstance(ui_meta, dict):
            ui_key = self._ui_meta_key(ui_meta)
            if ui_key and ui_key in (node.get("feature_stats", {}).get("ui_meta_counts", {}) or {}):
                ui_score = 1.0

        score = 0.65 * jacc + 0.25 * pkg_overlap + 0.10 * ui_score
        return max(0.0, min(1.0, score))

    # ----------------------------- observation extraction -----------------------------

    def _build_state_observation(
        self,
        hierarchy_xml: Optional[str],
        xml_file: Optional[str],
        screenshot_file: Optional[str],
        is_terminal: bool,
        terminal_reason: Optional[str],
        fallback_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        if is_terminal:
            reason = str(terminal_reason or "terminal")
            sig = f"TERMINAL::{reason}"
            return {
                "is_terminal": True,
                "strict_signature": sig,
                "coarse_signature": sig,
                "coarse_tokens": {sig},
                "package_counts": {},
                "keyword_counts": {},
                "xml_file": xml_file,
                "screenshot_file": screenshot_file,
            }

        if not hierarchy_xml:
            placeholder = f"UNKNOWN::{fallback_key or 'missing_xml'}"
            return {
                "is_terminal": False,
                "strict_signature": placeholder,
                "coarse_signature": placeholder,
                "coarse_tokens": {placeholder},
                "package_counts": {},
                "keyword_counts": {},
                "xml_file": xml_file,
                "screenshot_file": screenshot_file,
            }

        parsed = self._extract_xml_features(hierarchy_xml)
        return {
            "is_terminal": False,
            "strict_signature": parsed["strict_signature"],
            "coarse_signature": parsed["coarse_signature"],
            "coarse_tokens": parsed["coarse_tokens"],
            "package_counts": parsed["package_counts"],
            "keyword_counts": parsed["keyword_counts"],
            "xml_file": xml_file,
            "screenshot_file": screenshot_file,
        }

    def _extract_xml_features(self, hierarchy_xml: str) -> Dict[str, Any]:
        strict_tokens: List[str] = []
        coarse_tokens: List[str] = []
        package_counts: Dict[str, int] = {}
        keyword_counts: Dict[str, int] = {}

        try:
            root = ET.fromstring(hierarchy_xml)
        except Exception:
            fallback = hashlib.md5(hierarchy_xml.strip().encode("utf-8", errors="ignore")).hexdigest()
            return {
                "strict_signature": fallback,
                "coarse_signature": fallback,
                "coarse_tokens": {fallback},
                "package_counts": {},
                "keyword_counts": {},
            }

        for elem in root.iter():
            attrib = elem.attrib or {}
            if not attrib:
                continue
            cls = attrib.get("class", "")
            rid = attrib.get("resource-id", "")
            text = attrib.get("text", "")
            cdesc = attrib.get("content-desc", "")
            bounds = attrib.get("bounds", "")
            pkg = attrib.get("package", "")

            strict_token = "|".join([cls, rid, text, cdesc, bounds, pkg])
            coarse_token = "|".join([cls, rid, text, cdesc, pkg])
            if strict_token.strip("|"):
                strict_tokens.append(strict_token)
            if coarse_token.strip("|"):
                coarse_tokens.append(coarse_token)

            if pkg:
                package_counts[pkg] = int(package_counts.get(pkg, 0) or 0) + 1

            blob = " ".join([rid, text, cdesc]).lower()
            for kw in self.POPUP_KEYWORDS + self.INTERRUPTION_KEYWORDS:
                if kw and kw in blob:
                    keyword_counts[kw] = int(keyword_counts.get(kw, 0) or 0) + 1

        strict_payload = "\n".join(strict_tokens) if strict_tokens else hierarchy_xml.strip()
        strict_sig = hashlib.md5(strict_payload.encode("utf-8", errors="ignore")).hexdigest()
        coarse_payload = "\n".join(sorted(set(coarse_tokens))) if coarse_tokens else strict_payload
        coarse_sig = hashlib.md5(coarse_payload.encode("utf-8", errors="ignore")).hexdigest()

        return {
            "strict_signature": strict_sig,
            "coarse_signature": coarse_sig,
            "coarse_tokens": set(coarse_tokens) if coarse_tokens else {coarse_sig},
            "package_counts": package_counts,
            "keyword_counts": keyword_counts,
        }

    # ----------------------------- graph persistence -----------------------------

    def _load_or_init_graph(self) -> Dict[str, Any]:
        if os.path.exists(self.graph_file):
            try:
                with open(self.graph_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    loaded.setdefault("meta", {})
                    loaded.setdefault("nodes", [])
                    loaded.setdefault("edges", [])
                    loaded["meta"].setdefault("schema_version", self.SCHEMA_VERSION)
                    loaded["meta"].setdefault("app_name", self.app_name)
                    loaded["meta"].setdefault("run_count", 0)
                    loaded["meta"].setdefault("node_seq", 0)
                    loaded["meta"].setdefault("edge_seq", 0)
                    loaded["meta"].setdefault("updated_at", "")
                    return loaded
            except Exception:
                pass
        return {
            "meta": {
                "schema_version": self.SCHEMA_VERSION,
                "app_name": self.app_name,
                "run_count": 0,
                "node_seq": 0,
                "edge_seq": 0,
                "updated_at": "",
            },
            "nodes": [],
            "edges": [],
        }

    def _save_graph(self) -> None:
        tmp = f"{self.graph_file}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.graph, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.graph_file)

    def _touch_meta(self, run_increment: int = 0) -> None:
        meta = self.graph.setdefault("meta", {})
        meta["schema_version"] = self.SCHEMA_VERSION
        meta["app_name"] = self.app_name
        meta["run_count"] = int(meta.get("run_count", 0) or 0) + int(run_increment)
        meta["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    def _rebuild_indices(self) -> None:
        self.node_by_id: Dict[str, Dict[str, Any]] = {}
        self.edge_by_key: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self.strict_index: Dict[str, List[str]] = defaultdict(list)
        self.coarse_index: Dict[str, List[str]] = defaultdict(list)

        for node in self.graph.get("nodes", []):
            self._normalize_node_shape(node)
            node_id = node.get("node_id")
            if not node_id:
                continue
            self.node_by_id[node_id] = node
            for sig in node.get("signatures", {}).get("strict", []):
                self.strict_index[str(sig)].append(node_id)
            for sig in node.get("signatures", {}).get("coarse", []):
                self.coarse_index[str(sig)].append(node_id)

        for edge in self.graph.get("edges", []):
            self._normalize_edge_shape(edge)
            key = (edge.get("from"), edge.get("action_sig"), edge.get("to"))
            if all(key):
                self.edge_by_key[key] = edge

    # ----------------------------- utility helpers -----------------------------

    def _normalize_actions(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized = []
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

    def _build_context(
        self,
        task_hash: str,
        task_tokens: List[str],
        action_name: str,
        replayed: bool,
        stop_reason: Optional[str],
        step_idx: Optional[int],
        xml_file: Optional[str],
        screenshot_file: Optional[str],
    ) -> Dict[str, Any]:
        return {
            "task_hash": task_hash,
            "task_tokens": task_tokens,
            "action_name": action_name or "",
            "replayed": bool(replayed),
            "stop_reason": str(stop_reason or ""),
            "step_idx": step_idx,
            "xml_file": xml_file,
            "screenshot_file": screenshot_file,
        }

    def _extract_task_tokens(self, task_desc: str) -> List[str]:
        if not task_desc:
            return []
        text = task_desc.lower().strip()
        text = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", text)
        tokens = re.findall(r"[0-9a-z]+|[\u4e00-\u9fff]", text)
        cjk = "".join(re.findall(r"[\u4e00-\u9fff]", text))
        if len(cjk) > 1:
            tokens.extend([cjk[i : i + 2] for i in range(len(cjk) - 1)])
        return tokens[:64]

    def _new_node_id(self) -> str:
        meta = self.graph["meta"]
        meta["node_seq"] = int(meta.get("node_seq", 0) or 0) + 1
        return f"n{meta['node_seq']}"

    def _new_edge_id(self) -> str:
        meta = self.graph["meta"]
        meta["edge_seq"] = int(meta.get("edge_seq", 0) or 0) + 1
        return f"e{meta['edge_seq']}"

    def _add_node_to_indices(self, node: Dict[str, Any]) -> None:
        node_id = node["node_id"]
        for sig in node.get("signatures", {}).get("strict", []):
            self.strict_index[str(sig)].append(node_id)
        for sig in node.get("signatures", {}).get("coarse", []):
            self.coarse_index[str(sig)].append(node_id)

    def _normalize_node_shape(self, node: Dict[str, Any]) -> None:
        node.setdefault("type", "main")
        node.setdefault("visit_count", 0)
        node.setdefault("confidence", 0.0)
        node.setdefault("exact_match_count", 0)
        node.setdefault("fuzzy_match_count", 0)
        node.setdefault("uncertain_match_count", 0)
        node.setdefault("signatures", {"strict": [], "coarse": []})
        node["signatures"].setdefault("strict", [])
        node["signatures"].setdefault("coarse", [])
        node.setdefault(
            "context_counts",
            {"task_hash": {}, "task_tokens": {}, "action_name": {}, "replayed": {}, "stop_reason": {}},
        )
        node.setdefault(
            "feature_stats",
            {"coarse_token_counts": {}, "package_counts": {}, "keyword_counts": {}, "ui_meta_counts": {}},
        )
        node.setdefault("examples", [])

    def _normalize_edge_shape(self, edge: Dict[str, Any]) -> None:
        edge.setdefault("action_sig", "")
        edge.setdefault("action_type", "")
        edge.setdefault("count", 0)
        edge.setdefault("confidence", 0.0)
        edge.setdefault("probability", 0.0)
        edge.setdefault("transition_type", "normal")
        edge.setdefault("transition_type_counts", {})
        edge.setdefault("context_counts", {"task_hash": {}, "replayed": {}, "action_name": {}, "stop_reason": {}})
        edge.setdefault("action_payload", {})
        edge.setdefault("action_payload_counts", {})

    def _build_replay_payload(
        self,
        action: Dict[str, Any],
        ui_meta: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "action_type": str(action.get("action", "")).lower(),
            "target": str(action.get("target", "")),
            "text": str(action.get("text", "")),
            "direction": str(action.get("direction", "")).lower(),
        }
        position = action.get("position")
        if (
            isinstance(position, list)
            and len(position) == 2
            and all(isinstance(v, (int, float)) for v in position)
        ):
            payload["position"] = [int(round(float(position[0]))), int(round(float(position[1])))]
        else:
            payload["position"] = None

        if isinstance(ui_meta, dict):
            payload["ui_meta"] = self._normalize_ui_meta(ui_meta)
        else:
            payload["ui_meta"] = None
        return payload

    def _normalize_ui_meta(self, ui_meta: Dict[str, Any]) -> Dict[str, str]:
        keys = [
            "resource-id",
            "text",
            "content-desc",
            "class",
            "package",
            "clickable",
            "enabled",
            "selected",
            "checked",
            "focused",
            "scrollable",
            "long-clickable",
            "bounds",
        ]
        out: Dict[str, str] = {}
        for key in keys:
            out[key] = str(ui_meta.get(key, ""))
        return out

    def _payload_key(self, payload: Dict[str, Any]) -> str:
        try:
            return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        except Exception:
            return ""

    def _update_edge_payload(self, edge: Dict[str, Any], payload: Dict[str, Any]) -> None:
        action_type = str(payload.get("action_type", "")).lower()
        if not action_type:
            return
        payload_key = self._payload_key(payload)
        if not payload_key:
            return
        counts = edge.setdefault("action_payload_counts", {})
        counts[payload_key] = int(counts.get(payload_key, 0) or 0) + 1

        current_payload = edge.get("action_payload")
        current_count = 0
        if isinstance(current_payload, dict):
            current_key = self._payload_key(current_payload)
            current_count = int(counts.get(current_key, 0) or 0) if current_key else 0
        new_count = int(counts.get(payload_key, 0) or 0)
        if (not isinstance(current_payload, dict)) or (new_count >= current_count):
            edge["action_payload"] = payload

    def _edge_primary_payload(self, edge: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not isinstance(edge, dict):
            return None
        payload = edge.get("action_payload")
        if isinstance(payload, dict) and payload.get("action_type"):
            return payload

        # Backward compatibility for legacy graphs that only stored counts.
        counts = edge.get("action_payload_counts", {})
        if not isinstance(counts, dict) or not counts:
            return None
        top_items = sorted(counts.items(), key=lambda kv: int(kv[1] or 0), reverse=True)
        for payload_key, _ in top_items:
            if not isinstance(payload_key, str):
                continue
            try:
                parsed = json.loads(payload_key)
            except Exception:
                continue
            if isinstance(parsed, dict) and parsed.get("action_type"):
                edge["action_payload"] = parsed
                return parsed
        return None

    def _edge_by_id(self, edge_id: Any) -> Optional[Dict[str, Any]]:
        if not edge_id:
            return None
        for edge in self.graph.get("edges", []):
            if edge.get("edge_id") == edge_id:
                return edge
        return None

    def _summary_key_for_match(self, match_type: str) -> str:
        if match_type == "strict":
            return "node_matched_strict"
        if match_type == "high":
            return "node_matched_high"
        if match_type == "soft":
            return "node_matched_soft"
        return "node_matched_high"

    def _pick_best_node(self, node_ids: List[str]) -> str:
        ranked = sorted(
            (self.node_by_id[nid] for nid in node_ids if nid in self.node_by_id),
            key=lambda n: n.get("visit_count", 0),
            reverse=True,
        )
        return ranked[0]["node_id"]

    def _refresh_edge_group_confidence(self, from_node_id: str, action_sig: str) -> None:
        group = [
            e
            for e in self.graph.get("edges", [])
            if e.get("from") == from_node_id and e.get("action_sig") == action_sig
        ]
        total = sum(int(e.get("count", 0) or 0) for e in group)
        if total <= 0:
            return
        for edge in group:
            count = int(edge.get("count", 0) or 0)
            prob = count / float(total)
            edge["probability"] = round(prob, 6)
            score = 0.30 + 0.40 * min(count / 8.0, 1.0) + 0.30 * prob
            edge["confidence"] = round(max(0.05, min(0.99, score)), 6)

    def _looks_popup_obs(self, obs: Dict[str, Any]) -> bool:
        keyword_counts = obs.get("keyword_counts", {}) or {}
        return any(k in keyword_counts for k in self.POPUP_KEYWORDS)

    def _looks_interruption_obs(self, obs: Dict[str, Any]) -> bool:
        keyword_counts = obs.get("keyword_counts", {}) or {}
        if any(k in keyword_counts for k in self.INTERRUPTION_KEYWORDS):
            return True
        packages = obs.get("package_counts", {}) or {}
        total = sum(int(v) for v in packages.values()) or 1
        systemui_cnt = int(packages.get("com.android.systemui", 0) or 0)
        return (systemui_cnt / float(total)) >= 0.3

    def _promote_node_type(self, node: Dict[str, Any], target_type: str) -> None:
        priority = {
            "main": 0,
            "overlay_popup": 1,
            "interruption": 2,
            "terminal": 3,
        }
        cur = node.get("type", "main")
        if priority.get(target_type, 0) >= priority.get(cur, 0):
            node["type"] = target_type

    def _action_signature(self, action: Dict[str, Any]) -> str:
        name = str(action.get("action", "")).lower()
        if name in {"click", "click_input"}:
            target = self._normalize_text(str(action.get("target", "")))
            pos = action.get("position", [])
            if isinstance(pos, list) and len(pos) == 2 and all(isinstance(v, (int, float)) for v in pos):
                x_bin = int(float(pos[0]) // 80)
                y_bin = int(float(pos[1]) // 80)
                if name == "click_input":
                    text = self._normalize_text(str(action.get("text", "")))
                    return f"{name}:{target}:{text}:{x_bin}:{y_bin}"
                return f"{name}:{target}:{x_bin}:{y_bin}"
            if name == "click_input":
                text = self._normalize_text(str(action.get("text", "")))
                return f"{name}:{target}:{text}"
            return f"{name}:{target}"
        if name == "input":
            return f"input:{self._normalize_text(str(action.get('text', '')))}"
        if name == "swipe":
            return f"swipe:{self._normalize_text(str(action.get('direction', '')))}"
        return name

    def _normalize_text(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _safe_name(self, text: str) -> str:
        norm = re.sub(r"[^0-9a-zA-Z._-]+", "_", text or "").strip("._-")
        if norm:
            return norm
        return f"app_{hashlib.md5((text or 'app').encode('utf-8')).hexdigest()[:8]}"

    def _counter_overlap(self, a: Dict[str, int], b: Dict[str, int]) -> float:
        if not a:
            return 0.0
        total = float(sum(max(0, int(v)) for v in a.values()) or 1.0)
        overlap = 0.0
        for key, va in a.items():
            vb = b.get(key, 0)
            overlap += min(max(0, int(va)), max(0, int(vb)))
        return overlap / total

    def _jaccard(self, a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / float(max(1, len(a | b)))

    def _bump_nested_counter(self, container: Dict[str, Any], key: Optional[str], delta: int) -> None:
        if not key:
            return
        container[key] = int(container.get(key, 0) or 0) + int(delta)

    def _truncate_counter(self, container: Dict[str, int], max_items: int) -> None:
        if len(container) <= max_items:
            return
        top = sorted(container.items(), key=lambda kv: kv[1], reverse=True)[:max_items]
        container.clear()
        for k, v in top:
            container[k] = v

    def _append_unique_limited(self, arr: List[str], value: str, limit: int) -> None:
        if not value:
            return
        if value in arr:
            return
        arr.append(value)
        if len(arr) > limit:
            del arr[0 : len(arr) - limit]

    def _append_example(self, arr: List[Dict[str, Any]], example: Dict[str, Any], limit: int) -> None:
        if not example:
            return
        sig = (example.get("xml_file"), example.get("step_idx"), example.get("task_hash"))
        for ex in arr:
            ex_sig = (ex.get("xml_file"), ex.get("step_idx"), ex.get("task_hash"))
            if sig == ex_sig:
                return
        arr.append(example)
        if len(arr) > limit:
            del arr[0 : len(arr) - limit]

    def _pick_top_label(self, counts: Dict[str, int], default: str) -> str:
        if not counts:
            return default
        return sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[0][0]

    def _ui_meta_key(self, ui_meta: Dict[str, Any]) -> str:
        rid = str(ui_meta.get("resource-id", "")).strip()
        cls = str(ui_meta.get("class", "")).strip()
        txt = str(ui_meta.get("text", "")).strip()
        payload = "|".join([rid, cls, txt])
        return payload.strip("|")
