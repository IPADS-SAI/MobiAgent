#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import sys
import unittest
import uuid

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

from template_graph_builder import TemplateGraphBuilder


def _build_xml(
    page_tag: str,
    package: str = "me.ele",
    mutate_indices=None,
    add_popup: bool = False,
    popup_systemui: bool = False,
    popup_text: str = "允许",
) -> str:
    mutate_indices = set(mutate_indices or [])
    base_nodes = []
    y = 120
    for idx in range(8):
        rid = f"{page_tag}_id_{idx}"
        txt = f"{page_tag}_text_{idx}"
        if idx in mutate_indices:
            txt = f"{page_tag}_mut_{idx}"
        base_nodes.append(
            (
                f'<node class="android.widget.TextView" package="{package}" '
                f'resource-id="{rid}" text="{txt}" content-desc="" '
                f'bounds="[40,{y}][980,{y + 80}]" clickable="false" enabled="true"/>'
            )
        )
        y += 100

    popup_nodes = ""
    if add_popup:
        popup_pkg = "com.android.systemui" if popup_systemui else package
        popup_nodes = (
            f'<node class="android.app.Dialog" package="{popup_pkg}" resource-id="popup_container" '
            f'text="" content-desc="" bounds="[120,500][960,1500]" clickable="false" enabled="true">'
            f'<node class="android.widget.TextView" package="{popup_pkg}" resource-id="popup_msg" '
            f'text="{popup_text}" content-desc="" bounds="[220,700][860,820]" clickable="false" enabled="true"/>'
            f'<node class="android.widget.Button" package="{popup_pkg}" resource-id="popup_close" '
            f'text="关闭" content-desc="" bounds="[420,1160][660,1260]" clickable="true" enabled="true"/>'
            "</node>"
        )

    body = "".join(base_nodes) + popup_nodes
    return (
        '<hierarchy rotation="0">'
        f'<node class="android.widget.FrameLayout" package="{package}" '
        'resource-id="root" text="" content-desc="" bounds="[0,0][1080,2400]" '
        'clickable="false" enabled="true">'
        f"{body}</node></hierarchy>"
    )


def _write_xml(path: str, xml: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(xml)


class TemplateGraphBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        base_tmp_dir = os.path.join(MODULE_DIR, "data", "_ut_tmp")
        os.makedirs(base_tmp_dir, exist_ok=True)
        self.tmp_root = os.path.join(base_tmp_dir, f"template_graph_tests_{uuid.uuid4().hex}")
        os.makedirs(self.tmp_root, exist_ok=True)
        self.graph_dir = os.path.join(self.tmp_root, "_template_graph")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_root, ignore_errors=True)

    def _new_builder(self) -> TemplateGraphBuilder:
        return TemplateGraphBuilder(graph_dir=self.graph_dir, app_name="meituan")

    def _prepare_run_dir(self, name: str) -> str:
        path = os.path.join(self.tmp_root, name)
        os.makedirs(path, exist_ok=True)
        return path

    def test_merge_same_state_across_runs(self) -> None:
        builder = self._new_builder()
        xml_a = _build_xml("A")
        xml_b = _build_xml("B")

        for idx in (1, 2):
            run_dir = self._prepare_run_dir(f"run_{idx}")
            _write_xml(os.path.join(run_dir, "01.xml"), xml_a)
            _write_xml(os.path.join(run_dir, "02.xml"), xml_b)
            actions = [
                {"step": 1, "action": "click", "target": "go", "position": [200, 900], "xml_file": "01.xml"},
                {"step": 2, "action": "done", "xml_file": "02.xml"},
            ]
            builder.update_from_run(
                data_dir=run_dir,
                actions=actions,
                task_desc="打开页面并结束",
                task_hash=f"task_{idx}",
                stop_reason="done",
            )

        self.assertEqual(3, len(builder.graph["nodes"]))  # A, B, terminal
        inferred = builder.infer_current_node(xml_a, top_k=1)
        self.assertTrue(inferred)
        self.assertGreaterEqual(inferred[0]["visit_count"], 2)

    def test_branching_probability(self) -> None:
        builder = self._new_builder()
        xml_a = _build_xml("A")
        xml_b = _build_xml("B")
        xml_c = _build_xml("C")

        run1 = self._prepare_run_dir("branch_1")
        _write_xml(os.path.join(run1, "01.xml"), xml_a)
        _write_xml(os.path.join(run1, "02.xml"), xml_b)
        actions1 = [
            {"step": 1, "action": "click", "target": "open", "position": [220, 920], "xml_file": "01.xml"},
            {"step": 2, "action": "done", "xml_file": "02.xml"},
        ]
        builder.update_from_run(run1, actions1, "任务1", "branch_1", stop_reason="done")

        run2 = self._prepare_run_dir("branch_2")
        _write_xml(os.path.join(run2, "01.xml"), xml_a)
        _write_xml(os.path.join(run2, "02.xml"), xml_c)
        actions2 = [
            {"step": 1, "action": "click", "target": "open", "position": [220, 920], "xml_file": "01.xml"},
            {"step": 2, "action": "done", "xml_file": "02.xml"},
        ]
        builder.update_from_run(run2, actions2, "任务2", "branch_2", stop_reason="done")

        a_node = builder.infer_current_node(xml_a, top_k=1)[0]["node_id"]
        candidates = [x for x in builder.predict_next(a_node, top_k=5) if x["action_type"] == "click"]
        self.assertGreaterEqual(len(candidates), 2)
        probs = sorted(round(c["probability"], 2) for c in candidates[:2])
        self.assertEqual([0.5, 0.5], probs)

    def test_popup_enter_and_exit_edges(self) -> None:
        builder = self._new_builder()
        xml_main = _build_xml("Main")
        xml_popup = _build_xml("Main", add_popup=True, popup_text="允许")

        run_dir = self._prepare_run_dir("popup")
        _write_xml(os.path.join(run_dir, "01.xml"), xml_main)
        _write_xml(os.path.join(run_dir, "02.xml"), xml_popup)
        _write_xml(os.path.join(run_dir, "03.xml"), xml_main)
        actions = [
            {"step": 1, "action": "click", "target": "open_popup", "position": [200, 900], "xml_file": "01.xml"},
            {"step": 2, "action": "click", "target": "close_popup", "position": [500, 1200], "xml_file": "02.xml"},
            {"step": 3, "action": "done", "xml_file": "03.xml"},
        ]
        builder.update_from_run(run_dir, actions, "弹窗任务", "popup_task", stop_reason="done")

        transition_types = {edge["transition_type"] for edge in builder.graph["edges"]}
        self.assertIn("popup_enter", transition_types)
        self.assertIn("popup_exit", transition_types)
        popup_nodes = [n for n in builder.graph["nodes"] if n.get("type") == "overlay_popup"]
        self.assertTrue(popup_nodes)

    def test_soft_match_updates_existing_node(self) -> None:
        builder = self._new_builder()
        xml_a = _build_xml("A")
        xml_b1 = _build_xml("B")
        xml_b2 = _build_xml("B", mutate_indices=[0, 1])

        run1 = self._prepare_run_dir("soft_1")
        _write_xml(os.path.join(run1, "01.xml"), xml_a)
        _write_xml(os.path.join(run1, "02.xml"), xml_b1)
        actions = [
            {"step": 1, "action": "click", "target": "next", "position": [220, 920], "xml_file": "01.xml"},
            {"step": 2, "action": "done", "xml_file": "02.xml"},
        ]
        builder.update_from_run(run1, actions, "软匹配任务1", "soft_1", stop_reason="done")

        run2 = self._prepare_run_dir("soft_2")
        _write_xml(os.path.join(run2, "01.xml"), xml_a)
        _write_xml(os.path.join(run2, "02.xml"), xml_b2)
        summary2 = builder.update_from_run(run2, actions, "软匹配任务2", "soft_2", stop_reason="done")

        self.assertGreaterEqual(summary2["node_matched_soft"], 1)
        self.assertLessEqual(len(builder.graph["nodes"]), 4)  # A, B, terminal (+optional unknown)

    def test_infer_and_predict(self) -> None:
        builder = self._new_builder()
        xml_a = _build_xml("A")
        xml_b = _build_xml("B")
        run_dir = self._prepare_run_dir("infer_predict")
        _write_xml(os.path.join(run_dir, "01.xml"), xml_a)
        _write_xml(os.path.join(run_dir, "02.xml"), xml_b)
        actions = [
            {"step": 1, "action": "click", "target": "go", "position": [300, 1000], "xml_file": "01.xml"},
            {"step": 2, "action": "done", "xml_file": "02.xml"},
        ]
        builder.update_from_run(run_dir, actions, "推断测试", "infer_1", stop_reason="done")

        a_hit = builder.infer_current_node(xml_a, top_k=1)
        b_hit = builder.infer_current_node(xml_b, top_k=1)
        self.assertTrue(a_hit and b_hit)
        next_candidates = builder.predict_next(a_hit[0]["node_id"], top_k=3)
        self.assertTrue(any(c["to_node_id"] == b_hit[0]["node_id"] for c in next_candidates))

    def test_low_confidence_frontier(self) -> None:
        builder = self._new_builder()
        xml_a = _build_xml("A")
        xml_b = _build_xml("B")
        run_dir = self._prepare_run_dir("frontier")
        _write_xml(os.path.join(run_dir, "01.xml"), xml_a)
        _write_xml(os.path.join(run_dir, "02.xml"), xml_b)
        actions = [
            {"step": 1, "action": "click", "target": "go", "position": [300, 1000], "xml_file": "01.xml"},
            {"step": 2, "action": "done", "xml_file": "02.xml"},
        ]
        builder.update_from_run(run_dir, actions, "前沿测试", "frontier_1", stop_reason="done")
        frontier = builder.suggest_low_confidence_frontier(limit=5)
        self.assertTrue(frontier["nodes"] or frontier["edges"])


    def test_search_replay_actions_returns_payload(self) -> None:
        builder = self._new_builder()
        xml_a = _build_xml("A")
        xml_b = _build_xml("B")
        run_dir = self._prepare_run_dir("search_payload")
        _write_xml(os.path.join(run_dir, "01.xml"), xml_a)
        _write_xml(os.path.join(run_dir, "02.xml"), xml_b)
        actions = [
            {
                "step": 1,
                "action": "click",
                "target": "go",
                "position": [300, 1000],
                "xml_file": "01.xml",
                "ui_meta": {
                    "resource-id": "A_id_1",
                    "class": "android.widget.TextView",
                    "package": "me.ele",
                    "text": "A_text_1",
                },
            },
            {"step": 2, "action": "done", "xml_file": "02.xml"},
        ]
        builder.update_from_run(run_dir, actions, "search replay", "search_1", stop_reason="done")

        candidates = builder.search_replay_actions(xml_a, mode="conservative")
        self.assertTrue(candidates)
        top = candidates[0]
        self.assertEqual("click", top["action_type"])
        self.assertIsInstance(top.get("action_payload"), dict)
        self.assertEqual("click", top["action_payload"].get("action_type"))
        self.assertIn(top.get("source"), {"current", "root_fallback"})

    def test_multi_roots_and_root_fallback_search(self) -> None:
        builder = self._new_builder()
        xml_a = _build_xml("A")
        xml_b = _build_xml("B")
        xml_c = _build_xml("C")
        xml_d = _build_xml("D")
        xml_unknown = _build_xml("UNKNOWN", mutate_indices=[0, 1, 2, 3, 4, 5, 6, 7])

        run1 = self._prepare_run_dir("root_1")
        _write_xml(os.path.join(run1, "01.xml"), xml_a)
        _write_xml(os.path.join(run1, "02.xml"), xml_b)
        actions1 = [
            {"step": 1, "action": "click", "target": "path1", "position": [220, 920], "xml_file": "01.xml"},
            {"step": 2, "action": "done", "xml_file": "02.xml"},
        ]
        builder.update_from_run(run1, actions1, "root path 1", "root_1", stop_reason="done")

        run2 = self._prepare_run_dir("root_2")
        _write_xml(os.path.join(run2, "01.xml"), xml_c)
        _write_xml(os.path.join(run2, "02.xml"), xml_d)
        actions2 = [
            {"step": 1, "action": "click", "target": "path2", "position": [260, 960], "xml_file": "01.xml"},
            {"step": 2, "action": "done", "xml_file": "02.xml"},
        ]
        builder.update_from_run(run2, actions2, "root path 2", "root_2", stop_reason="done")

        roots = builder.get_root_nodes()
        root_ids = {r["node_id"] for r in roots}
        self.assertGreaterEqual(len(root_ids), 2)

        candidates = builder.search_replay_actions(xml_unknown, mode="conservative")
        self.assertTrue(candidates)
        self.assertTrue(any(c.get("source") == "root_fallback" for c in candidates))

    def test_legacy_edge_without_payload_is_skipped(self) -> None:
        builder = self._new_builder()
        xml_a = _build_xml("A")
        xml_b = _build_xml("B")
        run_dir = self._prepare_run_dir("legacy_no_payload")
        _write_xml(os.path.join(run_dir, "01.xml"), xml_a)
        _write_xml(os.path.join(run_dir, "02.xml"), xml_b)
        actions = [
            {"step": 1, "action": "click", "target": "go", "position": [200, 900], "xml_file": "01.xml"},
            {"step": 2, "action": "done", "xml_file": "02.xml"},
        ]
        builder.update_from_run(run_dir, actions, "legacy graph", "legacy_1", stop_reason="done")

        for edge in builder.graph.get("edges", []):
            edge.pop("action_payload", None)
            edge.pop("action_payload_counts", None)
        builder._save_graph()

        reloaded = TemplateGraphBuilder(graph_dir=self.graph_dir, app_name="meituan")
        candidates = reloaded.search_replay_actions(xml_a, mode="conservative")
        self.assertEqual([], candidates)


if __name__ == "__main__":
    unittest.main()
