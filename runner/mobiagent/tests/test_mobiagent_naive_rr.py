#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import shutil
import sys
import types
import unittest
import uuid
from typing import List

from PIL import Image, ImageDraw

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

IMPORT_ERROR = None
rr = None
TemplateGraphBuilder = None
try:
    if "openai" not in sys.modules:
        openai_stub = types.ModuleType("openai")

        class _DummyOpenAI:  # pragma: no cover - test shim
            def __init__(self, *args, **kwargs):
                _ = args, kwargs

        openai_stub.OpenAI = _DummyOpenAI
        sys.modules["openai"] = openai_stub

    import mobiagent_naive_rr as rr  # type: ignore
    from template_graph_builder import TemplateGraphBuilder  # type: ignore
except Exception as e:  # pragma: no cover
    IMPORT_ERROR = e


def _basic_xml() -> str:
    return (
        '<hierarchy rotation="0">'
        '<node class="android.widget.FrameLayout" package="me.ele" '
        'resource-id="root" text="" content-desc="" bounds="[0,0][1080,2400]" '
        'clickable="false" enabled="true">'
        '<node class="android.widget.TextView" package="me.ele" resource-id="title" '
        'text="page" content-desc="" bounds="[10,10][500,80]" clickable="false" enabled="true"/>'
        "</node></hierarchy>"
    )


def _make_pattern_image(path: str, pattern: str, size: int = 256) -> None:
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    if pattern == "vertical":
        draw.rectangle([0, 0, size // 2 - 1, size - 1], fill="black")
    elif pattern == "horizontal":
        draw.rectangle([0, 0, size - 1, size // 2 - 1], fill="black")
    elif pattern == "diag":
        for i in range(0, size, 8):
            draw.line([(0, i), (i, 0)], fill="black", width=4)
    elif pattern == "checker":
        block = 32
        for y in range(0, size, block):
            for x in range(0, size, block):
                if ((x // block) + (y // block)) % 2 == 0:
                    draw.rectangle([x, y, x + block - 1, y + block - 1], fill="black")
    elif pattern == "center_box":
        draw.rectangle([size // 4, size // 4, size * 3 // 4, size * 3 // 4], fill="black")
    else:
        raise ValueError(f"unknown pattern: {pattern}")
    img.save(path, format="JPEG")


class _FakeDevice:
    def __init__(self, screenshot_paths: List[str], hierarchy_xml: str) -> None:
        self._screenshot_paths = list(screenshot_paths)
        self._hierarchy_xml = hierarchy_xml
        self.click_calls = []
        self.input_calls = []
        self.swipe_calls = []

    def screenshot(self, path=None):
        if not self._screenshot_paths:
            raise RuntimeError("no screenshot prepared")
        src = self._screenshot_paths.pop(0)
        if path is None:
            return src
        shutil.copy(src, path)
        return path

    def dump_hierarchy(self):
        return self._hierarchy_xml

    def click(self, x, y):
        self.click_calls.append((int(x), int(y)))

    def input_text(self, text):
        self.input_calls.append(str(text))
        return True

    def swipe(self, direction):
        self.swipe_calls.append(str(direction))


@unittest.skipIf(rr is None, f"mobiagent_naive_rr import failed: {IMPORT_ERROR}")
class MobiAgentNaiveRRToctouTests(unittest.TestCase):
    def setUp(self) -> None:
        base_tmp_dir = os.path.join(MODULE_DIR, "data", "_ut_tmp")
        os.makedirs(base_tmp_dir, exist_ok=True)
        self.root = os.path.join(base_tmp_dir, f"naive_rr_toctou_{uuid.uuid4().hex}")
        os.makedirs(self.root, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _img(self, name: str, pattern: str) -> str:
        path = os.path.join(self.root, name)
        _make_pattern_image(path, pattern=pattern)
        return path

    def _run_with_patches(self, device: _FakeDevice, decider_payloads: List[str]):
        origin_decider = rr.get_decider_action
        origin_grounder = rr.get_grounder_coords
        origin_sleep = rr.time.sleep
        try:
            queue = list(decider_payloads)

            def _fake_decider(task, history, screenshot_b64, model=""):
                if not queue:
                    raise RuntimeError("decider payload exhausted")
                return queue.pop(0)

            def _fake_grounder(target_element, screenshot_b64, model="", use_qwen3=True):
                return '{"bbox": [100, 100, 200, 200]}'

            rr.get_decider_action = _fake_decider
            rr.get_grounder_coords = _fake_grounder
            rr.time.sleep = lambda _: None

            data_dir = os.path.join(self.root, "run")
            os.makedirs(data_dir, exist_ok=True)
            return rr.execute_task_with_rr(
                app_name="meituan",
                task_desc="test task",
                device=device,
                data_dir=data_dir,
                decider_model="",
                grounder_model="",
                use_qwen3=True,
                no_compress=False,
                template_graph="off",
            )
        finally:
            rr.get_decider_action = origin_decider
            rr.get_grounder_coords = origin_grounder
            rr.time.sleep = origin_sleep

    def test_ahash_similarity_identical_images(self) -> None:
        a = self._img("a.jpg", "vertical")
        b = self._img("b.jpg", "vertical")
        score = rr.compute_image_ahash_similarity(a, b)
        self.assertIsNotNone(score)
        self.assertGreaterEqual(score, 0.99)

    def test_ahash_similarity_changed_images(self) -> None:
        a = self._img("a.jpg", "vertical")
        b = self._img("b.jpg", "horizontal")
        score = rr.compute_image_ahash_similarity(a, b)
        self.assertIsNotNone(score)
        self.assertLess(score, rr.TOCTOU_IMAGE_SIM_MIN)

    def test_toctou_pre_action_allow_and_reject(self) -> None:
        ref = self._img("ref.jpg", "checker")
        same = self._img("same.jpg", "checker")
        diff = self._img("diff.jpg", "diag")

        device_ok = _FakeDevice([same], _basic_xml())
        ok = rr.evaluate_toctou_pre_action(
            device=device_ok,
            data_dir=self.root,
            step=1,
            attempt=1,
            reference_screenshot_path=ref,
        )
        self.assertTrue(ok["allow"])
        self.assertGreaterEqual(ok["score"], rr.TOCTOU_IMAGE_SIM_MIN)

        device_bad = _FakeDevice([diff], _basic_xml())
        bad = rr.evaluate_toctou_pre_action(
            device=device_bad,
            data_dir=self.root,
            step=1,
            attempt=2,
            reference_screenshot_path=ref,
        )
        self.assertFalse(bad["allow"])
        self.assertLess(bad["score"], rr.TOCTOU_IMAGE_SIM_MIN)

    def test_step_retry_succeeds_on_second_attempt(self) -> None:
        screenshots = [
            self._img("s1_init.jpg", "vertical"),
            self._img("s1_check1.jpg", "horizontal"),
            self._img("s1_check2.jpg", "horizontal"),
            self._img("s2_init.jpg", "horizontal"),
        ]
        device = _FakeDevice(screenshots, _basic_xml())
        decider_payloads = [
            json.dumps(
                {
                    "reasoning": "click attempt 1",
                    "action": "click",
                    "parameters": {"target_element": "button"},
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "reasoning": "click attempt 2",
                    "action": "click",
                    "parameters": {"target_element": "button"},
                },
                ensure_ascii=False,
            ),
            json.dumps({"reasoning": "done", "action": "done", "parameters": {}}, ensure_ascii=False),
        ]

        result = self._run_with_patches(device, decider_payloads)
        click_actions = [a for a in result["actions"] if a.get("action") == "click"]
        self.assertEqual(1, len(click_actions))
        self.assertEqual(1, len(device.click_calls))
        self.assertEqual(2, click_actions[0]["toctou_check"]["attempt"])
        self.assertEqual("done", result.get("stop_reason"))

    def test_step_retry_exhausted_skips_action(self) -> None:
        screenshots = [
            self._img("s1_init.jpg", "vertical"),
            self._img("s1_check1.jpg", "horizontal"),
            self._img("s1_check2.jpg", "checker"),
            self._img("s1_check3.jpg", "diag"),
            self._img("s2_init.jpg", "center_box"),
        ]
        device = _FakeDevice(screenshots, _basic_xml())
        decider_payloads = [
            json.dumps(
                {
                    "reasoning": "click attempt 1",
                    "action": "click",
                    "parameters": {"target_element": "button"},
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "reasoning": "click attempt 2",
                    "action": "click",
                    "parameters": {"target_element": "button"},
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "reasoning": "click attempt 3",
                    "action": "click",
                    "parameters": {"target_element": "button"},
                },
                ensure_ascii=False,
            ),
            json.dumps({"reasoning": "done", "action": "done", "parameters": {}}, ensure_ascii=False),
        ]

        result = self._run_with_patches(device, decider_payloads)
        click_actions = [a for a in result["actions"] if a.get("action") == "click"]
        self.assertEqual(0, len(click_actions))
        self.assertEqual(0, len(device.click_calls))
        self.assertEqual("done", result.get("stop_reason"))

    def test_toctou_threshold_stricter_than_node_reuse(self) -> None:
        self.assertGreater(rr.TOCTOU_IMAGE_SIM_MIN, TemplateGraphBuilder.HIGH_MATCH_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
