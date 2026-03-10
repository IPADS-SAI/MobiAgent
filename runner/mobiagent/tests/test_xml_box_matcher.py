#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import unittest

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

from xml_box_matcher import match_pred_box_to_xml, validate_match


class XmlBoxMatcherTests(unittest.TestCase):
    def test_large_pred_box_picks_small_clickable_button(self) -> None:
        xml = """
<hierarchy rotation="0">
  <node class="android.widget.FrameLayout" package="me.ele" bounds="[0,0][1240,2772]" clickable="false" enabled="true">
    <node class="android.widget.LinearLayout" package="me.ele" bounds="[0,0][1240,2772]" clickable="false" enabled="true">
      <node class="android.widget.Button" package="me.ele" bounds="[520,2100][720,2200]" clickable="true" enabled="true"
            text="pay now" resource-id="me.ele:id/submit_order"/>
    </node>
  </node>
</hierarchy>
"""
        pred_box = [100, 1700, 1140, 2500]
        match = match_pred_box_to_xml(
            xml_str=xml,
            pred_box=pred_box,
            action_type="CLICK",
            expected_text="pay",
            target_package="me.ele",
        )
        self.assertEqual("MATCHED", match.status)
        self.assertIsNotNone(match.best_node)
        self.assertIn("button", (match.best_node.class_name or "").lower())

        validation = validate_match(
            match_result=match,
            expected_text="pay",
            action_type="CLICK",
        )
        self.assertTrue(validation.ok)
        self.assertIsNotNone(validation.click_node)
        self.assertEqual("me.ele:id/submit_order", validation.click_node.resource_id)

    def test_ambiguous_multiple_clickable_nodes(self) -> None:
        xml = """
<hierarchy rotation="0">
  <node class="android.widget.FrameLayout" package="me.ele" bounds="[0,0][1240,2772]" clickable="false" enabled="true">
    <node class="android.widget.Button" package="me.ele" bounds="[200,200][450,330]" clickable="true" enabled="true"
          text="" resource-id="me.ele:id/left_btn"/>
    <node class="android.widget.Button" package="me.ele" bounds="[790,200][1040,330]" clickable="true" enabled="true"
          text="" resource-id="me.ele:id/right_btn"/>
  </node>
</hierarchy>
"""
        pred_box = [100, 150, 1140, 400]
        match = match_pred_box_to_xml(
            xml_str=xml,
            pred_box=pred_box,
            action_type="CLICK",
            expected_text=None,
            target_package="me.ele",
        )
        self.assertEqual("AMBIGUOUS", match.status)
        self.assertGreaterEqual(len(match.candidates), 2)

    def test_systemui_is_penalized_for_app_target(self) -> None:
        xml = """
<hierarchy rotation="0">
  <node class="android.widget.FrameLayout" package="me.ele" bounds="[0,0][1240,2772]" clickable="false" enabled="true">
    <node class="android.widget.Button" package="me.ele" bounds="[500,150][760,280]" clickable="true" enabled="true"
          text="" resource-id="me.ele:id/app_btn"/>
    <node class="android.widget.Button" package="com.android.systemui" bounds="[480,130][780,290]" clickable="true" enabled="true"
          text="" resource-id="com.android.systemui:id/sys_btn"/>
  </node>
</hierarchy>
"""
        pred_box = [470, 120, 790, 300]
        match = match_pred_box_to_xml(
            xml_str=xml,
            pred_box=pred_box,
            action_type="CLICK",
            expected_text=None,
            target_package="me.ele",
        )
        self.assertEqual("MATCHED", match.status)
        self.assertIsNotNone(match.best_node)
        self.assertEqual("me.ele", match.best_node.package)


if __name__ == "__main__":
    unittest.main()
