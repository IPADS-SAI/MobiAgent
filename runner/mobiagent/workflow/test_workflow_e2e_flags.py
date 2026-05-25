from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from runner.mobiagent.workflow.engine import WorkflowContext, WorkflowRunner
from runner.mobiagent.workflow_runner import build_parser


class WorkflowE2EFlagTests(unittest.TestCase):
    def test_workflow_cli_can_disable_e2e(self) -> None:
        parser = build_parser()

        args = parser.parse_args(["--workflow_file", "workflow.json", "--no-e2e"])

        self.assertFalse(args.e2e)

    def test_explicit_cli_e2e_value_overrides_workflow_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = WorkflowRunner.__new__(WorkflowRunner)
            runner.defaults = {"use_e2e": True}
            runner.use_e2e = False
            runner.use_qwen3 = True
            runner.use_graphrag = False
            runner.auto_accept_planner_changes = False
            runner.decider_protocol = "qwen_json"
            runner.device_type = "Harmony"
            runner.current_app_name = "微信"
            runner.current_package_name = "com.tencent.mm"
            runner.context = WorkflowContext(Path("workflow.json"), Path(temp_dir), {})

            with (
                patch.object(runner, "_prepare_step_dir", return_value=Path(temp_dir)),
                patch.object(runner, "_get_device", return_value=object()),
                patch("runner.mobiagent.workflow.engine.mobiagent.task_in_app") as task_in_app,
            ):
                runner._run_gui_task(
                    {
                        "id": "2",
                        "type": "gui_task",
                        "task_description": "搜索小赵",
                    },
                    "2",
                )

        self.assertFalse(task_in_app.call_args.args[8])


if __name__ == "__main__":
    unittest.main()
