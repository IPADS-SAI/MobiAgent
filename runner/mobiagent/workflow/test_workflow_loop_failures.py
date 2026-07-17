from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from runner.mobiagent.workflow.engine import WorkflowContext, WorkflowRunner


class WorkflowLoopFailureTests(unittest.TestCase):
    def test_loop_body_failure_is_not_masked_by_break_condition(self) -> None:
        runner = WorkflowRunner.__new__(WorkflowRunner)
        runner.context = WorkflowContext(Path("workflow.json"), Path("."), {})

        step = {
            "id": "3",
            "type": "loop",
            "max_times": 1,
            "break_if": {
                "left": "${steps.2.output.structured_output.contains_today_chat}",
                "operator": "==",
                "right": False,
            },
            "steps": [],
        }

        with patch.object(runner, "_execute_steps_in_scope", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "Loop body stopped at iteration 1"):
                runner._run_loop(step, "3")

    def test_loop_break_condition_can_read_current_iteration_tool_output(self) -> None:
        runner = WorkflowRunner.__new__(WorkflowRunner)
        runner.context = WorkflowContext(Path("workflow.json"), Path("."), {})

        step = {
            "id": "3",
            "type": "loop",
            "max_times": 1,
            "break_if": {
                "left": "${steps.2.output.structured_output.contains_today_chat}",
                "operator": "==",
                "right": False,
            },
            "steps": [],
        }

        def execute_body(_body_steps, actual_prefix, local_aliases):
            result = type(
                "Result",
                (),
                {
                    "to_dict": lambda self: {
                        "output": {
                            "structured_output": {
                                "contains_today_chat": False,
                            }
                        }
                    }
                },
            )()
            local_aliases["2"] = f"{actual_prefix}.2"
            runner.context.step_results[f"{actual_prefix}.2"] = result
            return False

        with patch.object(runner, "_execute_steps_in_scope", side_effect=execute_body):
            output = runner._run_loop(step, "3")

        self.assertTrue(output["broke_early"])
        self.assertEqual(output["break_iteration"], 1)


if __name__ == "__main__":
    unittest.main()
