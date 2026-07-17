from __future__ import annotations

import argparse
import json
import logging

from runner.mobiagent.decider_adapters import SUPPORTED_DECIDER_PROTOCOLS
from runner.mobiagent.workflow import WorkflowRunner


def parse_context_overrides(raw_items: list[str] | None) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in raw_items or []:
        if "=" not in item:
            raise ValueError(f"Invalid --context value '{item}'. Expected key=value")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --context value '{item}'. Key cannot be empty")
        overrides[key] = value
    return overrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MobiAgent Workflow Runner")
    parser.add_argument("--workflow_file", required=True, help="Path to the workflow JSON file")
    parser.add_argument("--service_ip", type=str, default="localhost", help="Ip for the services (default: localhost)")
    parser.add_argument("--decider_port", type=int, default=8000, help="Port for decider service (default: 8000)")
    parser.add_argument("--grounder_port", type=int, default=8001, help="Port for grounder service (default: 8001)")
    parser.add_argument("--planner_port", type=int, default=8002, help="Port for planner service (default: 8002)")
    parser.add_argument("--device", choices=["Android", "Harmony"], default="Android", help="Default device type for workflow steps")
    parser.add_argument("--user_profile", choices=["on", "off"], default="off", help="Enable user profile memory")
    parser.add_argument("--use_graphrag", choices=["on", "off"], default="off", help="Enable GraphRAG for user profile memory")
    parser.add_argument("--use_qwen3", choices=["on", "off"], default="on", help="Use Qwen3 for GUI execution steps")
    parser.add_argument("--e2e", action="store_true", default=True, help="Enable e2e mode for GUI task steps")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for workflow run outputs")
    parser.add_argument(
        "--decider_protocol",
        choices=SUPPORTED_DECIDER_PROTOCOLS,
        default=SUPPORTED_DECIDER_PROTOCOLS[0],
        help="Decider output protocol for gui_task steps",
    )
    parser.add_argument(
        "--context",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override workflow context values, for example --context contact_name=小赵",
    )
    parser.add_argument(
        "--accept_planner_changes",
        choices=["on", "off"],
        default="off",
        help="Automatically accept planner task rewrites during gui_task steps",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    context_overrides = parse_context_overrides(args.context)
    runner = WorkflowRunner(
        workflow_file=args.workflow_file,
        service_ip=args.service_ip,
        decider_port=args.decider_port,
        grounder_port=args.grounder_port,
        planner_port=args.planner_port,
        device_type=args.device,
        output_dir=args.output_dir,
        use_qwen3=(args.use_qwen3 == "on"),
        use_e2e=args.e2e,
        enable_user_profile=(args.user_profile == "on"),
        use_graphrag=(args.use_graphrag == "on"),
        auto_accept_planner_changes=(args.accept_planner_changes == "on"),
        decider_protocol=args.decider_protocol,
        context_overrides=context_overrides,
    )
    summary = runner.run()
    logging.info("Workflow finished with status: %s", summary["status"])
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())