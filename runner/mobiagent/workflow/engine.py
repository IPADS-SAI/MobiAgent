from __future__ import annotations

import json
import logging
import os
import re
import shlex
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from runner.mobiagent import mobiagent

from .tools import ToolRegistry, create_default_tool_registry


VARIABLE_PATTERN = re.compile(r"\$\{([^}]+)\}")
DAILY_LOG_METADATA_PATTERN = re.compile(r"\A<!-- DAILY_LOG_METADATA\n(.*?)\n-->\n*", re.DOTALL)


def load_workflow_definition(workflow_file: str | os.PathLike[str]) -> dict[str, Any]:
    path = Path(workflow_file).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_workflow_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized_steps: list[dict[str, Any]] = []
    seen_step_ids: set[str] = set()

    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            raise TypeError(f"Workflow step at index {index} must be an object")
        if "id" not in step:
            raise ValueError(f"Workflow step at index {index} is missing required field 'id'")
        if "type" not in step:
            raise ValueError(f"Workflow step '{step.get('id')}' is missing required field 'type'")

        normalized_step = dict(step)
        normalized_step["id"] = str(step["id"])
        normalized_step.setdefault("name", normalized_step["id"])

        if normalized_step["type"] == "loop":
            normalized_step["steps"] = normalize_workflow_steps(normalized_step.get("steps", []))
        elif normalized_step["type"] == "if":
            normalized_step["then_steps"] = normalize_workflow_steps(normalized_step.get("then_steps", []))
            normalized_step["else_steps"] = normalize_workflow_steps(normalized_step.get("else_steps", []))

        if normalized_step["id"] in seen_step_ids:
            raise ValueError(f"Duplicate workflow step id: {normalized_step['id']}")
        seen_step_ids.add(normalized_step["id"])
        normalized_steps.append(normalized_step)

    return normalized_steps


@dataclass
class StepResult:
    step_id: str
    status: str
    started_at: float
    finished_at: float
    output: dict[str, Any]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_sec": round(self.finished_at - self.started_at, 3),
            "output": self.output,
            "error": self.error,
        }


class WorkflowContext:
    def __init__(self, workflow_path: Path, run_dir: Path, initial_context: dict[str, Any] | None = None) -> None:
        self.workflow_path = workflow_path
        self.run_dir = run_dir
        self.initial_context = initial_context or {}
        self.step_results: dict[str, StepResult] = {}
        self._step_alias_scopes: list[dict[str, str]] = []
        self._runtime_scopes: list[dict[str, Any]] = []

    def set_step_result(self, result: StepResult) -> None:
        self.step_results[result.step_id] = result

    @contextmanager
    def push_scope(self, step_aliases: dict[str, str] | None = None, runtime_values: dict[str, Any] | None = None):
        alias_scope = step_aliases if step_aliases is not None else {}
        runtime_scope = runtime_values if runtime_values is not None else {}
        self._step_alias_scopes.append(alias_scope)
        self._runtime_scopes.append(runtime_scope)
        try:
            yield alias_scope
        finally:
            self._step_alias_scopes.pop()
            self._runtime_scopes.pop()

    def resolve(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: self.resolve(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.resolve(item) for item in value]
        if isinstance(value, str):
            return self._resolve_string(value)
        return value

    def summary(self) -> dict[str, Any]:
        return {
            "workflow_file": str(self.workflow_path),
            "run_dir": str(self.run_dir),
            "context": self.initial_context,
            "steps": {key: value.to_dict() for key, value in self.step_results.items()},
        }

    def _resolve_string(self, raw: str) -> str:
        matches = VARIABLE_PATTERN.findall(raw)
        if not matches:
            return raw

        if len(matches) == 1 and raw.strip() == f"${{{matches[0]}}}":
            return self._lookup(matches[0])

        resolved = raw
        for expression in matches:
            replacement = self._lookup(expression)
            if not isinstance(replacement, str):
                replacement = json.dumps(replacement, ensure_ascii=False)
            resolved = resolved.replace(f"${{{expression}}}", replacement)
        return resolved

    def _lookup(self, expression: str) -> Any:
        parts = expression.split(".")
        if not parts:
            raise KeyError("Empty variable expression")

        if parts[0] == "context":
            current: Any = self.initial_context
            for part in parts[1:]:
                current = current[part]
            return current

        for scope in reversed(self._runtime_scopes):
            if parts[0] in scope:
                current = scope[parts[0]]
                for part in parts[1:]:
                    current = current[part]
                return current

        if parts[0] == "run":
            mapping = {
                "dir": str(self.run_dir),
                "workflow_file": str(self.workflow_path),
                "workflow_dir": str(self.workflow_path.parent),
            }
            current = mapping
            for part in parts[1:]:
                current = current[part]
            return current

        if parts[0] == "steps":
            if len(parts) < 3:
                raise KeyError(f"Invalid step variable expression: {expression}")
            step_id = parts[1]
            for scope in reversed(self._step_alias_scopes):
                if step_id in scope:
                    step_id = scope[step_id]
                    break
            result = self.step_results[step_id].to_dict()
            current: Any = result
            for part in parts[2:]:
                current = current[part]
            return current

        raise KeyError(f"Unsupported variable root: {parts[0]}")


class WorkflowRunner:
    def __init__(
        self,
        workflow_file: str,
        service_ip: str,
        decider_port: int,
        grounder_port: int,
        planner_port: int,
        device_type: str = "Android",
        output_dir: str | None = None,
        use_qwen3: bool = True,
        use_e2e: bool = True,
        enable_user_profile: bool = False,
        use_graphrag: bool = False,
        auto_accept_planner_changes: bool = False,
        context_overrides: dict[str, Any] | None = None,
        decider_protocol: str = mobiagent.DECIDER_PROTOCOL_QWEN_JSON,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self.workflow_path = Path(workflow_file).expanduser().resolve()
        self.workflow = load_workflow_definition(self.workflow_path)
        self.defaults = self.workflow.get("defaults", {})
        self.metadata = self.workflow.get("metadata", {})
        self.context_values = dict(self.workflow.get("context", {}))
        if context_overrides:
            self.context_values.update(context_overrides)
        self.steps = normalize_workflow_steps(self.workflow.get("steps", []))
        self.device_type = device_type or self.defaults.get("device", "Android")
        self.use_qwen3 = use_qwen3
        self.use_e2e = use_e2e
        self.enable_user_profile = enable_user_profile
        self.use_graphrag = use_graphrag
        self.auto_accept_planner_changes = auto_accept_planner_changes
        self.decider_protocol = decider_protocol
        self.tool_registry = tool_registry or create_default_tool_registry()
        self._devices: dict[str, Any] = {}
        self.current_app_name: str | None = None
        self.current_package_name: str | None = None
        self.current_device_type: str | None = None

        if not self.steps:
            raise ValueError("Workflow must contain at least one step")

        base_output_dir = Path(output_dir or self.defaults.get("output_dir", self.workflow_path.parent / "runs"))
        self.base_output_dir = base_output_dir.expanduser().resolve()
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.run_date = time.strftime("%Y-%m-%d")
        workflow_name = self.metadata.get("name") or self.workflow_path.stem
        self.run_dir = self.base_output_dir / f"{timestamp}-{workflow_name}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.context = WorkflowContext(
            workflow_path=self.workflow_path,
            run_dir=self.run_dir,
            initial_context=self.context_values,
        )

        mobiagent.init(
            service_ip,
            decider_port,
            grounder_port,
            planner_port,
            enable_user_profile=enable_user_profile,
            use_graphrag=use_graphrag,
        )
        self.planner_client = mobiagent.planner_client
        self.planner_model = mobiagent.planner_model

    def run(self) -> dict[str, Any]:
        logging.info("Starting workflow run: %s", self.workflow_path)
        summary_path = self.run_dir / "run_summary.json"
        stopped = self._execute_steps(self.steps)

        summary = self.context.summary()
        summary["status"] = "failed" if stopped else "success"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        logging.info("Workflow summary written to %s", summary_path)
        return summary

    def append_daily_summary_log(self, summary_text: str) -> str:
        file_name = self._build_daily_log_file_name()
        daily_log_dir = self.base_output_dir / "daily-log" / self.run_date
        daily_log_dir.mkdir(parents=True, exist_ok=True)
        log_path = daily_log_dir / file_name

        text = summary_text.strip()
        if not text:
            return str(log_path)

        existing_metadata, existing_body = self._read_daily_log_document(log_path)
        entry_number = max(
            int(existing_metadata.get("latest_entry_index", 0) or 0),
            self._get_last_daily_log_entry_number(existing_body),
        ) + 1
        entry_text = f"{entry_number}. {text}"

        body = existing_body.rstrip()
        if body:
            body = f"{body}\n\n{entry_text}\n"
        else:
            body = f"{entry_text}\n"

        metadata = {
            "workflow_metadata": self.metadata,
            "latest_entry_index": entry_number,
        }
        log_path.write_text(self._render_daily_log_document(metadata, body), encoding="utf-8")
        return str(log_path)

    def _read_daily_log_document(self, log_path: Path) -> tuple[dict[str, Any], str]:
        metadata = {
            "workflow_metadata": self.metadata,
            "latest_entry_index": 0,
        }
        if not log_path.exists():
            return metadata, ""

        content = log_path.read_text(encoding="utf-8")
        match = DAILY_LOG_METADATA_PATTERN.match(content)
        if not match:
            return metadata, content

        raw_metadata = match.group(1)
        try:
            parsed_metadata = json.loads(raw_metadata)
        except json.JSONDecodeError:
            logging.warning("Failed to parse daily-log metadata header: %s", log_path)
            return metadata, content

        if isinstance(parsed_metadata, dict):
            metadata.update(parsed_metadata)
        body = content[match.end():]
        return metadata, body

    def _get_last_daily_log_entry_number(self, content: str) -> int:
        matches = re.findall(r"(?m)^(\d+)\.\s", content)
        if not matches:
            return 0
        return int(matches[-1])

    def _render_daily_log_document(self, metadata: dict[str, Any], body: str) -> str:
        serialized_metadata = json.dumps(metadata, ensure_ascii=False, indent=2)
        normalized_body = body.lstrip("\n")
        return f"<!-- DAILY_LOG_METADATA\n{serialized_metadata}\n-->\n\n{normalized_body}"

    def _build_daily_log_file_name(self) -> str:
        name_parts = [
            self._sanitize_file_name_part(self.current_package_name or "unknown-package"),
            self._sanitize_file_name_part(self.workflow_path.stem),
        ]
        context_values = self._get_context_file_name_parts()
        name_parts.extend(context_values)
        filtered_parts = [part for part in name_parts if part]
        return "__".join(filtered_parts) + ".md"

    def _get_context_file_name_parts(self) -> list[str]:
        context_values: list[str] = []
        for value in self.context.initial_context.values():
            if value is None:
                continue
            if isinstance(value, (dict, list)):
                serialized = json.dumps(value, ensure_ascii=False, sort_keys=True)
                context_values.append(self._sanitize_file_name_part(serialized))
            else:
                context_values.append(self._sanitize_file_name_part(str(value)))
        return [value for value in context_values if value]

    def _sanitize_file_name_part(self, value: str) -> str:
        sanitized = str(value).strip()
        sanitized = re.sub(r"[\\/:*?\"<>|]+", "_", sanitized)
        sanitized = re.sub(r"\s+", "-", sanitized)
        sanitized = sanitized.strip("._-")
        return sanitized

    def _execute_steps(
        self,
        steps: list[dict[str, Any]],
        actual_prefix: str = "",
        runtime_values: dict[str, Any] | None = None,
    ) -> bool:
        local_aliases: dict[str, str] = {}
        with self.context.push_scope(local_aliases, runtime_values):
            return self._execute_steps_in_scope(steps, actual_prefix, local_aliases)
        return False

    def _execute_steps_in_scope(
        self,
        steps: list[dict[str, Any]],
        actual_prefix: str,
        local_aliases: dict[str, str],
    ) -> bool:
        for step in steps:
            local_step_id = step["id"]
            actual_step_id = local_step_id if not actual_prefix else f"{actual_prefix}.{local_step_id}"
            started_at = time.time()
            try:
                output = self._execute_step(step, actual_step_id)
                result = StepResult(
                    step_id=actual_step_id,
                    status="success",
                    started_at=started_at,
                    finished_at=time.time(),
                    output=output,
                )
            except Exception as exc:
                logging.exception("Workflow step failed: %s", actual_step_id)
                result = StepResult(
                    step_id=actual_step_id,
                    status="failed",
                    started_at=started_at,
                    finished_at=time.time(),
                    output={},
                    error=str(exc),
                )
                self.context.set_step_result(result)
                local_aliases[local_step_id] = actual_step_id
                if step.get("on_error", "stop") != "continue":
                    return True
                continue

            self.context.set_step_result(result)
            local_aliases[local_step_id] = actual_step_id

    def _execute_step(self, step: dict[str, Any], actual_step_id: str) -> dict[str, Any]:
        step_type = step["type"]
        if step_type == "gui_task":
            return self._run_gui_task(self.context.resolve(step), actual_step_id)
        if step_type == "gui_action":
            return self._run_gui_action(self.context.resolve(step), actual_step_id)
        if step_type == "command":
            return self._run_command(self.context.resolve(step), actual_step_id)
        if step_type == "tool":
            return self._run_tool(self.context.resolve(step), actual_step_id)
        if step_type == "loop":
            return self._run_loop(step, actual_step_id)
        if step_type == "if":
            return self._run_if(step, actual_step_id)
        raise ValueError(f"Unsupported workflow step type: {step_type}")

    def _run_gui_task(self, step: dict[str, Any], actual_step_id: str) -> dict[str, Any]:
        step_dir = self._prepare_step_dir(actual_step_id)
        current_device_type = step.get("device", self.device_type)
        device = self._get_device(current_device_type)
        use_experience = bool(step.get("use_experience", self.defaults.get("use_experience", False)))
        use_e2e = bool(step.get("use_e2e", self.defaults.get("use_e2e", self.use_e2e)))
        use_qwen3 = bool(step.get("use_qwen3", self.defaults.get("use_qwen3", self.use_qwen3)))
        use_graphrag = bool(step.get("use_graphrag", self.defaults.get("use_graphrag", self.use_graphrag)))
        decider_protocol = str(
            step.get("decider_protocol", self.defaults.get("decider_protocol", self.decider_protocol))
        )
        auto_accept = bool(
            step.get(
                "accept_planner_changes",
                self.defaults.get("accept_planner_changes", self.auto_accept_planner_changes),
            )
        )
        task_description = str(step["task_description"])

        if actual_step_id == "1":
            app_name, package_name, planner_task_description = mobiagent.get_app_package_name(
                task_description,
                use_graphrag=use_graphrag,
                device_type=current_device_type,
                use_experience=use_experience,
            )
            logging.info(
                "Workflow gui_task step 1 uses planner only: app=%s package=%s",
                app_name,
                package_name,
            )
            device.app_start(package_name)
            self.current_app_name = app_name
            self.current_package_name = package_name
            self.current_device_type = current_device_type
            return {
                "step_dir": str(step_dir),
                "task_description": task_description,
                "planner_task_description": planner_task_description,
                "app_name": app_name,
                "package_name": package_name,
                "device": current_device_type,
                "mode": "planner_only",
                "use_experience": use_experience,
            }

        if not self.current_app_name:
            raise RuntimeError(
                "gui_task steps after id=1 require an active app context. "
                "Please ensure step 1 is a gui_task that lets planner choose and open the app first."
            )

        logging.info(
            "Workflow gui_task step %s uses decider only within current app %s (%s)",
            step["id"],
            self.current_app_name,
            self.current_package_name,
        )
        mobiagent.task_in_app(
            self.current_app_name,
            task_description,
            task_description,
            device,
            str(step_dir),
            True,
            use_qwen3,
            current_device_type,
            use_e2e,
            decider_protocol=decider_protocol,
        )
        return {
            "step_dir": str(step_dir),
            "task_description": task_description,
            "device": current_device_type,
            "app_name": self.current_app_name,
            "package_name": self.current_package_name,
            "mode": "decider_only",
            "use_experience": use_experience,
            "use_e2e": use_e2e,
            "accept_planner_changes": auto_accept,
            "decider_protocol": decider_protocol,
        }

    def _run_gui_action(self, step: dict[str, Any], actual_step_id: str) -> dict[str, Any]:
        step_dir = self._prepare_step_dir(actual_step_id)
        current_device_type = step.get("device", self.device_type)
        device = self._get_device(current_device_type)
        action = step["action"]
        output: dict[str, Any] = {
            "step_dir": str(step_dir),
            "action": action,
            "device": current_device_type,
        }

        if action == "click":
            device.click(int(step["x"]), int(step["y"]))
            output.update({"x": int(step["x"]), "y": int(step["y"])})
        elif action == "input":
            text = str(step["text"])
            device.input(text)
            output["text"] = text
        elif action == "swipe":
            device.swipe(str(step["direction"]))
            output["direction"] = str(step["direction"])
        elif action == "swipe_with_coords":
            start_x = int(step["start_x"])
            start_y = int(step["start_y"])
            end_x = int(step["end_x"])
            end_y = int(step["end_y"])
            device.swipe_with_coords(start_x, start_y, end_x, end_y)
            output.update({
                "start_x": start_x,
                "start_y": start_y,
                "end_x": end_x,
                "end_y": end_y,
            })
        elif action == "keyevent":
            key = step["key"]
            device.keyevent(key)
            output["key"] = key
        elif action == "sleep":
            seconds = float(step.get("seconds", 1.0))
            time.sleep(seconds)
            output["seconds"] = seconds
        elif action == "screenshot":
            file_name = step.get("file_name", "screenshot.jpg")
            image_path = step_dir / file_name
            device.screenshot(str(image_path))
            output["image_path"] = str(image_path)
        elif action == "app_start":
            package_name = str(step["package_name"])
            device.app_start(package_name)
            self.current_package_name = package_name
            self.current_device_type = current_device_type
            output["package_name"] = package_name
        elif action == "app_stop":
            package_name = str(step["package_name"])
            device.app_stop(package_name)
            if self.current_package_name == package_name:
                self.current_package_name = None
            output["package_name"] = package_name
        else:
            raise ValueError(f"Unsupported gui_action action: {action}")

        return output

    def _run_command(self, step: dict[str, Any], actual_step_id: str) -> dict[str, Any]:
        step_dir = self._prepare_step_dir(actual_step_id)
        timeout = float(step.get("timeout_sec", self.defaults.get("command_timeout_sec", 60)))
        capture_output = bool(step.get("capture_output", True))
        allow_failure = bool(step.get("allow_failure", False))
        env = os.environ.copy()
        extra_env = step.get("env", {})
        env.update({key: str(value) for key, value in extra_env.items()})

        cwd_value = step.get("cwd")
        cwd = self.workflow_path.parent if not cwd_value else Path(str(cwd_value)).expanduser()
        if not cwd.is_absolute():
            cwd = (self.workflow_path.parent / cwd).resolve()

        if "command" in step:
            command = str(step["command"])
            completed = subprocess.run(
                command,
                shell=True,
                cwd=str(cwd),
                env=env,
                timeout=timeout,
                capture_output=capture_output,
                text=True,
            )
            command_repr = command
        else:
            program = str(step["program"])
            args = [str(item) for item in step.get("args", [])]
            completed = subprocess.run(
                [program, *args],
                shell=False,
                cwd=str(cwd),
                env=env,
                timeout=timeout,
                capture_output=capture_output,
                text=True,
            )
            command_repr = shlex.join([program, *args])

        if completed.returncode != 0 and not allow_failure:
            raise RuntimeError(
                f"Command step failed with exit code {completed.returncode}: {command_repr}\n{completed.stderr}"
            )

        stdout_path = step_dir / "stdout.txt"
        stderr_path = step_dir / "stderr.txt"
        if capture_output:
            stdout_path.write_text(completed.stdout or "", encoding="utf-8")
            stderr_path.write_text(completed.stderr or "", encoding="utf-8")

        return {
            "step_dir": str(step_dir),
            "command": command_repr,
            "cwd": str(cwd),
            "returncode": completed.returncode,
            "stdout": completed.stdout or "",
            "stderr": completed.stderr or "",
            "stdout_path": str(stdout_path) if capture_output else "",
            "stderr_path": str(stderr_path) if capture_output else "",
        }

    def _run_tool(self, step: dict[str, Any], actual_step_id: str) -> dict[str, Any]:
        step_dir = self._prepare_step_dir(actual_step_id)
        tool_name = str(step["tool_name"])
        inputs = dict(step.get("inputs", {}))
        output = self.tool_registry.run(tool_name, inputs, self.context, self)
        output["step_dir"] = str(step_dir)
        return output

    def _run_loop(self, step: dict[str, Any], actual_step_id: str) -> dict[str, Any]:
        max_times_value = step.get("max_times", step.get("times", 0))
        max_times = int(self.context.resolve(max_times_value))
        if max_times < 0:
            raise ValueError("loop.max_times/times must be >= 0")

        executed_iterations: list[dict[str, Any]] = []
        body_steps = step.get("steps", [])
        broke_early = False
        break_iteration = None
        break_if_condition = step.get("break_if")

        for iteration_index in range(max_times):
            loop_runtime = {
                "loop": {
                    "index": iteration_index + 1,
                    "index0": iteration_index,
                    "count": max_times,
                    "first": iteration_index == 0,
                    "last": iteration_index == max_times - 1,
                }
            }
            iteration_prefix = f"{actual_step_id}.iter{iteration_index + 1}"
            local_aliases: dict[str, str] = {}
            with self.context.push_scope(local_aliases, loop_runtime):
                stopped = self._execute_steps_in_scope(body_steps, iteration_prefix, local_aliases)
                should_break = self._evaluate_condition(break_if_condition) if break_if_condition is not None else False
            executed_iterations.append(
                {
                    "iteration": iteration_index + 1,
                    "prefix": iteration_prefix,
                    "break_triggered": should_break,
                }
            )
            if stopped:
                raise RuntimeError(f"Loop body stopped at iteration {iteration_index + 1}")
            if should_break:
                broke_early = True
                break_iteration = iteration_index + 1
                break

        return {
            "times": len(executed_iterations),
            "max_times": max_times,
            "broke_early": broke_early,
            "break_iteration": break_iteration,
            "iterations": executed_iterations,
        }

    def _run_if(self, step: dict[str, Any], actual_step_id: str) -> dict[str, Any]:
        condition = step.get("condition")
        matched = self._evaluate_condition(condition)
        branch_name = "then" if matched else "else"
        branch_steps = step.get("then_steps", []) if matched else step.get("else_steps", [])
        if branch_steps:
            stopped = self._execute_steps(branch_steps, actual_prefix=f"{actual_step_id}.{branch_name}")
            if stopped:
                raise RuntimeError(f"If branch '{branch_name}' stopped due to step failure")
        return {
            "matched": matched,
            "branch": branch_name,
            "executed_step_count": len(branch_steps),
        }

    def _evaluate_condition(self, condition: Any) -> bool:
        if isinstance(condition, dict):
            left = self.context.resolve(condition.get("left"))
            right = self.context.resolve(condition.get("right"))
            operator = str(condition.get("operator", "==")).strip()
            return self._apply_condition_operator(left, operator, right)

        resolved = self.context.resolve(condition)
        if isinstance(resolved, str):
            lowered = resolved.strip().lower()
            if lowered in {"", "0", "false", "no", "none", "null"}:
                return False
        return bool(resolved)

    def _apply_condition_operator(self, left: Any, operator: str, right: Any) -> bool:
        if operator == "==":
            return left == right
        if operator == "!=":
            return left != right
        if operator == "contains":
            return str(right) in str(left)
        if operator == "not_contains":
            return str(right) not in str(left)
        if operator == ">":
            return left > right
        if operator == ">=":
            return left >= right
        if operator == "<":
            return left < right
        if operator == "<=":
            return left <= right
        if operator == "in":
            return left in right
        if operator == "not_in":
            return left not in right
        raise ValueError(f"Unsupported condition operator: {operator}")

    def _prepare_step_dir(self, step_id: str) -> Path:
        step_dir = self.run_dir / "steps" / step_id
        step_dir.mkdir(parents=True, exist_ok=True)
        return step_dir

    def _get_device(self, device_type: str):
        if device_type not in self._devices:
            if device_type == "Android":
                self._devices[device_type] = mobiagent.AndroidDevice()
            elif device_type == "Harmony":
                self._devices[device_type] = mobiagent.HarmonyDevice()
            else:
                raise ValueError(f"Unsupported device type: {device_type}")
        return self._devices[device_type]