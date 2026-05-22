from __future__ import annotations

import ast
import base64
import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class WorkflowTool(ABC):
    name: str

    @abstractmethod
    def run(self, inputs: dict[str, Any], context, runner) -> dict[str, Any]:
        """Run one workflow tool and return a JSON-serializable result."""


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, WorkflowTool] = {}

    def register(self, tool: WorkflowTool) -> None:
        self._tools[tool.name] = tool

    def run(self, name: str, inputs: dict[str, Any], context, runner) -> dict[str, Any]:
        if name not in self._tools:
            available = ", ".join(sorted(self._tools)) or "<none>"
            raise KeyError(f"Unknown workflow tool '{name}'. Available tools: {available}")
        return self._tools[name].run(inputs, context, runner)

    def list_tools(self) -> list[str]:
        return sorted(self._tools)


JSON_BLOCK_PATTERN = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _extract_object_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    start_indices = [index for index, char in enumerate(text) if char == "{"]
    end_indices = [index for index, char in enumerate(text) if char == "}"]
    for start in start_indices:
        for end in reversed(end_indices):
            if end <= start:
                continue
            candidate = text[start : end + 1].strip()
            if candidate and candidate not in candidates:
                candidates.append(candidate)
                break
    return candidates


def _load_json_from_text(raw_text: str) -> dict[str, Any]:
    text = (raw_text or "").strip()
    if not text:
        raise ValueError("Structured tool output is empty")

    candidates = [text]
    match = JSON_BLOCK_PATTERN.search(text)
    if match:
        candidates.insert(0, match.group(1).strip())
    for candidate in _extract_object_candidates(text):
        if candidate not in candidates:
            candidates.append(candidate)

    for candidate in candidates:
        try:
            loaded = json.loads(candidate)
        except json.JSONDecodeError:
            loaded = None
        if isinstance(loaded, dict):
            return loaded

        try:
            loaded = ast.literal_eval(candidate)
        except (SyntaxError, ValueError):
            continue
        if isinstance(loaded, dict):
            return loaded

    raise ValueError(f"Failed to parse structured JSON output from tool response: {text[:200]}")


def _normalize_json_schema(json_schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    for field_name, field_spec in (json_schema or {}).items():
        if isinstance(field_spec, str):
            normalized[field_name] = {"type": field_spec, "description": ""}
        elif isinstance(field_spec, dict):
            normalized[field_name] = {
                "type": str(field_spec.get("type", "string")),
                "description": str(field_spec.get("description", "")),
            }
        else:
            raise TypeError(f"json_schema field '{field_name}' must be a string or object")
    return normalized


def _build_json_schema_instruction(json_schema: dict[str, dict[str, Any]]) -> str:
    lines = [
        "请严格输出一个 JSON 对象，不要输出额外解释。",
        "JSON 字段要求如下：",
    ]
    for field_name, field_spec in json_schema.items():
        description = field_spec.get("description", "")
        if description:
            lines.append(f"- {field_name}: type={field_spec['type']}, description={description}")
        else:
            lines.append(f"- {field_name}: type={field_spec['type']}")
    lines.append("布尔值请使用 true/false，字符串请直接返回字符串。")
    return "\n".join(lines)


def _validate_structured_output(parsed_output: dict[str, Any], json_schema: dict[str, dict[str, Any]]) -> None:
    type_mapping = {
        "string": str,
        "boolean": bool,
        "number": (int, float),
        "integer": int,
        "object": dict,
        "array": list,
    }

    for field_name, field_spec in json_schema.items():
        if field_name not in parsed_output:
            raise ValueError(f"Structured tool output is missing required field: {field_name}")

        expected_type = field_spec["type"].lower()
        python_type = type_mapping.get(expected_type)
        if python_type is None:
            raise ValueError(f"Unsupported json_schema type: {field_spec['type']}")

        value = parsed_output[field_name]
        if expected_type == "number" and isinstance(value, bool):
            raise ValueError(f"Field '{field_name}' expected number but got boolean")
        if expected_type == "integer" and isinstance(value, bool):
            raise ValueError(f"Field '{field_name}' expected integer but got boolean")
        if not isinstance(value, python_type):
            raise ValueError(
                f"Field '{field_name}' expected type {field_spec['type']} but got {type(value).__name__}"
            )


def _image_path_to_data_url(image_path: str) -> str:
    path = Path(image_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    elif suffix == ".png":
        mime = "image/png"
    elif suffix == ".webp":
        mime = "image/webp"
    else:
        mime = "application/octet-stream"

    image_b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{image_b64}"


def _extract_summary_text(output_text: str, structured_output: dict[str, Any] | None) -> str:
    if structured_output is not None:
        summary = structured_output.get("summary")
        if isinstance(summary, str):
            return summary.strip()
    return (output_text or "").strip()


class VLMQATool(WorkflowTool):
    name = "vlm_qa"

    def run(self, inputs: dict[str, Any], context, runner) -> dict[str, Any]:
        question = str(inputs.get("question", "")).strip()
        image = str(inputs.get("image", "")).strip()
        mode = str(inputs.get("mode", "answer")).strip() or "answer"
        json_schema = inputs.get("json_schema")
        if not question:
            raise ValueError("vlm_qa requires a non-empty 'question'")
        if not image:
            raise ValueError("vlm_qa requires a non-empty 'image'")

        message_text = question
        if mode == "summary":
            message_text = f"请基于给定图片做总结：{question}"

        normalized_schema = None
        if json_schema:
            if not isinstance(json_schema, dict):
                raise TypeError("vlm_qa json_schema must be an object")
            normalized_schema = _normalize_json_schema(json_schema)
            message_text = f"{message_text}\n\n{_build_json_schema_instruction(normalized_schema)}"

        response = runner.planner_client.chat.completions.create(
            model=runner.planner_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": _image_path_to_data_url(image)}},
                        {"type": "text", "text": message_text},
                    ],
                }
            ],
        )
        output_text = response.choices[0].message.content
        structured_output = None
        result = {
            "tool_name": self.name,
            "mode": mode,
            "question": question,
            "image": image,
            "response": output_text,
        }
        if normalized_schema is not None:
            structured_output = _load_json_from_text(output_text)
            _validate_structured_output(structured_output, normalized_schema)
            result["structured_output"] = structured_output
            result["json_schema"] = normalized_schema

        if mode == "summary":
            summary_text = _extract_summary_text(output_text, structured_output)
            if summary_text:
                daily_log_path = runner.append_daily_summary_log(summary_text)
                result["daily_log_path"] = daily_log_path
        return result


def create_default_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(VLMQATool())
    return registry