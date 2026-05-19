from __future__ import annotations

import base64
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


class VLMQATool(WorkflowTool):
    name = "vlm_qa"

    def run(self, inputs: dict[str, Any], context, runner) -> dict[str, Any]:
        question = str(inputs.get("question", "")).strip()
        image = str(inputs.get("image", "")).strip()
        mode = str(inputs.get("mode", "answer")).strip() or "answer"
        if not question:
            raise ValueError("vlm_qa requires a non-empty 'question'")
        if not image:
            raise ValueError("vlm_qa requires a non-empty 'image'")

        message_text = question
        if mode == "summary":
            message_text = f"请基于给定图片做总结：{question}"

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
        return {
            "tool_name": self.name,
            "mode": mode,
            "question": question,
            "image": image,
            "response": output_text,
        }


def create_default_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(VLMQATool())
    return registry