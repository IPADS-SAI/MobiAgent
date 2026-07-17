from __future__ import annotations

from prompts.decider_qwen3_e2e import (
    DECIDER_CURRENT_STEP_PROMPT,
    DECIDER_SYSTEM_PROMPT,
    DECIDER_USER_PROMPT,
)

from ..json_utils import robust_json_loads
from .base import DeciderAdapter
from . import DECIDER_PROTOCOL_QWEN_JSON


def build_messages(task: str, history: list[str], screenshot_b64: str, use_e2e: bool, device_type: str) -> list[dict]:
    del use_e2e, device_type

    if len(history) == 0:
        history_str = "(No history)"
    else:
        history_str = "\n".join(f"{idx}. {item}" for idx, item in enumerate(history, 1))

    context_text = DECIDER_USER_PROMPT.format(task=task, history=history_str)

    return [
        {
            "role": "system",
            "content": DECIDER_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": context_text,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{screenshot_b64}",
                    },
                },
                {
                    "type": "text",
                    "text": DECIDER_CURRENT_STEP_PROMPT,
                },
            ],
        },
    ]


def parse_response(response_str: str) -> dict:
    return robust_json_loads(response_str)


def validate_response(response_dict: dict, use_e2e: bool) -> None:
    if not use_e2e:
        return

    action_name = response_dict["action"]
    parameters = response_dict["parameters"]

    if action_name == "click":
        if "bbox" not in parameters or parameters["bbox"] is None:
            raise ValueError("E2E mode: click action missing required parameter: 'bbox'")

    elif action_name == "swipe":
        if "start_coords" in parameters and "end_coords" in parameters:
            if parameters["start_coords"] is None or parameters["end_coords"] is None:
                raise ValueError("E2E mode: swipe action has null start_coords or end_coords")


def get_adapter() -> DeciderAdapter:
    return DeciderAdapter(
        protocol=DECIDER_PROTOCOL_QWEN_JSON,
        display_name="Qwen JSON adapter",
        build_messages=build_messages,
        parse_response=parse_response,
        validate_response=validate_response,
    )