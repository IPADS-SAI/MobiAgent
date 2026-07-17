from __future__ import annotations

import re

from prompts.decider_stepfun import build_stepfun_messages

from .base import DeciderAdapter
from . import DECIDER_PROTOCOL_STEPFUN_TSV


STEPFUN_ACTION_TO_CANONICAL = {
    "CLICK": "click",
    "TYPE": "input",
    "COMPLETE": "done",
    "WAIT": "wait",
    "AWAKE": "open_app",
    "INFO": "info",
    "ABORT": "abort",
    "SLIDE": "swipe",
    "SWIPE": "swipe",
    "LONGPRESS": "long_press",
    "BACK": "press_back",
    "CALL_USER": "call_user",
}


def _strip_model_thinking(raw_text: str) -> str:
    text = (raw_text or "").strip()
    text = re.sub(r"<\s*THINK\s*>", "<think>", text, flags=re.IGNORECASE)
    text = re.sub(r"<\s*/\s*THINK\s*>", "</think>", text, flags=re.IGNORECASE)
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    return text


def _split_stepfun_kv_segments(kv_text: str) -> list[str]:
    normalized = (kv_text or "").strip()
    if not normalized:
        return []
    if "action: CLICK point:" in normalized:
        normalized = normalized.replace("action: CLICK point:", "action:CLICK\tpoint:")
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    if len(lines) <= 1:
        return [segment.strip() for segment in normalized.split("\t") if segment.strip()]
    if "\t" not in normalized:
        return lines
    return [segment.strip() for segment in normalized.replace("\n", "\t").split("\t") if segment.strip()]


def _validate_stepfun_protocol_fields(fields: dict[str, str], field_order: list[str], protocol_line: str) -> None:
    required_fields = ["verify", "note", "explain", "action", "key_process"]
    missing = [field for field in required_fields if not fields.get(field)]
    if missing:
        raise ValueError(f"StepFun response missing required fields: {', '.join(missing)}")
    if not field_order or field_order[0] != "verify":
        raise ValueError(f"StepFun response must start with verify field: {protocol_line}")
    verify_text = fields["verify"]
    if not verify_text.endswith("因此我判断符合上一步预期") and not verify_text.endswith("因此我判断不符合上一步预期"):
        raise ValueError("StepFun verify field must end with the required judgment sentence")
    action_name = fields["action"].upper()
    if action_name == "CALL_USER":
        tag = fields.get("tag")
        if tag not in {"screenshot_issue", "confirm_action"}:
            raise ValueError(f"Invalid StepFun CALL_USER tag: {tag}")


def _validate_stepfun_legacy_fields(fields: dict[str, str], protocol_line: str) -> None:
    required_fields = ["explain", "action", "summary"]
    missing = [field for field in required_fields if not fields.get(field)]
    if missing:
        raise ValueError(f"StepFun response missing required fields: {', '.join(missing)}")
    action_name = fields.get("action", "").upper()
    if action_name == "CLICK" and "point" not in fields:
        raise ValueError(f"StepFun CLICK response missing point field: {protocol_line}")
    if action_name == "SLIDE" and ("point1" not in fields or "point2" not in fields):
        raise ValueError(f"StepFun SLIDE response missing point1/point2 fields: {protocol_line}")


def _parse_stepfun_point(value: str, field_name: str) -> list[int]:
    normalized = (value or "").strip().replace(" ", ",")
    parts = [item for item in normalized.split(",") if item]
    if len(parts) != 2:
        raise ValueError(f"Invalid StepFun {field_name}: {value}")
    return [int(float(parts[0])), int(float(parts[1]))]


def _infer_swipe_direction(point1: list[int], point2: list[int]) -> str:
    delta_x = point2[0] - point1[0]
    delta_y = point2[1] - point1[1]
    if abs(delta_x) >= abs(delta_y):
        return "RIGHT" if delta_x >= 0 else "LEFT"
    return "DOWN" if delta_y >= 0 else "UP"


def _parse_legacy_response(response_str: str) -> dict:
    command_str = (response_str or "").strip()
    command_str = (
        command_str
        .replace("<TINK>", "<THINK>").replace("</TINK>", "</THINK>")
        .replace("<think>", "<THINK>").replace("</think>", "</THINK>")
    )
    command_str = re.sub(
        r"<\s*/?THINK\s*>",
        lambda match: "<THINK>" if "/" not in match.group() else "</THINK>",
        command_str,
        flags=re.IGNORECASE,
    )

    try:
        cot_part = command_str.split("<THINK>")[1].split("</THINK>")[0].strip()
        kv_part = command_str.split("</THINK>")[1].strip()
    except IndexError:
        kv_part = command_str
        cot_part = ""

    kvs = [item.strip() for item in kv_part.split("\t") if item.strip()]
    fields: dict[str, str] = {}
    for kv in kvs:
        if ":" not in kv:
            continue
        key, value = kv.split(":", 1)
        fields[key.strip().lower()] = value.strip()

    _validate_stepfun_legacy_fields(fields, kv_part)

    stepfun_action = fields.get("action", "").upper()
    if stepfun_action not in STEPFUN_ACTION_TO_CANONICAL:
        raise ValueError(f"Unsupported StepFun action: {stepfun_action}")

    canonical_action = STEPFUN_ACTION_TO_CANONICAL[stepfun_action]
    parameters: dict[str, object] = {}

    if stepfun_action == "CLICK":
        parameters = {
            "target_element": fields.get("explain") or "stepfun_click_target",
            "coords": _parse_stepfun_point(fields["point"], "point"),
        }
    elif stepfun_action == "TYPE":
        text_value = fields.get("value", "")
        if "point" in fields:
            canonical_action = "click_input"
            parameters = {
                "target_element": fields.get("explain") or "stepfun_input_target",
                "coords": _parse_stepfun_point(fields["point"], "point"),
                "text": text_value,
            }
        else:
            parameters = {"text": text_value}
    elif stepfun_action == "COMPLETE":
        parameters = {
            "status": "success",
            "message": fields.get("return") or fields.get("summary") or "",
        }
    elif stepfun_action == "WAIT":
        parameters = {"seconds": float(fields.get("value", "1"))}
    elif stepfun_action == "AWAKE":
        parameters = {"app_name": fields.get("value", "")}
    elif stepfun_action == "INFO":
        parameters = {"question": fields.get("value", "")}
    elif stepfun_action == "ABORT":
        parameters = {"reason": fields.get("value") or fields.get("explain") or ""}
    elif stepfun_action == "SLIDE":
        point1 = _parse_stepfun_point(fields["point1"], "point1")
        point2 = _parse_stepfun_point(fields["point2"], "point2")
        parameters = {
            "direction": _infer_swipe_direction(point1, point2),
            "start_coords": point1,
            "end_coords": point2,
        }
    elif stepfun_action == "LONGPRESS":
        parameters = {
            "target_element": fields.get("explain") or "stepfun_long_press_target",
            "coords": _parse_stepfun_point(fields["point"], "point"),
        }

    return {
        "reasoning": cot_part or fields.get("explain") or kv_part,
        "action": canonical_action,
        "parameters": parameters,
        "raw_protocol": command_str,
        "stepfun_fields": fields,
        "stepfun_field_order": list(fields.keys()),
        "stepfun_format": "legacy_summary",
    }


def _parse_advanced_response(response_str: str) -> dict:
    text = _strip_model_thinking(response_str)
    if not text:
        raise ValueError("StepFun decider response is empty")

    protocol_line = text
    segments = _split_stepfun_kv_segments(protocol_line)
    if not segments:
        raise ValueError(f"StepFun response does not contain protocol fields: {response_str}")

    fields = {}
    field_order = []
    for segment in segments:
        segment = segment.strip()
        if not segment or ":" not in segment:
            continue
        key, value = segment.split(":", 1)
        lowered_key = key.strip().lower()
        field_order.append(lowered_key)
        fields[lowered_key] = value.strip()

    _validate_stepfun_protocol_fields(fields, field_order, protocol_line)

    stepfun_action = fields.get("action", "").upper()
    if stepfun_action not in STEPFUN_ACTION_TO_CANONICAL:
        raise ValueError(f"Unsupported StepFun action: {stepfun_action}")

    canonical_action = STEPFUN_ACTION_TO_CANONICAL[stepfun_action]
    parameters: dict[str, object] = {}

    if stepfun_action == "CLICK":
        parameters = {
            "target_element": fields.get("note") or fields.get("explain") or "stepfun_click_target",
            "coords": _parse_stepfun_point(fields["point"], "point"),
        }
    elif stepfun_action == "TYPE":
        text_value = fields.get("value", "")
        if "point" in fields:
            canonical_action = "click_input"
            parameters = {
                "target_element": fields.get("note") or fields.get("explain") or "stepfun_input_target",
                "coords": _parse_stepfun_point(fields["point"], "point"),
                "text": text_value,
            }
        else:
            parameters = {"text": text_value}
    elif stepfun_action == "COMPLETE":
        parameters = {
            "status": "success",
            "message": fields.get("return") or fields.get("summary") or fields.get("note") or "",
        }
    elif stepfun_action == "WAIT":
        parameters = {"seconds": float(fields.get("value", "1"))}
    elif stepfun_action == "AWAKE":
        parameters = {"app_name": fields.get("value", "")}
    elif stepfun_action == "INFO":
        parameters = {"question": fields.get("value", "")}
    elif stepfun_action == "ABORT":
        parameters = {"reason": fields.get("value") or fields.get("explain") or ""}
    elif stepfun_action in {"SLIDE", "SWIPE"}:
        point1 = _parse_stepfun_point(fields["point1"], "point1")
        point2 = _parse_stepfun_point(fields["point2"], "point2")
        parameters = {
            "direction": _infer_swipe_direction(point1, point2),
            "start_coords": point1,
            "end_coords": point2,
        }
    elif stepfun_action == "LONGPRESS":
        parameters = {
            "target_element": fields.get("note") or fields.get("explain") or "stepfun_long_press_target",
            "coords": _parse_stepfun_point(fields["point"], "point"),
        }
    elif stepfun_action == "BACK":
        parameters = {}
    elif stepfun_action == "CALL_USER":
        parameters = {
            "message": fields.get("value") or fields.get("note") or fields.get("explain") or "",
            "tag": fields.get("tag", "confirm_action"),
        }

    reasoning_parts = []
    for key in ["verify", "note", "key_process", "explain"]:
        value = fields.get(key)
        if value:
            reasoning_parts.append(f"{key}={value}")

    return {
        "reasoning": " | ".join(reasoning_parts) or protocol_line,
        "action": canonical_action,
        "parameters": parameters,
        "raw_protocol": protocol_line,
        "stepfun_fields": fields,
        "stepfun_field_order": field_order,
        "stepfun_format": "advanced_summary",
    }


def build_messages(task: str, history: list[str], screenshot_b64: str, use_e2e: bool, device_type: str) -> list[dict]:
    del use_e2e
    return build_stepfun_messages(
        task=task,
        history=history,
        screenshot_b64=screenshot_b64,
        device_type=device_type,
    )


def parse_response(response_str: str) -> dict:
    text = (response_str or "").strip()
    if not text:
        raise ValueError("StepFun decider response is empty")

    lowered = text.lower()
    if "verify:" in lowered and "key_process:" in lowered:
        return _parse_advanced_response(text)
    return _parse_legacy_response(text)


def validate_response(response_dict: dict, use_e2e: bool) -> None:
    del use_e2e
    fields = response_dict.get("stepfun_fields") or {}
    if response_dict.get("stepfun_format") == "advanced_summary":
        field_order = response_dict.get("stepfun_field_order") or []
        _validate_stepfun_protocol_fields(fields, field_order, response_dict.get("raw_protocol", ""))
        return
    _validate_stepfun_legacy_fields(fields, response_dict.get("raw_protocol", ""))


def get_adapter() -> DeciderAdapter:
    return DeciderAdapter(
        protocol=DECIDER_PROTOCOL_STEPFUN_TSV,
        display_name="StepFun TSV adapter",
        build_messages=build_messages,
        parse_response=parse_response,
        validate_response=validate_response,
    )