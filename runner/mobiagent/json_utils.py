from __future__ import annotations

import json
import logging
import re


def load_json_from_text(raw_text):
    """Try to extract one JSON object from mixed model output."""
    if raw_text is None:
        return None
    if not isinstance(raw_text, str):
        raw_text = str(raw_text)

    text = raw_text.strip()

    def _try_load(candidate):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None

    parsed = _try_load(text)
    if parsed is not None:
        return parsed

    for pattern in [r"```json\s*([\s\S]*?)\s*```", r"```\s*([\s\S]*?)\s*```"]:
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            parsed = _try_load(match.group(1).strip())
            if parsed is not None:
                return parsed

    normalized = text.replace("…", "...")
    if normalized != text:
        parsed = _try_load(normalized)
        if parsed is not None:
            return parsed
        text = normalized

    start_idx = text.find('{')
    if start_idx != -1:
        brace_count = 0
        for idx in range(start_idx, len(text)):
            if text[idx] == '{':
                brace_count += 1
            elif text[idx] == '}':
                brace_count -= 1
                if brace_count == 0:
                    candidate = text[start_idx:idx + 1]
                    parsed = _try_load(candidate)
                    if parsed is not None:
                        return parsed

    return None


def robust_json_loads(raw_text):
    """Load one JSON object and raise a readable error if parsing fails."""
    parsed = load_json_from_text(raw_text)
    if parsed is None:
        logging.error("解析 response_str 失败")
        logging.error(f"原始内容: {str(raw_text)[:300]}...")
        raise ValueError("无法解析 JSON 响应: 响应格式不正确")
    return parsed