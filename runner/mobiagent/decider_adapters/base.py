from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class DeciderAdapter:
    protocol: str
    display_name: str
    build_messages: Callable[[str, list[str], str, bool, str], list[dict]]
    parse_response: Callable[[str], dict]
    validate_response: Callable[[dict, bool], None]