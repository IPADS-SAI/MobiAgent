from __future__ import annotations

from .base import DeciderAdapter


DECIDER_PROTOCOL_QWEN_JSON = "qwen_json"
DECIDER_PROTOCOL_STEPFUN_TSV = "stepfun_tsv"
SUPPORTED_DECIDER_PROTOCOLS = (
    DECIDER_PROTOCOL_QWEN_JSON,
    DECIDER_PROTOCOL_STEPFUN_TSV,
)


def get_decider_adapter(decider_protocol: str) -> DeciderAdapter:
    if decider_protocol == DECIDER_PROTOCOL_QWEN_JSON:
        from .qwen import get_adapter

        return get_adapter()

    if decider_protocol == DECIDER_PROTOCOL_STEPFUN_TSV:
        from .stepfun import get_adapter

        return get_adapter()

    supported = ", ".join(SUPPORTED_DECIDER_PROTOCOLS)
    raise ValueError(f"Unsupported decider_protocol '{decider_protocol}'. Supported values: {supported}")