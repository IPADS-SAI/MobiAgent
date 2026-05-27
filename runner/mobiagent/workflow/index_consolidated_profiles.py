from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import textwrap
import unicodedata
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from runner.mobiagent.json_utils import robust_json_loads


DEFAULT_PROFILE_ROOT = Path(__file__).resolve().parent / "profile-consolidation" / "outputs"
DEFAULT_INDEX_DIR = Path(__file__).resolve().parent / "profile-consolidation" / "indexes"
DEFAULT_RUN_ROOT = Path(__file__).resolve().parent / "test-runs"
DEFAULT_API_KEY = os.getenv("MOBIAGENT_API_KEY", "")
DOMAINS = ("eat", "wear", "live", "travel", "work", "other")
ENTITY_TYPES = ("person", "place", "merchant")
ENTITY_TYPE_DESCRIPTIONS = {
    "person": "现实中的个人或联系人",
    "place": "现实地点、地址、场馆或地理位置",
    "merchant": "提供商品或服务的商户、门店或商业品牌",
}
DOMAIN_DESCRIPTIONS = {
    "eat": "吃：餐饮、外卖、食品和饮料购买",
    "wear": "穿：服装、鞋包和穿戴用品",
    "live": "住：住房、住宿、酒店和家居",
    "travel": "行：交通、出行和旅行移动",
    "work": "办公：工作、会议、协作和办公事务",
    "other": "无法归入以上分类的事件",
}
DOMAIN_DISPLAY_NAMES = {
    "eat": "吃",
    "wear": "穿",
    "live": "住",
    "travel": "行",
    "work": "办公",
    "other": "其他",
}
ENTITY_TYPE_DISPLAY_NAMES = {
    "person": "联系人",
    "place": "地点",
    "merchant": "商户",
}
DOMAIN_RULES = {
    "eat": ("餐饮", "美食", "外卖", "用餐", "咖啡", "饮料", "餐厅", "食品", "早餐", "午餐", "晚餐"),
    "wear": ("服装", "衣服", "裤", "裙", "鞋", "外套", "穿搭", "箱包", "背包"),
    "live": ("住宿", "酒店", "宾馆", "房租", "住房", "家居", "公寓", "民宿"),
    "travel": ("交通", "出行", "通勤", "乘车", "网约车", "出租车", "打车", "地铁", "公交", "高铁", "火车", "航班", "机票", "旅行", "旅游"),
    "work": ("工作", "上班", "办公", "会议", "汇报", "协作", "加班", "出差"),
}

ENRICH_SYSTEM_PROMPT = textwrap.dedent(
    """
    你是个人事件索引助手。请为一条已经整理好的事件提取查询关键词，并分类到固定领域。
    返回严格 JSON：
    {
      "keywords": ["适合检索的关键词或短语"],
      "domains": ["eat | wear | live | travel | work | other"],
      "entities": [{"type": "person | place | merchant", "name": "实体名称", "aliases": ["正文中出现的别名"]}]
    }

    领域定义：
    - eat: 吃，餐饮、外卖、食品和饮料购买
    - wear: 穿，服装、鞋包和穿戴用品
    - live: 住，住房、住宿、酒店和家居
    - travel: 行，交通、出行和旅行移动
    - work: 办公，工作、会议、协作和办公事务
    - other: 无法归入以上分类

    领域判定基于事件所描述的商品、服务或活动语义，不要求正文含有“餐饮”“出行”等类别字样。
    餐食、饮品或用餐订单归入 eat；客运接送、拼车/合乘、叫车、车票/机票或旅行服务归入 travel；
    住宿预订归入 live；服装鞋包购买归入 wear；工作会议与办公协作归入 work。
    如果事件不属于以上任何一个领域，归入 other。
    除非一条事件属于 other，否则它可属于多个领域。不要新增领域。

    实体是事件正文中明确涉及的现实对象，只支持：
    - person: 现实中的个人或联系人
    - place: 现实地点、地址、场馆或地理位置
    - merchant: 提供商品或服务的商户、门店或商业品牌
    实体识别依据专名在事件中承担的现实对象角色，不要求文本显式写出实体类别。
    简短标题、列表项或省略谓词的描述中，若专名明显表示联系人、地点或商品/服务提供方，也应提取。
    person 必须是可用于再次识别同一人的姓名或昵称，叙述主体的泛称或角色称谓不构成 person。
    merchant 必须是商品或服务提供方的专名；复合标题中的商品、权益或交易描述不构成 merchant。
    若复合标题中包含可识别的提供方专名与附随描述，只提取提供方专名，不因附随描述存在而遗漏实体。
    name 保留正文中能指向该对象的完整名称；aliases 只包含正文明确支持且可单独指向同一对象的其他写法。
    名称叠加商品、权益或交易描述形成的扩展片段不是别名。
    不要把商品或服务类别、活动、状态、金额、时间或一般描述性词组提取为实体。
    不明确指向现实对象的词不要提取；没有实体时返回空列表。不要新增实体类别。
    """
).strip()

ENTITY_RESOLUTION_SYSTEM_PROMPT = textwrap.dedent(
    """
    你是个人事件实体归并助手。输入是同一种类别下、从不同事件提取出的实体候选。
    将明确指向同一现实对象的候选合并，以便跨来源查询；仅在证据充分时合并，不要因名称相近而猜测。
    返回严格 JSON：{"entities": [{"name": "规范显示名", "aliases": ["同一对象的其他候选名称"]}]}。
    输出名称和别名必须来自输入候选，不要补充输入中不存在的名称，不要输出 markdown 或解释。
    """
).strip()

SUMMARY_SYSTEM_PROMPT = textwrap.dedent(
    """
    你是个人事件摘要助手。根据输入的一组事件或下级摘要，生成简短、可检索的中文概括。
    只返回严格 JSON：{"summary": "摘要文本"}。
    摘要应陈述事实，不猜测，不遗漏主要活动类型，不输出 markdown。
    summary 最多 160 个汉字，只概括主要活动类型和显著事件，不要逐条复述输入。
    """
).strip()


def parse_iso_date(raw: str) -> date:
    return datetime.strptime(raw, "%Y-%m-%d").date()


def normalize_free_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_lookup_text(value: Any) -> str:
    text = normalize_free_text(value).lower()
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", text)


def normalize_entity_name(value: Any) -> str:
    return normalize_free_text(unicodedata.normalize("NFKC", str(value or ""))).strip(" \t\r\n\"'“”")


def entity_name_token(value: Any) -> str:
    return normalize_entity_name(value).casefold()


def entity_lookup_key(entity_type: str, name: Any) -> str:
    return f"{entity_type}:{entity_name_token(name)}"


def entity_display_key(entity_type: str, name: Any) -> str:
    return f"{entity_type}:{normalize_entity_name(name)}"


def parse_entity_selector(raw: str) -> tuple[str, str]:
    entity_type, separator, name = str(raw).partition(":")
    entity_type = entity_type.strip().lower()
    name = normalize_entity_name(name)
    if not separator or entity_type not in ENTITY_TYPES or not name:
        allowed = ", ".join(ENTITY_TYPES)
        raise ValueError(f"Entity selector must be type:name with type in {{{allowed}}}: {raw!r}")
    return entity_type, name


def stable_unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = normalize_free_text(value)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def source_display_name(value: Any) -> str:
    name = Path(normalize_free_text(value).split("#", 1)[0]).name
    for suffix in (".state.json", ".md"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    name = name.rsplit("__", 1)[-1]
    name = re.sub(r"^\d+_(?:basic_gui_)?task_", "", name)
    return normalize_free_text(name)


def event_source_names(source_refs: Iterable[str], source_state_file: str) -> list[str]:
    references = list(source_refs)
    candidates = references or [source_state_file]
    return stable_unique(source_display_name(candidate) for candidate in candidates)


def path_after_directory(value: Any, directory_name: str) -> str | None:
    parts = Path(normalize_free_text(value)).parts
    positions = [index for index, part in enumerate(parts) if part == directory_name]
    if not positions:
        return None
    suffix = parts[positions[-1] + 1 :]
    return Path(*suffix).as_posix() if suffix else None


def local_image_path(value: Any, run_summary_path: Path, run_root: Path) -> str | None:
    path_text = normalize_free_text(value)
    if not path_text:
        return None
    run_name = run_summary_path.parent.name
    marker = f"/{run_name}/"
    normalized = path_text.replace("\\", "/")
    if marker in normalized:
        relative = f"{run_name}/{normalized.split(marker, 1)[1]}"
        local_path = (run_root / relative).resolve()
        return str(local_path) if local_path.is_file() else None
    path = Path(path_text).expanduser()
    return str(path.resolve()) if path.is_file() else None


def daily_log_entries(path: Path) -> dict[int, str]:
    text = path.read_text(encoding="utf-8")
    if "-->" in text:
        text = text.split("-->", 1)[1]
    matches = re.findall(r"(?ms)^(\d+)\.\s+(.*?)(?=\n\n\d+\.\s|\Z)", text)
    return {int(number): normalize_free_text(summary) for number, summary in matches}


def summary_output_text(output: dict[str, Any]) -> str:
    structured = output.get("structured_output", {})
    if isinstance(structured, dict) and isinstance(structured.get("summary"), str):
        return normalize_free_text(structured["summary"])
    response = output.get("response", "")
    return normalize_free_text(response) if isinstance(response, str) else ""


def build_image_reference_map(run_root: Path) -> dict[str, list[str]]:
    daily_root = run_root / "daily-log"
    if not daily_root.exists():
        return {}
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for summary_path in sorted(run_root.glob("*/run_summary.json")):
        payload = read_json(summary_path, {})
        steps = payload.get("steps", {}) if isinstance(payload, dict) else {}
        for step in steps.values() if isinstance(steps, dict) else []:
            output = step.get("output", {}) if isinstance(step, dict) else {}
            if (
                not isinstance(output, dict)
                or output.get("tool_name") != "vlm_qa"
                or output.get("mode") != "summary"
            ):
                continue
            log_relative = path_after_directory(output.get("daily_log_path"), "daily-log")
            image_path = local_image_path(output.get("image"), summary_path, run_root)
            summary = summary_output_text(output)
            if log_relative and image_path and summary:
                candidates[log_relative].append({"summary": summary, "image_path": image_path, "used": False})

    image_references: dict[str, list[str]] = {}
    for log_path in sorted(daily_root.glob("*/*.md")):
        relative = log_path.relative_to(daily_root).as_posix()
        available = candidates.get(relative, [])
        for entry_number, text in daily_log_entries(log_path).items():
            matched = next(
                (
                    candidate
                    for candidate in available
                    if not candidate["used"] and candidate["summary"] == text
                ),
                None,
            )
            if matched is not None:
                matched["used"] = True
                image_references[f"{relative}#{entry_number}"] = [matched["image_path"]]
    return image_references


def hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def hash_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hash_bytes(encoded)


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logging.warning("Failed to read JSON file %s: %s", path, exc)
        return default


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sync_json_directory(directory: Path, desired: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, payload in desired.items():
        write_json_atomic(directory / name, payload)
    for existing in directory.glob("*.json"):
        if existing.name not in desired:
            existing.unlink()


class ModelContext:
    def __init__(self, args: argparse.Namespace) -> None:
        self.disable_model = bool(args.disable_model)
        self.model_name = str(args.model_name or "")
        self.client: Any | None = None
        self.model_calls = 0
        self.model_fallbacks = 0
        if not self.disable_model:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("The build command needs the openai package unless --disable_model is used.") from exc
            self.client = OpenAI(
                api_key=args.api_key,
                base_url=f"http://{args.service_ip}:{args.model_port}/v1",
            )

    def resolve_model_name(self) -> str:
        if self.model_name.strip():
            return self.model_name.strip()
        if self.client is None:
            raise RuntimeError("Model client is disabled")
        response = self.client.models.list()
        models = list(getattr(response, "data", []) or [])
        model_id = getattr(models[0], "id", "") if models else ""
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("No usable model was returned by /v1/models; pass --model_name explicitly.")
        self.model_name = model_id.strip()
        return self.model_name

    def call_json(self, system_prompt: str, user_prompt: str, max_tokens: int) -> dict[str, Any]:
        if self.client is None:
            raise RuntimeError("Model client is disabled")
        self.model_calls += 1
        response_text = self.client.chat.completions.create(
            model=self.resolve_model_name(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=max_tokens,
            timeout=60,
        ).choices[0].message.content
        return robust_json_loads(response_text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and query indexes derived from consolidated profile records")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Create or incrementally update indexes")
    build.add_argument("--input_root", "--input-root", default=str(DEFAULT_PROFILE_ROOT), help="Root containing .md.state.json files")
    build.add_argument("--index_dir", "--index-dir", default=str(DEFAULT_INDEX_DIR), help="Directory to store derived indexes")
    build.add_argument("--run_root", "--run-root", default=str(DEFAULT_RUN_ROOT), help="Workflow output root containing run_summary.json and daily-log files")
    build.add_argument("--service_ip", "--service-ip", default="localhost", help="OpenAI-compatible model service IP")
    build.add_argument("--model_port", "--model-port", type=int, default=8000, help="OpenAI-compatible model service port")
    build.add_argument("--model_name", "--model-name", default="", help="Model name; defaults to the endpoint's first model")
    build.add_argument("--api_key", "--api-key", default=DEFAULT_API_KEY, help="API key for the model endpoint")
    build.add_argument("--disable_model", "--disable-model", action="store_true", help="Build with deterministic rule-based enrichment")
    build.add_argument("--force_rebuild", "--force-rebuild", action="store_true", help="Re-enrich all events and summaries")
    build.add_argument("--log_level", "--log-level", default="INFO", help="Logging level")

    query = subparsers.add_parser("query", help="Query existing local indexes without model calls")
    query.add_argument("--index_dir", "--index-dir", default=str(DEFAULT_INDEX_DIR), help="Directory containing derived indexes")
    query.add_argument("--keyword", "--keywords", dest="keywords", action="append", default=[], help="Keyword; repeat for AND matching")
    query.add_argument("--domain", action="append", choices=DOMAINS, default=[], help="Domain; repeat for OR matching")
    query.add_argument("--entity", dest="entities", action="append", default=[], help="Entity selector type:name; repeat for OR matching")
    query.add_argument("--date", dest="event_date", help="Select one event date, YYYY-MM-DD")
    query.add_argument("--month", help="Select one month, YYYY-MM")
    query.add_argument("--start_date", "--start-date", help="Inclusive lower date bound, YYYY-MM-DD")
    query.add_argument("--end_date", "--end-date", help="Inclusive upper date bound, YYYY-MM-DD")
    query.add_argument("--include_events", "--include-events", action="store_true", help="Attach leaf events to temporal summaries")
    query.add_argument("--limit", type=int, default=20, help="Maximum returned event rows")
    query.add_argument("--json", dest="json_output", action="store_true", help="Print raw JSON for programmatic use")
    query.add_argument("--log_level", "--log-level", default="INFO", help="Logging level")
    return parser


def configure_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def validate_query_args(args: argparse.Namespace) -> None:
    temporal_selectors = int(bool(args.event_date)) + int(bool(args.month)) + int(bool(args.start_date or args.end_date))
    if temporal_selectors > 1:
        raise ValueError("Use only one of --date, --month, or --start_date/--end_date")
    if args.event_date:
        parse_iso_date(args.event_date)
    if args.month and not re.fullmatch(r"\d{4}-\d{2}", args.month):
        raise ValueError("--month must be in YYYY-MM format")
    if args.start_date:
        parse_iso_date(args.start_date)
    if args.end_date:
        parse_iso_date(args.end_date)
    if args.start_date and args.end_date and parse_iso_date(args.end_date) < parse_iso_date(args.start_date):
        raise ValueError("--end_date must be greater than or equal to --start_date")
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    for selector in getattr(args, "entities", []):
        parse_entity_selector(selector)


def manifest_enrichment_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.disable_model:
        return {"mode": "heuristic"}
    return {
        "mode": "model",
        "service_ip": args.service_ip,
        "model_port": args.model_port,
        "model_name": args.model_name,
    }


def load_existing_events(index_dir: Path) -> dict[str, dict[str, Any]]:
    events: dict[str, dict[str, Any]] = {}
    event_dir = index_dir / "events"
    if not event_dir.exists():
        return events
    for path in sorted(event_dir.glob("*.json")):
        payload = read_json(path, {})
        for event in payload.get("events", []) if isinstance(payload, dict) else []:
            if isinstance(event, dict) and isinstance(event.get("event_id"), str):
                events[event["event_id"]] = event
    return events


def load_temporal_documents(index_dir: Path) -> dict[str, dict[str, Any]]:
    documents: dict[str, dict[str, Any]] = {}
    temporal_dir = index_dir / "temporal"
    if not temporal_dir.exists():
        return documents
    for path in sorted(temporal_dir.glob("*.json")):
        payload = read_json(path, {})
        if isinstance(payload, dict) and isinstance(payload.get("month"), str):
            documents[payload["month"]] = payload
    return documents


def base_events_from_state(
    path: Path,
    input_root: Path,
    image_references: dict[str, list[str]],
) -> list[dict[str, Any]]:
    payload = read_json(path, {})
    records = payload.get("records", []) if isinstance(payload, dict) else []
    relative_path = path.relative_to(input_root).as_posix()
    events: list[dict[str, Any]] = []
    if not isinstance(records, list):
        logging.warning("Ignoring state file without records list: %s", path)
        return events
    for record in records:
        if not isinstance(record, dict):
            continue
        record_id = record.get("record_id")
        summary = normalize_free_text(record.get("summary") or record.get("normalized_fact"))
        normalized_fact = normalize_free_text(record.get("normalized_fact"))
        event_date = normalize_free_text(record.get("first_seen_date"))
        if not str(record_id).isdigit() or not summary or not normalized_fact:
            logging.warning("Ignoring incomplete consolidated record in %s: %r", path, record_id)
            continue
        try:
            event_date = parse_iso_date(event_date).isoformat()
        except ValueError:
            logging.warning("Ignoring consolidated record with invalid date in %s: %r", path, record_id)
            continue
        last_seen_date = normalize_free_text(record.get("last_seen_date")) or event_date
        try:
            last_seen_date = parse_iso_date(last_seen_date).isoformat()
        except ValueError:
            last_seen_date = event_date
        tags = stable_unique(str(item) for item in record.get("tags", []) if isinstance(item, str))
        source_refs = stable_unique(str(item) for item in record.get("source_refs", []) if isinstance(item, str))
        events.append(
            {
                "event_id": f"{relative_path}#{int(record_id)}",
                "source_state_file": relative_path,
                "record_id": int(record_id),
                "event_date": event_date,
                "last_seen_date": last_seen_date,
                "summary": summary,
                "normalized_fact": normalized_fact,
                "tags": tags,
                "source_refs": source_refs,
                "sources": event_source_names(source_refs, relative_path),
                "image_paths": stable_unique(
                    image_path
                    for source_ref in source_refs
                    for image_path in image_references.get(source_ref, [])
                ),
            }
        )
    return events


def derive_rule_keywords(event: dict[str, Any]) -> list[str]:
    values: list[str] = []
    text = f"{event['summary']} {event['normalized_fact']}"
    for domain_terms in DOMAIN_RULES.values():
        for term in domain_terms:
            if term.lower() in text.lower():
                values.append(term)
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_.-]*|\d+(?:\.\d+)?元?", text):
        if len(token) > 1:
            values.append(token)
    for phrase in re.split(r"[，。；、：:（）()“”\"'\s]+", text):
        phrase = phrase.strip()
        if 2 <= len(phrase) <= 18 and not phrase.startswith(("当前页面", "这张截图", "记录中")):
            values.append(phrase)
    return stable_unique(values)[:30]


def derive_rule_domains(event: dict[str, Any]) -> list[str]:
    text = " ".join([event["summary"], event["normalized_fact"]]).lower()
    result = [domain for domain, terms in DOMAIN_RULES.items() if any(term.lower() in text for term in terms)]
    return result or ["other"]


def normalize_domains(raw_domains: Any) -> list[str]:
    if not isinstance(raw_domains, list):
        return []
    result = stable_unique(str(item).lower() for item in raw_domains if str(item).lower() in DOMAINS)
    if len(result) > 1 and "other" in result:
        result.remove("other")
    return result


def unique_entity_names(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        name = normalize_entity_name(value)
        token = entity_name_token(name)
        if name and token not in seen:
            seen.add(token)
            result.append(name)
    return result


def normalize_entity_mentions(raw_entities: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_entities, list):
        return []
    mentions: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for raw_entity in raw_entities:
        if not isinstance(raw_entity, dict):
            continue
        entity_type = normalize_free_text(raw_entity.get("type")).lower()
        name = normalize_entity_name(raw_entity.get("name"))
        token = entity_name_token(name)
        if entity_type not in ENTITY_TYPES or not token or (entity_type, token) in seen:
            continue
        aliases = unique_entity_names(raw_entity.get("aliases", []) if isinstance(raw_entity.get("aliases"), list) else [])
        aliases = [alias for alias in aliases if entity_name_token(alias) != token]
        mentions.append({"type": entity_type, "name": name, "aliases": aliases})
        seen.add((entity_type, token))
    return mentions


def derive_rule_entities(event: dict[str, Any]) -> list[dict[str, Any]]:
    text = " ".join([event["summary"], event["normalized_fact"]])
    candidates: list[dict[str, Any]] = []
    labeled_patterns = {
        "person": r"(?:人物|联系人姓名|联系人|聊天对象|姓名)\s*[:：]\s*[“\"']?([^，。；;\n“”\"']{2,40})",
        "place": r"(?:地点|位置|目的地|地址|场馆)\s*[:：]\s*[“\"']?([^，。；;\n“”\"']{2,60})",
        "merchant": r"(?:商户|商家|门店|店铺)\s*[:：]\s*[“\"']?([^，。；;\n“”\"']{2,60})",
    }
    for entity_type, pattern in labeled_patterns.items():
        for match in re.finditer(pattern, text):
            candidates.append({"type": entity_type, "name": match.group(1), "aliases": []})
    for match in re.finditer(r"(?:与|和)\s*[“\"']([^“”\"']{2,40})[”\"']\s*的?(?:聊天|对话)", text):
        candidates.append({"type": "person", "name": match.group(1), "aliases": []})
    return normalize_entity_mentions(candidates)


def enrich_event(event: dict[str, Any], model: ModelContext) -> dict[str, Any]:
    enriched = dict(event)
    rule_keywords = derive_rule_keywords(event)
    rule_domains = derive_rule_domains(event)
    rule_entities = derive_rule_entities(event)
    if model.disable_model:
        enriched["keywords"] = rule_keywords
        enriched["domains"] = rule_domains
        enriched["entity_mentions"] = rule_entities
        enriched["enrichment_method"] = "heuristic"
        return enriched

    prompt = json.dumps(
        {
            "summary": event["summary"],
            "normalized_fact": event["normalized_fact"],
        },
        ensure_ascii=False,
        indent=2,
    )
    try:
        payload = model.call_json(ENRICH_SYSTEM_PROMPT, prompt, max_tokens=800)
        model_keywords = payload.get("keywords", []) if isinstance(payload, dict) else []
        keywords = stable_unique(
            [str(item) for item in model_keywords if isinstance(item, str)] + rule_keywords
        )[:40]
        model_domains = normalize_domains(payload.get("domains", [])) if isinstance(payload, dict) else []
        domains = normalize_domains([*model_domains, *rule_domains])
        model_entities = normalize_entity_mentions(payload.get("entities", [])) if isinstance(payload, dict) else []
        enriched["keywords"] = keywords
        enriched["domains"] = domains
        enriched["entity_mentions"] = normalize_entity_mentions([*model_entities, *rule_entities])
        enriched["enrichment_method"] = "model"
    except Exception as exc:
        logging.warning("Falling back to heuristic event enrichment for %s: %s", event["event_id"], exc)
        model.model_fallbacks += 1
        enriched["keywords"] = rule_keywords
        enriched["domains"] = rule_domains
        enriched["entity_mentions"] = rule_entities
        enriched["enrichment_method"] = "heuristic_fallback"
    return enriched


def index_terms(value: Any) -> set[str]:
    compact = normalize_lookup_text(value)
    terms: set[str] = set()
    for run in re.findall(r"[\u4e00-\u9fff]+", compact):
        terms.update(run)
        terms.update(run[index : index + 2] for index in range(len(run) - 1))
    for token in re.findall(r"[a-z0-9_]+", compact):
        terms.update(token)
        terms.update(token[index : index + 2] for index in range(len(token) - 1))
    return {term for term in terms if term}


def event_search_text(event: dict[str, Any]) -> str:
    return " ".join(
        [
            event.get("summary", ""),
            event.get("normalized_fact", ""),
            *event.get("keywords", []),
        ]
    )


def build_inverted_document(events: dict[str, dict[str, Any]]) -> dict[str, Any]:
    postings: dict[str, set[str]] = defaultdict(set)
    for event_id, event in events.items():
        for term in index_terms(event_search_text(event)):
            postings[term].add(event_id)
    return {
        "normalization": "lowercase compact text with CJK character/bigram and alphanumeric character/bigram postings",
        "postings": {term: sorted(event_ids) for term, event_ids in sorted(postings.items())},
    }


def source_search_text(event: dict[str, Any]) -> str:
    return " ".join(event.get("sources", []))


def build_source_document(events: dict[str, dict[str, Any]]) -> dict[str, Any]:
    postings: dict[str, set[str]] = defaultdict(set)
    sources: dict[str, set[str]] = defaultdict(set)
    for event_id, event in events.items():
        for name in event.get("sources", []):
            sources[name].add(event_id)
        for term in index_terms(source_search_text(event)):
            postings[term].add(event_id)
    return {
        "normalization": "lowercase compact text with CJK character/bigram and alphanumeric character/bigram postings",
        "sources": {name: sorted(event_ids) for name, event_ids in sorted(sources.items())},
        "postings": {term: sorted(event_ids) for term, event_ids in sorted(postings.items())},
    }


def build_domain_document(events: dict[str, dict[str, Any]]) -> dict[str, Any]:
    postings: dict[str, list[str]] = {domain: [] for domain in DOMAINS}
    for event_id, event in sorted(events.items()):
        for domain in event.get("domains", ["other"]):
            if domain in postings:
                postings[domain].append(event_id)
    return {
        "domains": DOMAIN_DESCRIPTIONS,
        "postings": postings,
    }


def load_entity_document(index_dir: Path) -> dict[str, Any]:
    payload = read_json(index_dir / "entity" / "index.json", {})
    return payload if isinstance(payload, dict) else {}


def add_entity_definition(
    definitions: dict[str, dict[str, Any]],
    entity_type: str,
    name: str,
    aliases: Iterable[Any],
) -> str:
    key = entity_display_key(entity_type, name)
    definition = definitions.setdefault(
        key,
        {
            "type": entity_type,
            "name": normalize_entity_name(name),
            "aliases": [],
            "event_ids": [],
        },
    )
    definition["aliases"] = unique_entity_names(
        [*definition.get("aliases", []), *aliases]
    )
    definition["aliases"] = [
        alias for alias in definition["aliases"] if entity_name_token(alias) != entity_name_token(definition["name"])
    ]
    return key


def build_entity_document(events: dict[str, dict[str, Any]], model: ModelContext) -> dict[str, Any]:
    mentions_by_type: dict[str, dict[str, dict[str, Any]]] = {entity_type: {} for entity_type in ENTITY_TYPES}
    for event in events.values():
        event_mentions = normalize_entity_mentions(event.get("entity_mentions", []))
        event["entity_mentions"] = event_mentions
        for mention in event_mentions:
            token = entity_name_token(mention["name"])
            existing = mentions_by_type[mention["type"]].get(token)
            if existing is None:
                mentions_by_type[mention["type"]][token] = dict(mention)
            else:
                existing["aliases"] = unique_entity_names([*existing.get("aliases", []), *mention.get("aliases", [])])

    assignments: dict[tuple[str, str], str] = {}
    definitions: dict[str, dict[str, Any]] = {}
    for entity_type, mention_map in mentions_by_type.items():
        if not mention_map:
            continue
        source_display: dict[str, str] = {}
        for mention in mention_map.values():
            for name in [mention["name"], *mention.get("aliases", [])]:
                source_display.setdefault(entity_name_token(name), normalize_entity_name(name))

        if not model.disable_model:
            prompt = json.dumps(
                {
                    "type": entity_type,
                    "mentions": list(mention_map.values()),
                },
                ensure_ascii=False,
                indent=2,
            )
            try:
                payload = model.call_json(ENTITY_RESOLUTION_SYSTEM_PROMPT, prompt, max_tokens=800)
                raw_entities = payload.get("entities", []) if isinstance(payload, dict) else []
                if not isinstance(raw_entities, list):
                    raw_entities = []
                resolved_tokens: dict[str, set[str]] = defaultdict(set)
                for raw_entity in raw_entities:
                    if not isinstance(raw_entity, dict):
                        continue
                    canonical_token = entity_name_token(raw_entity.get("name"))
                    if canonical_token not in source_display:
                        continue
                    aliases = raw_entity.get("aliases", []) if isinstance(raw_entity.get("aliases"), list) else []
                    covered_tokens = {
                        token
                        for token in [canonical_token, *(entity_name_token(alias) for alias in aliases)]
                        if token in source_display
                    }
                    key = add_entity_definition(
                        definitions,
                        entity_type,
                        source_display[canonical_token],
                        [source_display[token] for token in covered_tokens if token != canonical_token],
                    )
                    for token in covered_tokens:
                        resolved_tokens[token].add(key)
                for token in mention_map:
                    keys = resolved_tokens.get(token, set())
                    if len(keys) == 1:
                        assignments[(entity_type, token)] = next(iter(keys))
            except Exception as exc:
                logging.warning("Falling back to surface entity matching for %s: %s", entity_type, exc)
                model.model_fallbacks += 1

        for token, mention in mention_map.items():
            typed_token = (entity_type, token)
            if typed_token not in assignments:
                assignments[typed_token] = add_entity_definition(definitions, entity_type, mention["name"], [])

    for event_id, event in sorted(events.items()):
        entity_keys: list[str] = []
        for mention in normalize_entity_mentions(event.get("entity_mentions", [])):
            key = assignments.get((mention["type"], entity_name_token(mention["name"])))
            if key and key not in entity_keys:
                entity_keys.append(key)
                definitions[key]["event_ids"].append(event_id)
        event["entities"] = entity_keys

    alias_postings: dict[str, list[str]] = defaultdict(list)
    for key, definition in sorted(definitions.items()):
        definition["event_ids"] = sorted(set(definition["event_ids"]))
        if not definition["event_ids"]:
            continue
        for name in [definition["name"], *definition["aliases"]]:
            lookup = entity_lookup_key(definition["type"], name)
            if key not in alias_postings[lookup]:
                alias_postings[lookup].append(key)
    return {
        "types": list(ENTITY_TYPES),
        "type_descriptions": ENTITY_TYPE_DESCRIPTIONS,
        "entities": {
            key: definition for key, definition in sorted(definitions.items()) if definition["event_ids"]
        },
        "alias_postings": {lookup: sorted(keys) for lookup, keys in sorted(alias_postings.items())},
    }


def truncate_text(value: str, max_length: int = 360) -> str:
    text = normalize_free_text(value)
    return text if len(text) <= max_length else text[:max_length] + "..."


def heuristic_summary(level: str, items: list[str], event_count: int) -> str:
    prefix = {"day": "当天", "week": "本周", "month": "本月"}[level]
    details = "；".join(truncate_text(item, 90) for item in items if item)
    return truncate_text(f"{prefix}共{event_count}条事件：" + (details or "无事件"))


def generate_summary(
    level: str,
    items: list[str],
    event_count: int,
    input_hash: str,
    previous: dict[str, Any] | None,
    model: ModelContext,
) -> tuple[dict[str, Any], bool]:
    if isinstance(previous, dict) and previous.get("input_hash") == input_hash and previous.get("text"):
        return previous, False
    if model.disable_model:
        return {
            "text": heuristic_summary(level, items, event_count),
            "input_hash": input_hash,
            "generation": "heuristic",
        }, True
    prompt = json.dumps(
        {"level": level, "event_count": event_count, "items": [truncate_text(item, 280) for item in items]},
        ensure_ascii=False,
    )
    try:
        payload = model.call_json(SUMMARY_SYSTEM_PROMPT, prompt, max_tokens=500)
        text = normalize_free_text(payload.get("summary", "")) if isinstance(payload, dict) else ""
        if not text:
            raise ValueError("Model returned an empty summary")
        return {"text": text, "input_hash": input_hash, "generation": "model"}, True
    except Exception as exc:
        logging.warning("Falling back to heuristic %s summary: %s", level, exc)
        model.model_fallbacks += 1
        return {
            "text": heuristic_summary(level, items, event_count),
            "input_hash": input_hash,
            "generation": "heuristic_fallback",
        }, True


def week_group_key(raw_date: str) -> str:
    day = parse_iso_date(raw_date)
    return (day - timedelta(days=day.weekday())).isoformat()


def build_temporal_documents(
    events: dict[str, dict[str, Any]],
    previous_documents: dict[str, dict[str, Any]],
    model: ModelContext,
) -> tuple[dict[str, dict[str, Any]], int]:
    events_by_month: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events.values():
        events_by_month[event["event_date"][:7]].append(event)

    documents: dict[str, dict[str, Any]] = {}
    rebuilt_summaries = 0
    for month, month_events in sorted(events_by_month.items()):
        month_events.sort(key=lambda item: (item["event_date"], item["event_id"]))
        previous_doc = previous_documents.get(month, {})
        previous_days = {
            item.get("date"): item
            for week in previous_doc.get("weeks", [])
            if isinstance(week, dict)
            for item in week.get("days", [])
            if isinstance(item, dict)
        }
        previous_weeks = {item.get("week_id"): item for item in previous_doc.get("weeks", []) if isinstance(item, dict)}
        daily_events: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for event in month_events:
            daily_events[event["event_date"]].append(event)

        day_nodes: list[dict[str, Any]] = []
        for day, items in sorted(daily_events.items()):
            event_ids = [item["event_id"] for item in items]
            digest = hash_json(
                [{"event_id": item["event_id"], "summary": item["summary"], "fact": item["normalized_fact"]} for item in items]
            )
            previous_summary = previous_days.get(day, {}).get("summary")
            summary, rebuilt = generate_summary(
                "day",
                [item["summary"] or item["normalized_fact"] for item in items],
                len(items),
                digest,
                previous_summary,
                model,
            )
            rebuilt_summaries += int(rebuilt)
            day_nodes.append({"date": day, "event_ids": event_ids, "summary": summary})

        weekly_days: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for node in day_nodes:
            weekly_days[week_group_key(node["date"])].append(node)
        week_nodes: list[dict[str, Any]] = []
        for week_id, days in sorted(weekly_days.items()):
            event_ids = [event_id for day_node in days for event_id in day_node["event_ids"]]
            digest = hash_json(
                [{"date": day_node["date"], "summary": day_node["summary"]["text"], "event_ids": day_node["event_ids"]} for day_node in days]
            )
            previous_summary = previous_weeks.get(week_id, {}).get("summary")
            summary, rebuilt = generate_summary(
                "week",
                [day_node["summary"]["text"] for day_node in days],
                len(event_ids),
                digest,
                previous_summary,
                model,
            )
            rebuilt_summaries += int(rebuilt)
            monday = parse_iso_date(week_id)
            month_first = parse_iso_date(month + "-01")
            next_month = (month_first.replace(day=28) + timedelta(days=4)).replace(day=1)
            month_last = next_month - timedelta(days=1)
            week_nodes.append(
                {
                    "week_id": week_id,
                    "start_date": max(monday, month_first).isoformat(),
                    "end_date": min(monday + timedelta(days=6), month_last).isoformat(),
                    "event_ids": event_ids,
                    "summary": summary,
                    "days": days,
                }
            )

        all_event_ids = [event["event_id"] for event in month_events]
        month_digest = hash_json(
            [{"week_id": node["week_id"], "summary": node["summary"]["text"], "event_ids": node["event_ids"]} for node in week_nodes]
        )
        month_summary, rebuilt = generate_summary(
            "month",
            [node["summary"]["text"] for node in week_nodes],
            len(all_event_ids),
            month_digest,
            previous_doc.get("summary") if isinstance(previous_doc, dict) else None,
            model,
        )
        rebuilt_summaries += int(rebuilt)
        documents[month] = {
            "month": month,
            "event_ids": all_event_ids,
            "summary": month_summary,
            "weeks": week_nodes,
        }
    return documents, rebuilt_summaries


def event_partitions(events: dict[str, dict[str, Any]]) -> dict[str, Any]:
    partitions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events.values():
        partitions[event["event_date"][:7]].append(event)
    return {
        f"{month}.json": {
            "month": month,
            "events": sorted(values, key=lambda item: (item["event_date"], item["event_id"])),
        }
        for month, values in sorted(partitions.items())
    }


def run_build(args: argparse.Namespace) -> dict[str, Any]:
    input_root = Path(args.input_root).expanduser().resolve()
    index_dir = Path(args.index_dir).expanduser().resolve()
    run_root = Path(getattr(args, "run_root", DEFAULT_RUN_ROOT)).expanduser().resolve()
    if not input_root.exists():
        raise ValueError(f"Input root does not exist: {input_root}")
    state_paths = sorted(input_root.rglob("*.md.state.json"))
    previous_manifest = read_json(index_dir / "manifest.json", {})
    previous_events = load_existing_events(index_dir)
    config = manifest_enrichment_config(args)
    reuse_allowed = (
        not args.force_rebuild
        and isinstance(previous_manifest, dict)
        and previous_manifest.get("input_root") == str(input_root)
        and previous_manifest.get("enrichment") == config
        and previous_manifest.get("entity_types") == list(ENTITY_TYPES)
    )
    previous_sources = previous_manifest.get("sources", {}) if reuse_allowed else {}
    model = ModelContext(args)
    image_references = build_image_reference_map(run_root)
    events: dict[str, dict[str, Any]] = {}
    sources: dict[str, dict[str, Any]] = {}
    reused_events = 0
    enriched_events = 0

    for state_path in state_paths:
        relative_path = state_path.relative_to(input_root).as_posix()
        digest = hash_bytes(state_path.read_bytes())
        previous_source = previous_sources.get(relative_path, {}) if isinstance(previous_sources, dict) else {}
        expected_ids = previous_source.get("event_ids", []) if isinstance(previous_source, dict) else []
        can_reuse = (
            reuse_allowed
            and previous_source.get("sha256") == digest
            and expected_ids
            and all(event_id in previous_events for event_id in expected_ids)
        )
        if can_reuse:
            source_event_ids = list(expected_ids)
            for event_id in source_event_ids:
                event = dict(previous_events[event_id])
                event["sources"] = event_source_names(
                    event.get("source_refs", []),
                    event.get("source_state_file", relative_path),
                )
                event["image_paths"] = stable_unique(
                    image_path
                    for source_ref in event.get("source_refs", [])
                    for image_path in image_references.get(source_ref, [])
                )
                events[event_id] = event
            reused_events += len(source_event_ids)
        else:
            source_event_ids = []
            for base_event in base_events_from_state(state_path, input_root, image_references):
                event = enrich_event(base_event, model)
                events[event["event_id"]] = event
                source_event_ids.append(event["event_id"])
                enriched_events += 1
        sources[relative_path] = {"sha256": digest, "event_ids": sorted(source_event_ids)}

    current_ids = set(events)
    previous_ids = set(previous_events)
    previous_entities = load_entity_document(index_dir) if reuse_allowed else {}
    if (
        reuse_allowed
        and enriched_events == 0
        and current_ids == previous_ids
        and previous_entities.get("types") == list(ENTITY_TYPES)
    ):
        entity_document = previous_entities
    else:
        entity_document = build_entity_document(events, model)
    previous_temporal = load_temporal_documents(index_dir) if reuse_allowed else {}
    temporal_documents, rebuilt_summaries = build_temporal_documents(events, previous_temporal, model)
    changed_ids = {
        event_id for event_id in current_ids & previous_ids if hash_json(events[event_id]) != hash_json(previous_events[event_id])
    }

    sync_json_directory(index_dir / "events", event_partitions(events))
    write_json_atomic(index_dir / "inverted" / "index.json", build_inverted_document(events))
    source_document = build_source_document(events)
    write_json_atomic(index_dir / "source" / "index.json", source_document)
    write_json_atomic(index_dir / "domain" / "index.json", build_domain_document(events))
    write_json_atomic(index_dir / "entity" / "index.json", entity_document)
    sync_json_directory(
        index_dir / "temporal",
        {f"{month}.json": payload for month, payload in temporal_documents.items()},
    )
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_root": str(input_root),
        "index_dir": str(index_dir),
        "run_root": str(run_root),
        "enrichment": config,
        "domains": list(DOMAINS),
        "entity_types": list(ENTITY_TYPES),
        "sources": sources,
        "event_count": len(events),
    }
    write_json_atomic(index_dir / "manifest.json", manifest)
    return {
        "command": "build",
        "input_root": str(input_root),
        "index_dir": str(index_dir),
        "source_files": len(state_paths),
        "total_events": len(events),
        "new_events": len(current_ids - previous_ids),
        "updated_events": len(changed_ids),
        "deleted_events": len(previous_ids - current_ids),
        "reused_events": reused_events,
        "enriched_events": enriched_events,
        "rebuilt_summary_nodes": rebuilt_summaries,
        "total_entities": len(entity_document.get("entities", {})),
        "total_sources": len(source_document.get("sources", {})),
        "events_with_images": sum(bool(event.get("image_paths")) for event in events.values()),
        "total_image_paths": len({image_path for event in events.values() for image_path in event.get("image_paths", [])}),
        "model_calls": model.model_calls,
        "model_fallbacks": model.model_fallbacks,
    }


def public_event(event: dict[str, Any]) -> dict[str, Any]:
    return {
        key: event.get(key)
        for key in (
            "event_id",
            "event_date",
            "last_seen_date",
            "summary",
            "normalized_fact",
            "domains",
            "keywords",
            "entities",
            "sources",
            "image_paths",
            "tags",
            "source_refs",
        )
    }


def indexed_text_match_ids(
    keyword: str,
    events: dict[str, dict[str, Any]],
    postings: dict[str, Any],
    text_getter: Any,
) -> set[str]:
    compact_keyword = normalize_lookup_text(keyword)
    if not compact_keyword:
        return set(events)
    terms = index_terms(compact_keyword)
    if not terms:
        return set()
    matching: set[str] | None = None
    for term in terms:
        term_ids = set(postings.get(term, []))
        matching = term_ids if matching is None else matching & term_ids
    return {
        event_id
        for event_id in (matching or set())
        if compact_keyword in normalize_lookup_text(text_getter(events[event_id]))
    }


def keyword_match_ids(
    keywords: list[str],
    events: dict[str, dict[str, Any]],
    inverted: dict[str, Any],
    source_document: dict[str, Any],
) -> set[str]:
    result = set(events)
    inverted_postings = inverted.get("postings", {}) if isinstance(inverted, dict) else {}
    source_postings = source_document.get("postings", {}) if isinstance(source_document, dict) else {}
    for keyword in keywords:
        content_ids = indexed_text_match_ids(keyword, events, inverted_postings, event_search_text)
        source_ids = indexed_text_match_ids(keyword, events, source_postings, source_search_text)
        result &= content_ids | source_ids
    return result


def entity_match_ids(selectors: list[str], index_dir: Path) -> set[str]:
    document = load_entity_document(index_dir)
    entity_definitions = document.get("entities", {}) if isinstance(document, dict) else {}
    alias_postings = document.get("alias_postings", {}) if isinstance(document, dict) else {}
    result: set[str] = set()
    for selector in selectors:
        entity_type, name = parse_entity_selector(selector)
        for entity_key in alias_postings.get(entity_lookup_key(entity_type, name), []):
            definition = entity_definitions.get(entity_key, {})
            result.update(definition.get("event_ids", []) if isinstance(definition, dict) else [])
    return result


def date_filtered_ids(args: argparse.Namespace, events: dict[str, dict[str, Any]]) -> set[str]:
    if args.event_date:
        return {event_id for event_id, event in events.items() if event["event_date"] == args.event_date}
    if args.month:
        return {event_id for event_id, event in events.items() if event["event_date"].startswith(args.month + "-")}
    if args.start_date or args.end_date:
        lower = args.start_date or "0000-01-01"
        upper = args.end_date or "9999-12-31"
        return {event_id for event_id, event in events.items() if lower <= event["event_date"] <= upper}
    return set(events)


def temporal_selection_ids(args: argparse.Namespace, index_dir: Path) -> set[str]:
    documents = load_temporal_documents(index_dir)
    if args.month:
        document = documents.get(args.month, {})
        return set(document.get("event_ids", []))
    if args.event_date:
        document = documents.get(args.event_date[:7], {})
        for week in document.get("weeks", []):
            for day in week.get("days", []):
                if day.get("date") == args.event_date:
                    return set(day.get("event_ids", []))
        return set()
    if args.start_date or args.end_date:
        lower = args.start_date or "0000-01-01"
        upper = args.end_date or "9999-12-31"
        return {
            event_id
            for document in documents.values()
            for week in document.get("weeks", [])
            for day in week.get("days", [])
            if lower <= day.get("date", "") <= upper
            for event_id in day.get("event_ids", [])
        }
    return set()


def filtered_event_ids(args: argparse.Namespace, events: dict[str, dict[str, Any]], index_dir: Path) -> set[str]:
    matching = date_filtered_ids(args, events)
    if args.keywords:
        inverted = read_json(index_dir / "inverted" / "index.json", {})
        source_document = read_json(index_dir / "source" / "index.json", {})
        matching &= keyword_match_ids(args.keywords, events, inverted, source_document)
    if args.domain:
        domain_document = read_json(index_dir / "domain" / "index.json", {})
        postings = domain_document.get("postings", {}) if isinstance(domain_document, dict) else {}
        domain_ids: set[str] = set()
        for domain in args.domain:
            domain_ids.update(postings.get(domain, []))
        matching &= domain_ids
    selectors = getattr(args, "entities", [])
    if selectors:
        matching &= entity_match_ids(selectors, index_dir)
    return matching


def sorted_events(event_ids: Iterable[str], events: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [events[event_id] for event_id in event_ids if event_id in events],
        key=lambda event: (event["event_date"], event["event_id"]),
    )


def attach_limited_events(
    temporal_payload: dict[str, Any],
    events: dict[str, dict[str, Any]],
    allowed_ids: set[str],
    limit: int,
) -> int:
    attached = 0
    for week in temporal_payload.get("weeks", []):
        for day in week.get("days", []):
            day_events = []
            for event_id in day.get("event_ids", []):
                if event_id in allowed_ids and attached < limit:
                    day_events.append(public_event(events[event_id]))
                    attached += 1
            day["events"] = day_events
    return attached


def strip_temporal_event_ids(payload: dict[str, Any]) -> None:
    payload.pop("event_ids", None)
    for week in payload.get("weeks", []):
        week.pop("event_ids", None)
        for day in week.get("days", []):
            day.pop("event_ids", None)


def query_temporal_summary(
    args: argparse.Namespace,
    events: dict[str, dict[str, Any]],
    index_dir: Path,
    matching_ids: set[str],
) -> dict[str, Any] | None:
    if args.keywords or args.domain or getattr(args, "entities", []):
        return None
    temporal = load_temporal_documents(index_dir)
    if args.month:
        document = temporal.get(args.month)
        if document is None:
            return {"result_type": "temporal_summary", "total_events": 0, "temporal": None}
        result = json.loads(json.dumps(document, ensure_ascii=False))
        if args.include_events:
            attached = attach_limited_events(result, events, matching_ids, args.limit)
            result["events_truncated"] = attached < len(matching_ids)
        strip_temporal_event_ids(result)
        return {"result_type": "temporal_summary", "total_events": len(matching_ids), "temporal": result}
    if args.event_date:
        document = temporal.get(args.event_date[:7], {})
        day = next(
            (
                day_node
                for week in document.get("weeks", [])
                for day_node in week.get("days", [])
                if day_node.get("date") == args.event_date
            ),
            None,
        )
        if day is None:
            return {"result_type": "temporal_day", "total_events": 0, "day": None}
        result = json.loads(json.dumps(day, ensure_ascii=False))
        if args.include_events:
            day_events = sorted_events(matching_ids, events)[: args.limit]
            result["events"] = [public_event(event) for event in day_events]
            result["events_truncated"] = len(day_events) < len(matching_ids)
        result.pop("event_ids", None)
        return {"result_type": "temporal_day", "total_events": len(matching_ids), "day": result}
    if args.start_date or args.end_date:
        days: list[dict[str, Any]] = []
        for document in temporal.values():
            for week in document.get("weeks", []):
                for day in week.get("days", []):
                    if any(event_id in matching_ids for event_id in day.get("event_ids", [])):
                        selected_day = json.loads(json.dumps(day, ensure_ascii=False))
                        selected_day.pop("event_ids", None)
                        days.append(selected_day)
        days.sort(key=lambda item: item["date"])
        result = {"result_type": "temporal_range", "total_events": len(matching_ids), "days": days}
        if args.include_events:
            selected_events = sorted_events(matching_ids, events)[: args.limit]
            result["events"] = [public_event(event) for event in selected_events]
            result["events_truncated"] = len(selected_events) < len(matching_ids)
        return result
    return None


def run_query(args: argparse.Namespace) -> dict[str, Any]:
    validate_query_args(args)
    index_dir = Path(args.index_dir).expanduser().resolve()
    manifest = read_json(index_dir / "manifest.json", None)
    if not isinstance(manifest, dict):
        raise ValueError(f"No built index found at: {index_dir}")
    query_description = {
        "keywords": args.keywords,
        "domains": args.domain,
        "entities": getattr(args, "entities", []),
        "date": args.event_date,
        "month": args.month,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "include_events": args.include_events,
        "limit": args.limit,
    }
    pure_temporal = not args.keywords and not args.domain and not getattr(args, "entities", []) and bool(
        args.event_date or args.month or args.start_date or args.end_date
    )
    if pure_temporal:
        matching_ids = temporal_selection_ids(args, index_dir)
        events = load_existing_events(index_dir) if args.include_events else {}
        temporal_result = query_temporal_summary(args, events, index_dir, matching_ids)
        if temporal_result is not None:
            return {"command": "query", "query": query_description, **temporal_result}
    events = load_existing_events(index_dir)
    matching_ids = filtered_event_ids(args, events, index_dir)
    results = sorted_events(matching_ids, events)
    return {
        "command": "query",
        "query": query_description,
        "result_type": "events",
        "total_events": len(results),
        "events": [public_event(event) for event in results[: args.limit]],
        "events_truncated": len(results) > args.limit,
    }


def display_width(value: str) -> int:
    return sum(2 if unicodedata.east_asian_width(character) in {"W", "F"} else 1 for character in value)


def truncate_display(value: Any, maximum_width: int) -> str:
    text = normalize_free_text(value) or "-"
    if display_width(text) <= maximum_width:
        return text
    suffix = "..."
    available = maximum_width - display_width(suffix)
    pieces: list[str] = []
    width = 0
    for character in text:
        character_width = display_width(character)
        if width + character_width > available:
            break
        pieces.append(character)
        width += character_width
    return "".join(pieces) + suffix


def format_terminal_table(
    headers: list[str],
    rows: list[list[Any]],
    maximum_widths: list[int],
) -> str:
    cells = [
        [truncate_display(value, maximum_widths[index]) for index, value in enumerate(row)]
        for row in rows
    ]
    widths = [
        max(
            display_width(header),
            max((display_width(row[index]) for row in cells), default=0),
        )
        for index, header in enumerate(headers)
    ]

    def divider() -> str:
        return "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def row_text(row: list[str]) -> str:
        padded = [
            value + (" " * (width - display_width(value)))
            for value, width in zip(row, widths)
        ]
        return "| " + " | ".join(padded) + " |"

    lines = [divider(), row_text(headers), divider()]
    lines.extend(row_text(row) for row in cells)
    lines.append(divider())
    return "\n".join(lines)


def display_domains(domains: Any) -> str:
    return "、".join(DOMAIN_DISPLAY_NAMES.get(domain, str(domain)) for domain in (domains or []))


def display_entities(entities: Any) -> str:
    formatted: list[str] = []
    for entity in entities or []:
        entity_type, separator, name = str(entity).partition(":")
        if separator:
            formatted.append(f"{ENTITY_TYPE_DISPLAY_NAMES.get(entity_type, entity_type)}:{name}")
        else:
            formatted.append(str(entity))
    return "、".join(formatted)


def temporal_summary_text(node: dict[str, Any]) -> str:
    summary = node.get("summary", "")
    return str(summary.get("text", "")) if isinstance(summary, dict) else str(summary)


def format_event_table(events: list[dict[str, Any]]) -> str:
    rows = [
        [
            index,
            event.get("event_date", ""),
            "、".join(event.get("sources", [])),
            event.get("summary", ""),
            display_domains(event.get("domains", [])),
            display_entities(event.get("entities", [])),
        ]
        for index, event in enumerate(events, start=1)
    ]
    lines = [format_terminal_table(
        ["#", "日期", "来源", "事件摘要", "领域", "实体"],
        rows,
        [4, 10, 24, 56, 18, 42],
    )]
    image_rows = [
        [index, image_path]
        for index, event in enumerate(events, start=1)
        for image_path in event.get("image_paths", [])
    ]
    if image_rows:
        lines.extend(
            [
                "",
                "原始图片路径:",
                format_terminal_table(["事件", "路径"], image_rows, [4, 180]),
            ]
        )
    return "\n".join(lines)


def appended_temporal_events(result: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    result_type = result.get("result_type")
    if result_type == "temporal_summary" and result.get("temporal"):
        temporal = result["temporal"]
        events = [
            event
            for week in temporal.get("weeks", [])
            for day in week.get("days", [])
            for event in day.get("events", [])
        ]
        return events, bool(temporal.get("events_truncated"))
    if result_type == "temporal_day" and result.get("day"):
        day = result["day"]
        return list(day.get("events", [])), bool(day.get("events_truncated"))
    return list(result.get("events", [])), bool(result.get("events_truncated"))


def format_temporal_result(result: dict[str, Any]) -> str:
    result_type = result["result_type"]
    total_events = result["total_events"]
    rows: list[list[Any]] = []
    if result_type == "temporal_summary":
        temporal = result.get("temporal")
        if temporal is None:
            return "匹配事件数: 0\n没有匹配的时间摘要。"
        rows.append(["月", temporal.get("month", ""), temporal_summary_text(temporal)])
        for week in temporal.get("weeks", []):
            period = f"{week.get('start_date', '')} ~ {week.get('end_date', '')}"
            rows.append(["周", period, temporal_summary_text(week)])
            rows.extend(["日", day.get("date", ""), temporal_summary_text(day)] for day in week.get("days", []))
    elif result_type == "temporal_day":
        day = result.get("day")
        if day is None:
            return "匹配事件数: 0\n没有匹配的时间摘要。"
        rows.append(["日", day.get("date", ""), temporal_summary_text(day)])
    else:
        rows.extend(["日", day.get("date", ""), temporal_summary_text(day)] for day in result.get("days", []))
        if not rows:
            return "匹配事件数: 0\n没有匹配的时间摘要。"

    lines = [
        f"匹配事件数: {total_events}",
        format_terminal_table(["层级", "时间", "摘要"], rows, [4, 23, 92]),
    ]
    events, truncated = appended_temporal_events(result)
    if events:
        label = f"关联事件（显示 {len(events)} 条"
        if truncated:
            label += f"，共 {total_events} 条"
        lines.extend(["", label + "）:", format_event_table(events)])
    return "\n".join(lines)


def format_query_result(result: dict[str, Any]) -> str:
    if result.get("result_type") != "events":
        return format_temporal_result(result)
    events = result.get("events", [])
    total_events = result.get("total_events", 0)
    if not events:
        return "匹配事件数: 0\n没有匹配的事件。"
    label = f"匹配事件数: {total_events}"
    if result.get("events_truncated"):
        label += f"（显示前 {len(events)} 条）"
    return "\n".join([label, format_event_table(events)])


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)
    try:
        result = run_build(args) if args.command == "build" else run_query(args)
    except (OSError, ValueError, RuntimeError) as exc:
        logging.error("%s", exc)
        return 1
    if args.command == "query" and not args.json_output:
        print(format_query_result(result))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
