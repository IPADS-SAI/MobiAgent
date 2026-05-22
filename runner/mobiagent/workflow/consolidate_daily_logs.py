from __future__ import annotations

import argparse
import json
import logging
import os
import re
import textwrap
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from openai import OpenAI

from runner.mobiagent.json_utils import robust_json_loads
from runner.mobiagent.workflow.engine import DAILY_LOG_METADATA_PATTERN


CONSOLIDATED_METADATA_PATTERN = re.compile(r"\A<!-- CONSOLIDATED_PROFILE_METADATA\n(.*?)\n-->\n*", re.DOTALL)
CONSOLIDATED_RECORDS_PATTERN = re.compile(r"\A<!-- CONSOLIDATED_PROFILE_RECORDS\n(.*?)\n-->\n*", re.DOTALL)
NUMBERED_ENTRY_PATTERN = re.compile(r"(?ms)^(\d+)\.\s+(.*?)(?=^\d+\.\s|\Z)")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "profile-consolidation" / "outputs"
DEFAULT_INPUT_ROOT = Path(__file__).resolve().parent / "test-runs" / "daily-log"
DEFAULT_API_KEY = os.getenv("MOBIAGENT_API_KEY", "mobiagent-key")
STATE_FILE_SUFFIX = ".state.json"
RECENT_MATCH_DAYS = 7
RECENT_MATCH_MAX_RECORDS = 100

FACT_SPLIT_SYSTEM_PROMPT = textwrap.dedent(
    """
    你是一个日志整理助手。你会把一条原始 daily-log 记录拆分成 1 到多个可去重的事实。
    返回严格 JSON，格式为：
    {
      "facts": [
        {
          "summary": "一句话摘要",
          "normalized_fact": "适合长期画像沉淀的整理后表述",
          "event_date": "YYYY-MM-DD 或空字符串",
          "tags": ["标签1", "标签2"],
          "model_notes": "简短说明这条事实是如何从原文拆出来的"
        }
      ]
    }

    要求：
    1. 一条原始记录中如果包含多件不同事情，拆成多条 facts。
    2. 如果原文只是同一事件的重复表述，不要重复拆分。
    3. event_date 如果无法明确判断，就使用空字符串。
    4. 如果是订单、账单、交易、消费记录，normalized_fact 必须尽量保留关键细节，例如商家、商品/服务、金额、时间、状态；不要压缩成只有“实付25.8元”这种只剩金额的信息。
    4. 不要输出 markdown，不要解释，只输出 JSON。
    """
).strip()

DEDUP_SYSTEM_PROMPT = textwrap.dedent(
    """
    你是一个日志去重助手。你会比较一条候选事实和若干条历史记录，判断它是否应该新增、视为重复、合并到现有记录，或因信息不足而跳过。

    返回严格 JSON，格式为：
    {
      "decision": "new | duplicate | merge_with_existing | skip_ambiguous",
      "matched_record_ids": [1, 2],
      "reason": "简短理由",
      "merged_summary": "如果需要合并，可给出更好的摘要，否则为空字符串",
      "merged_fact": "如果需要合并，可给出更好的整理后事实，否则为空字符串",
      "merged_tags": ["标签1"]
    }

    判断标准：
    1. duplicate: 候选事实和历史记录表达的是同一件事，且没有新增信息。
    2. merge_with_existing: 候选事实和历史记录高度相关，但带来补充细节，适合并入已有记录；只能用于同一个原子条目，绝不能把多个账单条目或多段聊天合并成一个大记录。
    3. new: 是新的独立事实。
    4. skip_ambiguous: 无法稳定判断。

    额外要求：
    1. 一条账单记录只能对应一个账单条目，例如一笔收益、一笔消费、一笔买入。
    2. 不要把“0.37元、0.38元、0.39元”这种多条账单合并成一条记录，除非它们本来就是同一条账单事实的重复表述。
    3. matched_record_ids 最多返回一个最相关的历史记录。

    不要输出 markdown，不要解释，只输出 JSON。
    """
).strip()


@dataclass
class SourceEntry:
    source_date: str
    file_name: str
    entry_index: int
    entry_text: str
    workflow_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def source_ref(self) -> str:
        return f"{self.source_date}/{self.file_name}#{self.entry_index}"


@dataclass
class CandidateFact:
    summary: str
    normalized_fact: str
    event_date: str
    tags: list[str]
    model_notes: str
    raw_excerpt: str
    source_ref: str
    source_date: str
    dedup_text: str


@dataclass
class AggregatedRecord:
    record_id: int
    first_seen_date: str
    last_seen_date: str
    source_refs: list[str]
    source_dates: list[str]
    summary: str
    normalized_fact: str
    raw_excerpts: list[str]
    tags: list[str]
    dedup_status: str
    model_notes: list[str]
    dedup_text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consolidate daily-log markdown files into structured profile records")
    parser.add_argument("--input_root", type=str, default=str(DEFAULT_INPUT_ROOT), help="Root folder of daily-log/YYYY-MM-DD")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory for consolidated markdown files")
    parser.add_argument("--start_date", required=True, help="Start date in YYYY-MM-DD")
    parser.add_argument("--end_date", required=True, help="End date in YYYY-MM-DD")
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--file_name", type=str, help="Exact raw daily-log file name to consolidate")
    selector.add_argument("--file_glob", type=str, help="Glob pattern for raw daily-log file names")
    parser.add_argument("--service_ip", type=str, default="localhost", help="Model service IP")
    parser.add_argument("--model_port", type=int, default=8000, help="OpenAI-compatible model port")
    parser.add_argument(
        "--model_name",
        type=str,
        default="",
        help="Optional model name; if omitted, the first model returned by /v1/models will be used",
    )
    parser.add_argument("--api_key", type=str, default=DEFAULT_API_KEY, help="API key for the OpenAI-compatible endpoint")
    parser.add_argument("--dedup_window", type=int, default=5, help="How many recent records are sent to the model for dedup")
    parser.add_argument("--max_entries", type=int, default=None, help="Optional cap on how many source entries to process")
    parser.add_argument("--dry_run", action="store_true", help="Run the pipeline without writing output files")
    parser.add_argument("--disable_model", action="store_true", help="Use heuristic splitting and dedup instead of model calls")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    return parser.parse_args()


def parse_iso_date(raw: str) -> date:
    return datetime.strptime(raw, "%Y-%m-%d").date()


def iter_date_strings(start_date: date, end_date: date) -> list[str]:
    values: list[str] = []
    current = start_date
    while current <= end_date:
        values.append(current.isoformat())
        current += timedelta(days=1)
    return values


def read_daily_log_document(log_path: Path) -> tuple[dict[str, Any], str]:
    metadata: dict[str, Any] = {"workflow_metadata": {}, "latest_entry_index": 0}
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


def parse_numbered_entries(body: str) -> list[tuple[int, str]]:
    entries: list[tuple[int, str]] = []
    for match in NUMBERED_ENTRY_PATTERN.finditer(body.strip()):
        entry_index = int(match.group(1))
        entry_text = match.group(2).strip()
        if entry_text:
            entries.append((entry_index, entry_text))
    return entries


def collect_source_entries(
    input_root: Path,
    start_date: date,
    end_date: date,
    file_name: str | None,
    file_glob: str | None,
    max_entries: int | None,
) -> dict[str, list[SourceEntry]]:
    grouped_entries: dict[str, list[SourceEntry]] = {}
    processed = 0
    for day in iter_date_strings(start_date, end_date):
        day_dir = input_root / day
        if not day_dir.exists():
            continue
        if file_name:
            candidates = [day_dir / file_name]
        else:
            candidates = sorted(day_dir.glob(file_glob or "*.md"))
        for candidate in candidates:
            if not candidate.exists() or not candidate.is_file():
                continue
            metadata, body = read_daily_log_document(candidate)
            for entry_index, entry_text in parse_numbered_entries(body):
                grouped_entries.setdefault(candidate.name, []).append(
                    SourceEntry(
                        source_date=day,
                        file_name=candidate.name,
                        entry_index=entry_index,
                        entry_text=entry_text,
                        workflow_metadata=metadata.get("workflow_metadata", {}) or {},
                    )
                )
                processed += 1
                if max_entries is not None and processed >= max_entries:
                    return grouped_entries
    return grouped_entries


def group_entries_by_month(
    grouped_entries: dict[str, list[SourceEntry]],
) -> dict[tuple[str, str], list[SourceEntry]]:
    grouped_by_month: dict[tuple[str, str], list[SourceEntry]] = {}
    for file_name, entries in grouped_entries.items():
        for entry in entries:
            month_key = entry.source_date[:7]
            grouped_by_month.setdefault((month_key, file_name), []).append(entry)
    for key in grouped_by_month:
        grouped_by_month[key].sort(key=lambda item: (item.source_date, item.entry_index, item.source_ref))
    return grouped_by_month


def build_client(service_ip: str, model_port: int, api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=f"http://{service_ip}:{model_port}/v1")


def resolve_model_name(client: OpenAI, model_name: str) -> str:
    if model_name.strip():
        return model_name.strip()

    response = client.models.list()
    models = list(getattr(response, "data", []) or [])
    if not models:
        raise ValueError("No models were returned by the OpenAI-compatible endpoint. Please pass --model_name explicitly.")

    first_model = getattr(models[0], "id", None)
    if not isinstance(first_model, str) or not first_model.strip():
        raise ValueError("Failed to resolve a usable model id from the OpenAI-compatible endpoint. Please pass --model_name explicitly.")
    return first_model.strip()


def call_json_model(client: OpenAI, model_name: str, system_prompt: str, user_prompt: str, max_tokens: int = 800) -> dict[str, Any]:
    resolved_model_name = resolve_model_name(client, model_name)
    response_text = client.chat.completions.create(
        model=resolved_model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=max_tokens,
        timeout=60,
    ).choices[0].message.content
    return robust_json_loads(response_text)


def heuristic_split(entry: SourceEntry) -> list[CandidateFact]:
    atomic_clauses = extract_atomic_clauses(entry.entry_text)
    if atomic_clauses:
        return [
            CandidateFact(
                summary=clause,
                normalized_fact=clause,
                event_date=entry.source_date,
                tags=[],
                model_notes="heuristic_atomic_split",
                raw_excerpt=entry.entry_text,
                source_ref=entry.source_ref,
                source_date=entry.source_date,
                dedup_text=normalize_dedup_text(clause),
            )
            for clause in atomic_clauses
        ]

    normalized = normalize_free_text(entry.entry_text)
    summary = normalized[:60] + ("..." if len(normalized) > 60 else "")
    return [
        CandidateFact(
            summary=summary or "未命名记录",
            normalized_fact=normalized,
            event_date=entry.source_date,
            tags=[],
            model_notes="heuristic_split",
            raw_excerpt=entry.entry_text,
            source_ref=entry.source_ref,
            source_date=entry.source_date,
            dedup_text=normalize_dedup_text(normalized),
        )
    ]


def split_entry_into_facts(entry: SourceEntry, client: OpenAI | None, model_name: str, disable_model: bool) -> list[CandidateFact]:
    if disable_model:
        return heuristic_split(entry)

    if client is None:
        raise ValueError("Model client is required unless --disable_model is set")

    prompt = textwrap.dedent(
        f"""
        来源日期: {entry.source_date}
        来源文件: {entry.file_name}
        来源条目: {entry.entry_index}
        workflow 元数据: {json.dumps(entry.workflow_metadata, ensure_ascii=False)}

        原始 daily-log 记录:
        {entry.entry_text}
        """
    ).strip()
    try:
        payload = call_json_model(client, model_name, FACT_SPLIT_SYSTEM_PROMPT, prompt)
    except Exception as exc:
        logging.warning("Falling back to heuristic fact split for %s due to model parse failure: %s", entry.source_ref, exc)
        return heuristic_split(entry)
    facts_payload = payload.get("facts", [])
    if not isinstance(facts_payload, list) or not facts_payload:
        return heuristic_split(entry)

    facts: list[CandidateFact] = []
    for item in facts_payload:
        if not isinstance(item, dict):
            continue
        summary = normalize_free_text(str(item.get("summary", "")))
        normalized_fact = normalize_free_text(str(item.get("normalized_fact", "")))
        if not summary or not normalized_fact:
            continue
        event_date = normalize_optional_date(str(item.get("event_date", "")).strip(), entry.source_date)
        tags = normalize_tags(item.get("tags", []))
        model_notes = normalize_free_text(str(item.get("model_notes", "")))
        facts.append(
            CandidateFact(
                summary=summary,
                normalized_fact=normalized_fact,
                event_date=event_date,
                tags=tags,
                model_notes=model_notes,
                raw_excerpt=entry.entry_text,
                source_ref=entry.source_ref,
                source_date=entry.source_date,
                dedup_text=normalize_dedup_text(normalized_fact),
            )
        )
    return atomicize_candidate_facts(entry, facts) or heuristic_split(entry)


def normalize_optional_date(raw_value: str, fallback: str) -> str:
    if not raw_value:
        return fallback
    try:
        return parse_iso_date(raw_value).isoformat()
    except ValueError:
        return fallback


def normalize_tags(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    tags: list[str] = []
    for item in value:
        text = normalize_free_text(str(item))
        if text and text not in tags:
            tags.append(text)
    return tags


def normalize_free_text(value: str) -> str:
    normalized = re.sub(r"\s+", " ", str(value or "")).strip()
    return normalized


def normalize_dedup_text(value: str) -> str:
    normalized = normalize_free_text(value)
    normalized = re.sub(r"^(\d{4}-\d{2}-\d{2}|\d{1,2}-\d{1,2}号?|\d{1,2}月\d{1,2}日)[:：\s-]*", "", normalized)
    return normalized


def summarize_text(value: str, max_length: int = 80) -> str:
    normalized = normalize_free_text(value)
    if len(normalized) <= max_length:
        return normalized
    return normalized[:max_length] + "..."


def detail_score(value: str) -> int:
    normalized = normalize_free_text(value)
    score = len(normalized)
    if "下单时间" in normalized:
        score += 40
    if "订单" in normalized:
        score += 20
    if re.search(r"[“\"'].+?[”\"']", normalized):
        score += 20
    if re.search(r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}", normalized):
        score += 20
    return score


def prefer_richer_fact(current_fact: str, candidate_fact: str) -> bool:
    current = normalize_free_text(current_fact)
    candidate = normalize_free_text(candidate_fact)
    if not candidate:
        return False
    if not current:
        return True
    return detail_score(candidate) > detail_score(current)


def extract_atomic_clauses(text: str) -> list[str]:
    normalized = normalize_free_text(text)
    if not normalized:
        return []

    transaction_clauses = extract_transaction_clauses(normalized)
    if transaction_clauses:
        return transaction_clauses

    segments: list[str] = []
    marker_pattern = re.compile(r"(?:账单列表显示了|具体记录有[:：]|记录有[:：]|账单如下[:：])(.*?)(?:。|$)")
    for match in marker_pattern.finditer(normalized):
        segments.append(match.group(1))

    if not segments and len(re.findall(r"\d+(?:\.\d+)?元", normalized)) <= 1:
        return []

    if not segments:
        segments = [normalized]

    clauses: list[str] = []
    seen: set[str] = set()
    for segment in segments:
        segment = segment.replace("以及", "，")
        parts = re.split(r"[，；。]", segment)
        for part in parts:
            clause = normalize_free_text(part)
            clause = re.sub(r"^(包括|显示了|具体记录有|记录有|账单如下)[:：\s]*", "", clause)
            if not clause:
                continue
            if not re.search(r"\d+(?:\.\d+)?元", clause):
                continue
            clause_key = normalize_dedup_text(clause)
            if clause_key and clause_key not in seen:
                seen.add(clause_key)
                clauses.append(clause)
    return clauses


def extract_transaction_clauses(text: str) -> list[str]:
    if len(re.findall(r"\d+(?:\.\d+)?元", text)) <= 1:
        return []

    cleaned = re.sub(
        r"^当前页面显示了.*?(?:订单记录|消费记录|账单记录)[，,:：\s]*",
        "",
        text,
    )
    cleaned = re.sub(
        r"^近期在\d+家店(?:有)?消费(?:过|记录)?[，,:：\s]*",
        "",
        cleaned,
    )

    order_marker = r"(?:其中最近一单是|其中有一单是|最近一单是|第一单(?:同样是|是)?|第二单(?:同样是|是)?|第三单(?:同样是|是)?|第[一二三四五六七八九十\d]+单(?:同样是|是)?|接着是|随后是)"
    sentence_chunks = re.split(r"(?<=。)", cleaned)
    clauses: list[str] = []
    seen: set[str] = set()

    for chunk in sentence_chunks:
        chunk = normalize_free_text(chunk.strip("。；; "))
        if not chunk or not re.search(r"\d+(?:\.\d+)?元", chunk):
            continue

        parts = re.split(f"(?={order_marker})", chunk)
        for part in parts:
            clause = normalize_free_text(part.strip("，。；; "))
            if not clause or not re.search(r"\d+(?:\.\d+)?元", clause):
                continue
            clause = re.sub(rf"^(?:{order_marker})", "", clause)
            clause = re.sub(r"^(同样是|还有一单是|还有|一单是)", "", clause)
            clause = normalize_free_text(clause.strip("，。；; "))
            if not clause:
                continue
            clause_key = normalize_dedup_text(clause)
            if clause_key and clause_key not in seen:
                seen.add(clause_key)
                clauses.append(clause)

    return clauses


def atomicize_candidate_facts(entry: SourceEntry, facts: list[CandidateFact]) -> list[CandidateFact]:
    atomic_facts: list[CandidateFact] = []
    seen: set[str] = set()
    for fact in facts:
        clauses = extract_atomic_clauses(fact.normalized_fact)
        if not clauses:
            clauses = extract_atomic_clauses(fact.raw_excerpt)

        if clauses:
            for clause in clauses:
                dedup_text = normalize_dedup_text(clause)
                if not dedup_text or dedup_text in seen:
                    continue
                seen.add(dedup_text)
                atomic_facts.append(
                    CandidateFact(
                        summary=summarize_text(clause),
                        normalized_fact=clause,
                        event_date=fact.event_date or entry.source_date,
                        tags=list(fact.tags),
                        model_notes=fact.model_notes or "atomicized_from_model_fact",
                        raw_excerpt=fact.raw_excerpt,
                        source_ref=fact.source_ref,
                        source_date=fact.source_date,
                        dedup_text=dedup_text,
                    )
                )
            continue

        dedup_text = normalize_dedup_text(fact.normalized_fact)
        if dedup_text and dedup_text not in seen:
            seen.add(dedup_text)
            atomic_facts.append(
                CandidateFact(
                    summary=summarize_text(fact.summary or fact.normalized_fact),
                    normalized_fact=normalize_free_text(fact.normalized_fact),
                    event_date=fact.event_date or entry.source_date,
                    tags=list(fact.tags),
                    model_notes=fact.model_notes,
                    raw_excerpt=fact.raw_excerpt,
                    source_ref=fact.source_ref,
                    source_date=fact.source_date,
                    dedup_text=dedup_text,
                )
            )
    return atomic_facts


def normalized_similarity_key(value: str) -> str:
    lowered = normalize_dedup_text(value).lower()
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", lowered)


def extract_amount_tokens(value: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?元", value))


def normalized_transaction_signature(value: str) -> str:
    normalized = normalize_dedup_text(value)
    normalized = re.sub(r"(收益发放|发放|交通出行|投资理财|支出|消费|买入)", "", normalized)
    normalized = re.sub(r"\d+(?:\.\d+)?元", "", normalized)
    return normalized_similarity_key(normalized)


def parse_record_date(raw_value: str) -> date:
    return datetime.strptime(raw_value, "%Y-%m-%d").date()


def select_recent_scope_records(records: list[AggregatedRecord], fact: CandidateFact) -> list[AggregatedRecord]:
    if not records:
        return []

    fact_date = parse_record_date(fact.source_date)
    lower_bound = fact_date - timedelta(days=RECENT_MATCH_DAYS - 1)
    recent_by_days = [
        record
        for record in records
        if lower_bound <= parse_record_date(record.last_seen_date) <= fact_date
    ]

    sorted_records = sorted(
        records,
        key=lambda record: (record.last_seen_date, record.record_id),
        reverse=True,
    )
    recent_by_count = list(reversed(sorted_records[:RECENT_MATCH_MAX_RECORDS]))

    if not recent_by_days:
        return recent_by_count
    if not recent_by_count:
        return recent_by_days
    if len(recent_by_days) <= len(recent_by_count):
        return recent_by_days
    return recent_by_count


def select_candidate_records(records: list[AggregatedRecord], fact: CandidateFact, limit: int) -> list[AggregatedRecord]:
    if not records:
        return []

    fact_key = normalized_similarity_key(fact.dedup_text)
    fact_amounts = extract_amount_tokens(fact.dedup_text)
    scored: list[tuple[int, int, AggregatedRecord]] = []
    for index, record in enumerate(records):
        record_key = normalized_similarity_key(record.dedup_text)
        score = 0
        if fact_key and record_key:
            if fact_key == record_key:
                score += 100
            elif fact_key in record_key or record_key in fact_key:
                score += 40
        shared_amounts = fact_amounts & extract_amount_tokens(record.dedup_text)
        score += 20 * len(shared_amounts)
        if score > 0:
            scored.append((score, index, record))

    if not scored:
        return records[-limit:] if limit > 0 else records

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in scored[:limit]]


def find_exact_existing_match(records: list[AggregatedRecord], fact: CandidateFact) -> AggregatedRecord | None:
    fact_key = normalized_similarity_key(fact.dedup_text)
    fact_amounts = extract_amount_tokens(fact.dedup_text)
    for record in records:
        record_key = normalized_similarity_key(record.dedup_text)
        if fact.source_ref in record.source_refs and fact_key and fact_key == record_key:
            return record
        if fact.source_ref in record.source_refs:
            record_amounts = extract_amount_tokens(record.dedup_text)
            if fact_amounts and fact_amounts == record_amounts:
                return record
    for record in records:
        record_key = normalized_similarity_key(record.dedup_text)
        if fact_key and fact_key == record_key:
            return record
    return None


def decide_dedup(
    fact: CandidateFact,
    candidate_records: list[AggregatedRecord],
    client: OpenAI | None,
    model_name: str,
    disable_model: bool,
) -> dict[str, Any]:
    if not candidate_records:
        return {"decision": "new", "matched_record_ids": [], "reason": "no_recent_records"}

    if disable_model:
        fact_key = normalized_similarity_key(fact.dedup_text)
        fact_amounts = extract_amount_tokens(fact.dedup_text)
        fact_signature = normalized_transaction_signature(fact.dedup_text)
        for record in reversed(candidate_records):
            record_key = normalized_similarity_key(record.dedup_text)
            if fact_key and fact_key == record_key:
                return {
                    "decision": "duplicate",
                    "matched_record_ids": [record.record_id],
                    "reason": "heuristic_exact_match",
                }
            record_amounts = extract_amount_tokens(record.dedup_text)
            record_signature = normalized_transaction_signature(record.dedup_text)
            if fact_amounts and fact_amounts == record_amounts and fact_signature and fact_signature == record_signature:
                return {
                    "decision": "merge_with_existing",
                    "matched_record_ids": [record.record_id],
                    "reason": "heuristic_signature_match",
                }
        return {"decision": "new", "matched_record_ids": [], "reason": "heuristic_no_match"}

    if client is None:
        raise ValueError("Model client is required unless --disable_model is set")

    recent_payload = [
        {
            "record_id": record.record_id,
            "first_seen_date": record.first_seen_date,
            "last_seen_date": record.last_seen_date,
            "summary": record.summary,
            "normalized_fact": record.normalized_fact,
            "dedup_text": record.dedup_text,
            "tags": record.tags,
        }
        for record in candidate_records
    ]
    prompt = textwrap.dedent(
        f"""
        候选事实:
        {json.dumps(asdict(fact), ensure_ascii=False, indent=2)}

        最近历史记录:
        {json.dumps(recent_payload, ensure_ascii=False, indent=2)}
        """
    ).strip()
    try:
        payload = call_json_model(client, model_name, DEDUP_SYSTEM_PROMPT, prompt)
    except Exception as exc:
        logging.warning("Falling back to heuristic dedup for %s due to model parse failure: %s", fact.source_ref, exc)
        return decide_dedup(fact, candidate_records, client, model_name, disable_model=True)
    decision = str(payload.get("decision", "new")).strip() or "new"
    matched_record_ids = payload.get("matched_record_ids", [])
    if not isinstance(matched_record_ids, list):
        matched_record_ids = []
    return {
        "decision": decision,
        "matched_record_ids": [int(item) for item in matched_record_ids if str(item).isdigit()],
        "reason": normalize_free_text(str(payload.get("reason", ""))),
        "merged_summary": normalize_free_text(str(payload.get("merged_summary", ""))),
        "merged_fact": normalize_free_text(str(payload.get("merged_fact", ""))),
        "merged_tags": normalize_tags(payload.get("merged_tags", [])),
    }


def read_consolidated_document(path: Path) -> tuple[dict[str, Any], list[AggregatedRecord]]:
    default_metadata = {
        "source_date_range": {},
        "source_files": [],
        "latest_record_index": 0,
        "dedup_window_size": 0,
        "updated_at": "",
    }
    state_path = get_state_file_path(path)
    if state_path.exists():
        return read_consolidated_state(state_path, default_metadata)

    if not path.exists():
        return default_metadata, []

    content = path.read_text(encoding="utf-8")
    metadata_match = CONSOLIDATED_METADATA_PATTERN.match(content)
    if not metadata_match:
        return default_metadata, []

    raw_metadata = metadata_match.group(1)
    try:
        metadata = json.loads(raw_metadata)
    except json.JSONDecodeError:
        logging.warning("Failed to parse consolidated metadata: %s", path)
        return default_metadata, []
    if not isinstance(metadata, dict):
        metadata = dict(default_metadata)
    else:
        default_metadata.update(metadata)
        metadata = default_metadata

    remaining = content[metadata_match.end():]
    records_match = CONSOLIDATED_RECORDS_PATTERN.match(remaining)
    if not records_match:
        return metadata, []

    raw_records = records_match.group(1)
    try:
        parsed_records = json.loads(raw_records)
    except json.JSONDecodeError:
        logging.warning("Failed to parse consolidated records: %s", path)
        return metadata, []

    records: list[AggregatedRecord] = []
    for item in parsed_records:
        if not isinstance(item, dict):
            continue
        item = dict(item)
        item.setdefault("dedup_text", normalize_dedup_text(str(item.get("normalized_fact", ""))))
        records.append(AggregatedRecord(**item))
    return metadata, records


def get_state_file_path(markdown_path: Path) -> Path:
    return markdown_path.with_name(markdown_path.name + STATE_FILE_SUFFIX)


def read_consolidated_state(state_path: Path, default_metadata: dict[str, Any]) -> tuple[dict[str, Any], list[AggregatedRecord]]:
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logging.warning("Failed to parse consolidated state file: %s", state_path)
        return default_metadata, []

    metadata = dict(default_metadata)
    parsed_metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    if isinstance(parsed_metadata, dict):
        metadata.update(parsed_metadata)

    parsed_records = payload.get("records", []) if isinstance(payload, dict) else []
    records: list[AggregatedRecord] = []
    if isinstance(parsed_records, list):
        for item in parsed_records:
            if not isinstance(item, dict):
                continue
            item = dict(item)
            item.setdefault("dedup_text", normalize_dedup_text(str(item.get("normalized_fact", ""))))
            records.append(AggregatedRecord(**item))
    return metadata, records


def write_consolidated_state(state_path: Path, metadata: dict[str, Any], records: list[AggregatedRecord]) -> None:
    payload = {
        "metadata": metadata,
        "records": [asdict(record) for record in records],
    }
    state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def escape_table_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def render_consolidated_document(metadata: dict[str, Any], records: list[AggregatedRecord]) -> str:
    lines = [
        "| 序号 | 记录日期 | 最近来源日期 | 内容 | 来源条目 | 去重状态 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for record in records:
        source_refs = "<br>".join(escape_table_cell(item) for item in record.source_refs[-3:])
        lines.append(
            "| {record_id} | {first} | {last} | {content} | {refs} | {status} |".format(
                record_id=record.record_id,
                first=escape_table_cell(record.first_seen_date),
                last=escape_table_cell(record.last_seen_date),
                content=escape_table_cell(record.normalized_fact),
                refs=source_refs or "-",
                status=escape_table_cell(record.dedup_status),
            )
        )

    body = "\n".join(lines).rstrip() + "\n"
    return body


def update_existing_record(record: AggregatedRecord, fact: CandidateFact, decision_payload: dict[str, Any]) -> None:
    record.last_seen_date = max(record.last_seen_date, fact.source_date)
    if fact.event_date and fact.event_date not in record.source_dates:
        record.source_dates.append(fact.event_date)
        record.source_dates.sort()
    if fact.source_ref not in record.source_refs:
        record.source_refs.append(fact.source_ref)
    if fact.raw_excerpt not in record.raw_excerpts:
        record.raw_excerpts.append(fact.raw_excerpt)

    merged_summary = decision_payload.get("merged_summary") or ""
    merged_fact = decision_payload.get("merged_fact") or ""
    merged_tags = decision_payload.get("merged_tags") or []
    if merged_summary:
        record.summary = summarize_text(merged_summary)
    if merged_fact:
        record.normalized_fact = normalize_free_text(merged_fact)
        record.dedup_text = normalize_dedup_text(record.normalized_fact)
    elif prefer_richer_fact(record.normalized_fact, fact.normalized_fact):
        record.normalized_fact = normalize_free_text(fact.normalized_fact)
        record.summary = summarize_text(fact.summary or fact.normalized_fact)
        record.dedup_text = normalize_dedup_text(record.normalized_fact)
    for tag in merged_tags:
        if tag not in record.tags:
            record.tags.append(tag)
    for tag in fact.tags:
        if tag not in record.tags:
            record.tags.append(tag)

    record.dedup_status = decision_payload.get("decision", record.dedup_status)
    reason = decision_payload.get("reason") or fact.model_notes
    if reason and reason not in record.model_notes:
        record.model_notes.append(reason)


def create_new_record(next_record_id: int, fact: CandidateFact, decision_payload: dict[str, Any]) -> AggregatedRecord:
    reason = decision_payload.get("reason") or fact.model_notes
    return AggregatedRecord(
        record_id=next_record_id,
        first_seen_date=fact.event_date or fact.source_date,
        last_seen_date=fact.source_date,
        source_refs=[fact.source_ref],
        source_dates=sorted({fact.event_date or fact.source_date, fact.source_date}),
        summary=summarize_text(fact.summary or fact.normalized_fact),
        normalized_fact=normalize_free_text(fact.normalized_fact),
        raw_excerpts=[fact.raw_excerpt],
        tags=list(fact.tags),
        dedup_status=decision_payload.get("decision", "new"),
        model_notes=[reason] if reason else [],
        dedup_text=fact.dedup_text,
    )


def consolidate_group(
    month_key: str,
    file_name: str,
    entries: list[SourceEntry],
    output_path: Path,
    client: OpenAI | None,
    model_name: str,
    dedup_window: int,
    disable_model: bool,
    dry_run: bool,
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    metadata, records = read_consolidated_document(output_path)
    next_record_id = max([record.record_id for record in records], default=0) + 1
    added_records = 0
    merged_records = 0
    skipped_records = 0

    for entry in entries:
        facts = split_entry_into_facts(entry, client, model_name, disable_model)
        for fact in facts:
            scoped_records = select_recent_scope_records(records, fact)
            exact_match = find_exact_existing_match(scoped_records, fact)
            if exact_match is not None:
                update_existing_record(
                    exact_match,
                    fact,
                    {
                        "decision": "duplicate",
                        "matched_record_ids": [exact_match.record_id],
                        "reason": "exact_existing_match",
                    },
                )
                merged_records += 1
                continue

            candidate_records = (
                select_candidate_records(scoped_records, fact, dedup_window)
                if dedup_window > 0
                else scoped_records
            )
            decision_payload = decide_dedup(fact, candidate_records, client, model_name, disable_model)
            decision = decision_payload.get("decision", "new")

            if decision == "new":
                records.append(create_new_record(next_record_id, fact, decision_payload))
                next_record_id += 1
                added_records += 1
                continue

            matched_record_ids = decision_payload.get("matched_record_ids", [])
            matched_record = None
            if matched_record_ids:
                for record in records:
                    if record.record_id == matched_record_ids[0]:
                        matched_record = record
                        break

            if decision in {"duplicate", "merge_with_existing"} and matched_record is not None:
                update_existing_record(matched_record, fact, decision_payload)
                merged_records += 1
                continue

            skipped_records += 1
            logging.info("Skipped fact from %s due to decision=%s", fact.source_ref, decision)

    metadata.update(
        {
            "month": month_key,
            "source_date_range": {"start_date": start_date, "end_date": end_date},
            "source_files": sorted(set(metadata.get("source_files", [])) | {file_name}),
            "latest_record_index": max([record.record_id for record in records], default=0),
            "dedup_window_size": dedup_window,
            "recent_match_days": RECENT_MATCH_DAYS,
            "recent_match_max_records": RECENT_MATCH_MAX_RECORDS,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "total_records": len(records),
        }
    )

    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        state_path = get_state_file_path(output_path)
        write_consolidated_state(state_path, metadata, records)
        output_path.write_text(render_consolidated_document(metadata, records), encoding="utf-8")

    return {
        "file_name": file_name,
        "output_path": str(output_path),
        "added_records": added_records,
        "merged_or_duplicate_records": merged_records,
        "skipped_records": skipped_records,
        "total_records": len(records),
        "dry_run": dry_run,
    }


def configure_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    start_date = parse_iso_date(args.start_date)
    end_date = parse_iso_date(args.end_date)
    if end_date < start_date:
        raise ValueError("end_date must be greater than or equal to start_date")

    input_root = Path(args.input_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    grouped_entries = collect_source_entries(
        input_root=input_root,
        start_date=start_date,
        end_date=end_date,
        file_name=args.file_name,
        file_glob=args.file_glob,
        max_entries=args.max_entries,
    )
    if not grouped_entries:
        logging.warning("No matching source entries found")
        return 1

    client = None if args.disable_model else build_client(args.service_ip, args.model_port, args.api_key)
    grouped_entries_by_month = group_entries_by_month(grouped_entries)
    summaries = []
    for (month_key, file_name), entries in sorted(grouped_entries_by_month.items()):
        output_path = output_dir / month_key / file_name
        result = consolidate_group(
            month_key=month_key,
            file_name=file_name,
            entries=entries,
            output_path=output_path,
            client=client,
            model_name=args.model_name,
            dedup_window=args.dedup_window,
            disable_model=args.disable_model,
            dry_run=args.dry_run,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )
        summaries.append(result)
        logging.info(
            "Processed %s: added=%d, merged_or_duplicate=%d, skipped=%d, total=%d",
            result["file_name"],
            result["added_records"],
            result["merged_or_duplicate_records"],
            result["skipped_records"],
            result["total_records"],
        )

    print(json.dumps({"results": summaries}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())