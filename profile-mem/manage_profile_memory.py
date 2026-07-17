from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

try:
    import pymilvus  # noqa: F401
except Exception as exc:
    raise RuntimeError(
        "Failed to import pymilvus. "
        f"Current interpreter: {sys.executable}. "
        "Please make sure you are using the MobiMind environment and install dependencies with "
        "`pip install -r requirements.txt`. "
        f"Original error: {exc!r}"
    ) from exc

from mem0 import Memory


DEFAULT_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = DEFAULT_REPO_ROOT / "runner" / "mobiagent" / ".env"
DEFAULT_USER_ID = "default_user"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect and manage personal information stored in profile-mem (Mem0 + Milvus)."
    )
    parser.add_argument(
        "--env-file",
        default=str(DEFAULT_ENV_FILE),
        help="Path to the .env file used by runner.mobiagent.mobiagent",
    )
    parser.add_argument(
        "--user-id",
        default=DEFAULT_USER_ID,
        help="Mem0 user_id to operate on",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw JSON output instead of readable text",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List stored personal information")
    list_parser.add_argument("--query", default="", help="Optional semantic query for filtered retrieval")
    list_parser.add_argument("--limit", type=int, default=20, help="Maximum number of records to return")

    add_parser = subparsers.add_parser("add", help="Insert one personal information record")
    add_parser.add_argument("text", help="Personal information text to insert")
    add_parser.add_argument("--type", default="preference", help="Metadata type field")
    add_parser.add_argument("--task-type", default="manual", help="Metadata task_type field")

    update_parser = subparsers.add_parser("update", help="Update one personal information record by id")
    update_parser.add_argument("--id", required=True, help="Memory id returned by the list command")
    update_parser.add_argument("--text", required=True, help="New personal information text")

    delete_parser = subparsers.add_parser("delete", help="Delete one personal information record by id")
    delete_parser.add_argument("--id", required=True, help="Memory id returned by the list command")

    delete_all_parser = subparsers.add_parser(
        "delete-all",
        help="Delete all personal information records for the current user",
    )
    delete_all_parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually perform the bulk deletion",
    )

    return parser.parse_args()


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def build_config() -> dict[str, Any]:
    embedding_model = Path(require_env("EMBEDDING_MODEL")).expanduser().resolve()
    if not embedding_model.exists():
        raise RuntimeError(
            f"Embedding model path does not exist: {embedding_model}. "
            "Please check EMBEDDING_MODEL in runner/mobiagent/.env."
        )

    return {
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": str(embedding_model),
            },
        },
        "vector_store": {
            "provider": "milvus",
            "config": {
                "collection_name": os.getenv("MEM0_COLLECTION_NAME", "mobiagent"),
                "embedding_model_dims": require_env("EMBEDDING_MODEL_DIMS"),
                "url": require_env("MILVUS_URL"),
                "db_name": "default",
                "token": "",
            },
        },
        "llm": {
            "provider": "openai",
            "config": {
                "model": "gpt-4o-mini",
                "api_key": require_env("OPENAI_API_KEY"),
                "openai_base_url": require_env("OPENAI_BASE_URL"),
            },
        },
    }


def get_memory_client(env_file: str) -> Memory:
    load_dotenv(env_file)
    return Memory.from_config(config_dict=build_config())


def normalize_results(results: Any) -> list[dict[str, Any]]:
    if isinstance(results, dict) and "results" in results:
        normalized = results["results"]
    elif isinstance(results, list):
        normalized = results
    else:
        normalized = []
    return [item for item in normalized if isinstance(item, dict)]


def print_results(records: list[dict[str, Any]], as_json: bool) -> None:
    if as_json:
        print(json.dumps(records, ensure_ascii=False, indent=2))
        return

    if not records:
        print("No records found.")
        return

    for index, item in enumerate(records, start=1):
        print(f"[{index}] id={item.get('id', '')}")
        print(f"    memory: {item.get('memory', '')}")
        score = item.get("score")
        if score is not None:
            print(f"    score: {score}")
        metadata = item.get("metadata")
        if metadata:
            print(f"    metadata: {json.dumps(metadata, ensure_ascii=False)}")


def command_list(memory: Memory, args: argparse.Namespace) -> int:
    results = memory.search(args.query, user_id=args.user_id, limit=args.limit)
    records = normalize_results(results)
    print_results(records, args.json)
    return 0


def command_add(memory: Memory, args: argparse.Namespace) -> int:
    result = memory.add(
        args.text,
        user_id=args.user_id,
        infer=False,
        metadata={
            "type": args.type,
            "task_type": args.task_type,
            "user_id": args.user_id,
            "source": "manual_cli",
            "timestamp": time.time(),
        },
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        print("Inserted record successfully.")
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


def command_update(memory: Memory, args: argparse.Namespace) -> int:
    update_func = getattr(memory, "update", None)
    if update_func is None:
        raise RuntimeError("Current mem0 client does not expose an update method.")

    try:
        result = update_func(args.id, args.text)
    except TypeError:
        try:
            result = update_func(memory_id=args.id, data=args.text)
        except TypeError:
            result = update_func(memory_id=args.id, text=args.text)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        print(f"Updated record: {args.id}")
        if result is not None:
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


def command_delete(memory: Memory, args: argparse.Namespace) -> int:
    delete_func = getattr(memory, "delete", None) or getattr(memory, "remove", None)
    if delete_func is None:
        raise RuntimeError("Current mem0 client does not expose delete/remove methods.")

    try:
        result = delete_func(args.id)
    except TypeError:
        result = delete_func(memory_id=args.id)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        print(f"Deleted record: {args.id}")
        if result is not None:
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


def command_delete_all(memory: Memory, args: argparse.Namespace) -> int:
    if not args.yes:
        raise RuntimeError("delete-all is destructive. Re-run with --yes to confirm.")

    delete_func = getattr(memory, "delete", None) or getattr(memory, "remove", None)
    if delete_func is None:
        raise RuntimeError("Current mem0 client does not expose delete/remove methods.")

    results = memory.search("", user_id=args.user_id, limit=1000)
    records = normalize_results(results)
    deleted_ids: list[str] = []

    for item in records:
        memory_id = item.get("id")
        if not memory_id:
            continue
        try:
            delete_func(memory_id)
        except TypeError:
            delete_func(memory_id=memory_id)
        deleted_ids.append(memory_id)

    payload = {
        "user_id": args.user_id,
        "deleted_count": len(deleted_ids),
        "deleted_ids": deleted_ids,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        print(f"Deleted {len(deleted_ids)} records for user_id={args.user_id}")
        if deleted_ids:
            print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


def main() -> int:
    args = parse_args()
    try:
        memory = get_memory_client(args.env_file)
        if args.command == "list":
            return command_list(memory, args)
        if args.command == "add":
            return command_add(memory, args)
        if args.command == "update":
            return command_update(memory, args)
        if args.command == "delete":
            return command_delete(memory, args)
        if args.command == "delete-all":
            return command_delete_all(memory, args)
        raise RuntimeError(f"Unsupported command: {args.command}")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())