from __future__ import annotations

import argparse
import os
import time
import uuid
from pathlib import Path

import requests
from dotenv import load_dotenv

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from mem0 import Memory

from model_utils import ensure_model_downloaded


DEFAULT_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = DEFAULT_REPO_ROOT / "runner" / "mobiagent" / ".env"
DEFAULT_EMBEDDING_MODEL_ID = "BAAI/bge-small-zh"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify local embedding + Milvus + local OpenAI-compatible LLM")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE), help="Path to the .env file used by runner.mobiagent.mobiagent")
    return parser.parse_args()


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def wait_for_llm(base_url: str) -> None:
    response = requests.get(f"{base_url.rstrip('/')}/models", timeout=10)
    response.raise_for_status()


def verify_llm_chat_completion(base_url: str) -> None:
    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "请只回复 ok"}],
            "temperature": 0,
            "max_tokens": 16,
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    print("LLM_COMPLETION:", content)


def main() -> None:
    args = parse_args()
    load_dotenv(args.env_file)

    embedding_model = Path(require_env("EMBEDDING_MODEL")).expanduser().resolve()
    ensure_model_downloaded(DEFAULT_EMBEDDING_MODEL_ID, embedding_model)

    openai_base_url = require_env("OPENAI_BASE_URL")
    wait_for_llm(openai_base_url)
    verify_llm_chat_completion(openai_base_url)

    config = {
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
                "openai_base_url": openai_base_url,
            },
        },
    }

    memory = Memory.from_config(config_dict=config)
    user_id = f"profile_mem_verify_{uuid.uuid4().hex[:8]}"
    input_message = "我点外卖时更偏向价格适中、配送快、评分高的餐厅。"
    query = "帮我找一家配送快而且价格不要太高的外卖店"

    add_result = memory.add(
        input_message,
        user_id=user_id,
        infer=False,
        metadata={
            "type": "preference",
            "user_id": user_id,
            "timestamp": time.time(),
        },
    )
    print("ADD_RESULT:", add_result)

    time.sleep(2)
    search_result = memory.search(query, user_id=user_id, limit=5)
    print("SEARCH_RESULT:", search_result)

    results = search_result.get("results", []) if isinstance(search_result, dict) else search_result
    memories = [item.get("memory", "") for item in results if isinstance(item, dict)]
    if not any("配送快" in memory_text or "价格适中" in memory_text for memory_text in memories):
        raise RuntimeError("Mem0 search results did not return the inserted preference memory")

    print("VERIFICATION_OK: local embedding + Milvus + local OpenAI-compatible LLM are working together.")


if __name__ == "__main__":
    main()