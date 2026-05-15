from __future__ import annotations

import argparse
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from transformers import AutoModelForCausalLM, AutoTokenizer

from model_utils import ensure_model_downloaded


SERVER_STATE: Dict[str, Any] = {
    "tokenizer": None,
    "model": None,
    "device": "cpu",
    "model_name": "",
    "model_dir": "",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local OpenAI-compatible LLM server for profile memory")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct", help="Hugging Face model id to download")
    parser.add_argument("--model-dir", required=True, help="Local path to store or load the model")
    parser.add_argument("--served-model-name", default="gpt-4o-mini", help="Model name exposed via the OpenAI-compatible API")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=18001, help="Server port")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Default max tokens for generation")
    parser.add_argument("--download-only", action="store_true", help="Download the model and exit")
    return parser.parse_args()


def choose_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def choose_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def build_prompt(messages: List[Dict[str, Any]], tokenizer: AutoTokenizer) -> str:
    normalized_messages: List[Dict[str, str]] = []
    for message in messages:
        normalized_messages.append(
            {
                "role": str(message.get("role", "user")),
                "content": flatten_content(message.get("content", "")),
            }
        )

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            normalized_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    prompt_lines = []
    for message in normalized_messages:
        prompt_lines.append(f"{message['role']}: {message['content']}")
    prompt_lines.append("assistant:")
    return "\n".join(prompt_lines)


def load_model(model_id: str, model_dir: str, served_model_name: str) -> None:
    local_model_dir = ensure_model_downloaded(model_id, model_dir)
    tokenizer = AutoTokenizer.from_pretrained(local_model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = choose_dtype()
    model = AutoModelForCausalLM.from_pretrained(
        local_model_dir,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    device = choose_device()
    model.to(device)
    model.eval()

    SERVER_STATE.update(
        {
            "tokenizer": tokenizer,
            "model": model,
            "device": device,
            "model_name": served_model_name,
            "model_dir": str(local_model_dir),
        }
    )


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: Optional[str] = None
    messages: List[Dict[str, Any]]
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    stream: bool = False


@asynccontextmanager
async def lifespan(_: FastAPI):
    args = APP_ARGS
    load_model(args.model_id, args.model_dir, args.served_model_name)
    yield


app = FastAPI(lifespan=lifespan)
APP_ARGS = parse_args()


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "Local OpenAI-compatible profile memory LLM server"}


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    if SERVER_STATE["model"] is None or SERVER_STATE["tokenizer"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "model": SERVER_STATE["model_name"]}


@app.get("/v1/models")
def list_models() -> Dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": SERVER_STATE["model_name"],
                "object": "model",
                "created": int(time.time()),
                "owned_by": "profile-mem-local",
            }
        ],
    }


@app.post("/v1/chat/completions")
def create_chat_completion(request: ChatCompletionRequest) -> Dict[str, Any]:
    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming is not supported by this local server")

    tokenizer = SERVER_STATE["tokenizer"]
    model = SERVER_STATE["model"]
    device = SERVER_STATE["device"]
    if tokenizer is None or model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    prompt = build_prompt(request.messages, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    input_token_count = int(inputs["input_ids"].shape[-1])

    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": request.max_tokens or APP_ARGS.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if request.temperature and request.temperature > 0:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = request.temperature
    else:
        generation_kwargs["do_sample"] = False

    with torch.inference_mode():
        outputs = model.generate(**inputs, **generation_kwargs)

    generated_tokens = outputs[0][input_token_count:]
    completion_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    completion_token_count = int(generated_tokens.shape[-1])

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model or SERVER_STATE["model_name"],
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": completion_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": input_token_count,
            "completion_tokens": completion_token_count,
            "total_tokens": input_token_count + completion_token_count,
        },
    }


def main() -> None:
    if APP_ARGS.download_only:
        ensure_model_downloaded(APP_ARGS.model_id, APP_ARGS.model_dir)
        print(f"Model ready at {APP_ARGS.model_dir}")
        return

    uvicorn.run(app, host=APP_ARGS.host, port=APP_ARGS.port, log_level="info")


if __name__ == "__main__":
    main()