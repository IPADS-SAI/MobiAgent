from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download
from modelscope import snapshot_download as modelscope_snapshot_download


def ensure_model_downloaded(model_id: str, target_dir: str | Path) -> Path:
    target_path = Path(target_dir).expanduser().resolve()
    if (target_path / "config.json").exists() or (target_path / "tokenizer_config.json").exists():
        return target_path

    target_path.mkdir(parents=True, exist_ok=True)
    sources = [source.strip() for source in os.getenv("PROFILE_MEM_MODEL_SOURCES", "modelscope,huggingface").split(",") if source.strip()]
    last_error: Exception | None = None

    for source in sources:
        try:
            if source == "modelscope":
                modelscope_snapshot_download(model_id=model_id, local_dir=str(target_path))
            elif source == "huggingface":
                snapshot_download(
                    repo_id=model_id,
                    local_dir=str(target_path),
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )
            else:
                raise ValueError(f"Unsupported model source: {source}")

            if (target_path / "config.json").exists() or (target_path / "tokenizer_config.json").exists():
                return target_path
        except Exception as exc:
            last_error = exc

    if last_error is not None:
        raise RuntimeError(f"Failed to download model {model_id}: {last_error}") from last_error
    raise RuntimeError(f"Failed to download model {model_id}: no download source was configured")