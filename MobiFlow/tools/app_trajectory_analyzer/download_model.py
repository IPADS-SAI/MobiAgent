import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="google/owlvit-base-patch32",
    local_dir="./owlvit-base-patch32",
    local_dir_use_symlinks=False
)