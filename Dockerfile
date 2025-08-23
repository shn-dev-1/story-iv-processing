# syntax=docker/dockerfile:1.6
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libjpeg62-turbo zlib1g \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Ensure requirements.txt is present before pip install
COPY requirements.txt /app/requirements.txt

# Install wheels: toolchain + CPU torch/ORT first (ARM64-friendly)
ENV PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
RUN pip install --upgrade pip setuptools wheel && \
    pip install --only-binary=:all: --extra-index-url $PYTORCH_INDEX_URL \
        torch==2.1.0 onnxruntime==1.17.0 && \
    pip install --only-binary=:all: -r /app/requirements.txt

# --- Model caching (ONLINE during build only) ---
ENV HF_HOME=/opt/hfcache \
HUGGINGFACE_HUB_CACHE=/opt/hfcache/hub \
TRANSFORMERS_CACHE=/opt/hfcache/transformers \
HF_HUB_DISABLE_TELEMETRY=1 \
MODEL_DIR=/opt/models/sd15-onnx

RUN mkdir -p /opt/hfcache/hub /opt/hfcache/transformers $MODEL_DIR

ARG MODEL_REPO="nmkd/stable-diffusion-1.5-onnx-fp16"

RUN python - <<'PY'
import os
from huggingface_hub import snapshot_download
# ensure we're ONLINE at build time
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)

repo   = os.environ.get("MODEL_REPO")
target = os.environ.get("MODEL_DIR", "/opt/models/sd15-onnx")

# Pull the ONNX pipeline plus any external data files the .onnx graphs reference
allow = [
    "model_index.json",
    "**/*.onnx",
    "**/*.onnx_data",
    "**/*.pb",
    "**/*.bin",
    "**/*.data",
    # common aux files
    "tokenizer/**",
    "vocab*.json", "merges.txt", "tokenizer.json", "special_tokens_map.json",
    "scheduler/**",
    "feature_extractor/**",
    "**/config.json",
]
snapshot_download(
    repo_id=repo,
    local_dir=target,
    local_dir_use_symlinks=False,
    allow_patterns=allow,
)
print("Downloaded model into:", target)
PY

# Force offline for runtime
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# Copy app last (keeps layer cache)
COPY main.py /app/main.py

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8080"]
