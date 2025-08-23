# Build for ARM64 with:
#   docker buildx build --platform linux/arm64 -t sd15-onnx-arm64 . --no-cache

FROM python:3.11-slim

# System deps (minimal; add libjpeg/zlib for Pillow)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libjpeg62-turbo \
    zlib1g \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LANG=C.UTF-8 LC_ALL=C.UTF-8

# Use ONE cache/model path at build+runtime
ENV HF_HOME=/opt/hfcache \
    HUGGINGFACE_HUB_CACHE=/opt/hfcache/hub \
    TRANSFORMERS_CACHE=/opt/hfcache/transformers \
    MODEL_DIR=/opt/models/sd15-onnx

RUN mkdir -p $HF_HOME $MODEL_DIR

WORKDIR /app
COPY requirements.txt /app/

# Core libs: onnxruntime (AArch64), diffusers, optimum, fastapi, boto3
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# ---- PRE-DOWNLOAD MODEL (online only during build) ----
# We use a public ONNX SD1.5 repo so license acceptance isn’t gated.
# You can swap MODEL_REPO at build-arg time if you prefer another ONNX export.
ARG MODEL_REPO="nmkd/stable-diffusion-1.5-onnx-fp16"
RUN python - <<'PY'
from huggingface_hub import snapshot_download
import os, shutil
repo = os.environ.get("MODEL_REPO", "nmkd/stable-diffusion-1.5-onnx-fp16")
target = os.environ.get("MODEL_DIR", "/opt/models/sd15-onnx")
# Grab entire repo locally (weights + tokenizer files if present)
snapshot_download(repo_id=repo, local_dir=target, local_dir_use_symlinks=False)
print("Downloaded:", repo, "->", target)
PY

# ---- FORCE OFFLINE FOR RUNTIME ----
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1

# Make model + cache readable by non-root
RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /opt
COPY main.py /app/main.py

USER appuser
EXPOSE 8080
CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8080"]
