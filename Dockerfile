# syntax=docker/dockerfile:1.6
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LANG=C.UTF-8 LC_ALL=C.UTF-8

# System deps for Pillow, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libjpeg62-turbo \
    zlib1g \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Make sure requirements.txt is in the image BEFORE using it ---
COPY requirements.txt /app/requirements.txt

# --- Install Python deps with ARM-friendly wheels ---
# 1) Upgrade tooling
RUN pip install --upgrade pip setuptools wheel

# 2) Install CPU wheels for torch & ORT FIRST (ARM64 aarch64 wheels exist)
#    Use PyTorch's CPU index URL to avoid source builds.
ENV PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
RUN pip install --only-binary=:all: --extra-index-url $PYTORCH_INDEX_URL \
      torch==2.1.0 onnxruntime==1.17.0 && \
    pip install --only-binary=:all: -r /app/requirements.txt

# --- (Optional) pre-bake model weights if you need them at runtime & offline ---
ENV HF_HOME=/opt/hfcache \
    HUGGINGFACE_HUB_CACHE=/opt/hfcache/hub \
    TRANSFORMERS_CACHE=/opt/hfcache/transformers \
    HF_HUB_DISABLE_TELEMETRY=1

RUN mkdir -p /opt/hfcache/hub /opt/hfcache/transformers /opt/models

# Example: pre-download a model (uncomment and set your repo if needed)
# ARG MODEL_REPO="nmkd/stable-diffusion-1.5-onnx-fp16"
# RUN python - <<'PY'
# from huggingface_hub import snapshot_download
# import os
# repo = os.environ.get("MODEL_REPO")
# snapshot_download(repo, local_dir="/opt/models/sd15-onnx", local_dir_use_symlinks=False)
# print("Model cached")
# PY

# Force offline at runtime
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# Copy app code after deps so code edits don't bust cache
COPY main.py /app/main.py

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8080"]
