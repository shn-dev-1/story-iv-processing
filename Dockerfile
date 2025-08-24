# syntax=docker/dockerfile:1.6
FROM --platform=linux/amd64 pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 LANG=C.UTF-8 LC_ALL=C.UTF-8

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps (torch is already in the base image)
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel \
 && pip install -r /app/requirements.txt

# ---- Model snapshot (Wan 2.2 TI2V-5B Diffusers) ----
ENV HF_HOME=/opt/hfcache \
    HUGGINGFACE_HUB_CACHE=/opt/hfcache/hub \
    TRANSFORMERS_CACHE=/opt/hfcache/transformers \
    HF_HUB_DISABLE_TELEMETRY=1 \
    MODEL_DIR=/opt/models/wan2.2-ti2v-5b

RUN mkdir -p /opt/hfcache/hub /opt/hfcache/transformers $MODEL_DIR

# Override at build time if needed: --build-arg MODEL_REPO="Wan-AI/Wan2.2-TI2V-5B-Diffusers"
ARG MODEL_REPO="Wan-AI/Wan2.2-TI2V-5B-Diffusers"

# If the repo is gated/private, pass a token via: --secret id=hf_token,src=hf_token.txt
RUN --mount=type=secret,id=hf_token,required=false python - <<'PY'
import os, sys
from pathlib import Path
from huggingface_hub import snapshot_download

os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)

repo   = os.environ.get("MODEL_REPO")
target = os.environ.get("MODEL_DIR", "/opt/models/wan2.2-ti2v-5b")
tok = Path("/run/secrets/hf_token")
token = tok.read_text().strip() if tok.exists() else None

print(f"[model] downloading diffusers snapshot: {repo} -> {target}")
snapshot_download(repo_id=repo, local_dir=target, local_dir_use_symlinks=False, token=token)

mi = Path(target) / "model_index.json"
has_st = any(Path(target).rglob("*.safetensors"))
if not mi.exists() or not has_st:
    print("[FATAL] Snapshot missing model_index.json or *.safetensors", file=sys.stderr)
    sys.exit(2)
print("[model] Diffusers snapshot OK:", target)
PY

# Force offline at runtime
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# App
COPY main_video.py /app/main_video.py

EXPOSE 8080
CMD ["uvicorn", "main_video:app", "--host=0.0.0.0", "--port=8080"]
