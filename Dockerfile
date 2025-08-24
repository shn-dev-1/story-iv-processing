# syntax=docker/dockerfile:1.6
# Build and run on x86_64 (amd64) with CUDA. Use EC2 GPU (g5/g6), not Fargate.
FROM --platform=linux/amd64 pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LANG=C.UTF-8 LC_ALL=C.UTF-8

# System deps (ffmpeg for muxing video)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg ca-certificates curl git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ----------------- Python deps -----------------
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel \
 && pip install -r /app/requirements.txt \
 && pip cache purge

# ----------------- Model snapshot via ModelScope -----------------
# Where we’ll store the model at build time
ENV MODEL_DIR=/opt/models/wan2.2-ti2v-5b \
    HF_HUB_DISABLE_TELEMETRY=1

RUN mkdir -p ${MODEL_DIR}

# Pull Wan 2.2 TI2V-5B from ModelScope (no HF auth needed)
RUN python - <<'PY'
import os
from modelscope.hub.snapshot_download import snapshot_download

target = os.environ.get("MODEL_DIR", "/opt/models/wan2.2-ti2v-5b")
repo   = "Wan-AI/Wan2.2-TI2V-5B"

print(f"[modelscope] downloading {repo} -> {target}")
# Place files directly into MODEL_DIR; no symlinks (good for runtime-only hosts)
snapshot_download(model_id=repo, local_dir=target, local_dir_use_symlinks=False)
print("[modelscope] Download complete:", target)
PY

# Make the container run fully offline at runtime
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# ----------------- App code -----------------
# Your FastAPI worker that uses AutoPipelineForText2Video and ffmpeg
COPY main.py /app/main.py

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8080"]
