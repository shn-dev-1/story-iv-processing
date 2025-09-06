# syntax=docker/dockerfile:1.6
# Build and run on x86_64 (amd64) with CUDA. Use EC2 GPU (g5/g6), not Fargate.
FROM --platform=linux/amd64 pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LANG=C.UTF-8 LC_ALL=C.UTF-8

# System deps (ffmpeg for muxing video)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ----------------- Python deps -----------------
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel \
 && pip install -r /app/requirements.txt

# ----------------- Runtime env -----------------
# Where the model will be stored after we pull it from S3 at startup
ENV MODEL_DIR=/opt/models/wan2.2-ti2v-5b

# Optional: keep HuggingFace offline at runtime so nothing tries to hit the hub
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1

# Put your FastAPI worker
COPY main.py /app/main.py

# Create the model dir now (owned by root, fine for container)
RUN mkdir -p "${MODEL_DIR}"

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8080"]
