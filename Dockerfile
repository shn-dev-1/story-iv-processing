# syntax=docker/dockerfile:1.6
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LANG=C.UTF-8 LC_ALL=C.UTF-8

# Minimal system deps (Pillow etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libjpeg62-turbo \
    zlib1g \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# -------- Python deps (ARM64-friendly) --------
COPY requirements.txt /app/requirements.txt

# Upgrade tooling
RUN pip install --upgrade pip setuptools wheel

# Install CPU wheels for torch + onnxruntime FIRST (no source builds on ARM64)
ENV PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
RUN pip install --only-binary=:all: --extra-index-url $PYTORCH_INDEX_URL \
      torch==2.1.0 onnxruntime==1.17.0 && \
    pip install --only-binary=:all: -r /app/requirements.txt

# -------- Model caching (ONLINE during build only) --------
# One cache path used at build and runtime
ENV HF_HOME=/opt/hfcache \
    HUGGINGFACE_HUB_CACHE=/opt/hfcache/hub \
    TRANSFORMERS_CACHE=/opt/hfcache/transformers \
    HF_HUB_DISABLE_TELEMETRY=1 \
    MODEL_DIR=/opt/models/sd15-onnx

RUN mkdir -p /opt/hfcache/hub /opt/hfcache/transformers $MODEL_DIR

# Choose an ONNX SD1.5 export with embedded tensors.
# You can override with --build-arg MODEL_REPO="...".
ARG MODEL_REPO="TensorStack/stable-diffusion-v1-5-onnx"

# Pull the FULL snapshot (no filters) so all needed files are present.
# We temporarily unset offline flags for this step only.
RUN python - <<'PY'
import os, sys, pathlib
from huggingface_hub import snapshot_download

# ensure we're ONLINE at build time
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)

repo   = os.environ.get("MODEL_REPO")
target = os.environ.get("MODEL_DIR", "/opt/models/sd15-onnx")

print(f"Downloading model repo: {repo}")
snapshot_download(
    repo_id=repo,
    local_dir=target,
    local_dir_use_symlinks=False
)

# quick sanity: expect .onnx graphs present; embedded tensors → no weights.pb required
p_unet = pathlib.Path(target) / "unet" / "model.onnx"
if not p_unet.exists():
    print(f"[FATAL] Missing UNet ONNX at {p_unet}", file=sys.stderr)
    sys.exit(2)
print("Model snapshot OK:", target)
PY

# -------- Force offline for runtime --------
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# -------- App code --------
COPY main.py /app/main.py

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8080"]
