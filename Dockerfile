# syntax=docker/dockerfile:1.6
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LANG=C.UTF-8 LC_ALL=C.UTF-8

# Minimal system deps (Pillow, certs, curl for health checks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libjpeg62-turbo \
    zlib1g \
    curl \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ----------------- Python deps -----------------
# Put requirements.txt next to this Dockerfile in your repo
COPY requirements.txt /app/requirements.txt

# 1) Upgrade tooling
RUN pip install --upgrade pip setuptools wheel

# 2) Install ARM64 CPU wheels for torch + onnxruntime FIRST (no source builds)
ENV PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
RUN pip install --only-binary=:all: --extra-index-url $PYTORCH_INDEX_URL \
      torch==2.1.0 onnxruntime==1.17.0 && \
    pip install --only-binary=:all: -r /app/requirements.txt

# ----------------- Model snapshot -----------------
# One cache path used at build and runtime
ENV HF_HOME=/opt/hfcache \
    HUGGINGFACE_HUB_CACHE=/opt/hfcache/hub \
    TRANSFORMERS_CACHE=/opt/hfcache/transformers \
    HF_HUB_DISABLE_TELEMETRY=1 \
    MODEL_DIR=/opt/models/sd15-onnx

RUN mkdir -p /opt/hfcache/hub /opt/hfcache/transformers $MODEL_DIR

# Default to the NMKD full ONNX export; override with --build-arg if desired
ARG MODEL_REPO="nmkd/stable-diffusion-1.5-onnx-fp16"

# Download the FULL snapshot (no filters) so external data (if any) is included
RUN python - <<'PY'
import os, sys, pathlib
from huggingface_hub import snapshot_download

# Ensure we're ONLINE during build
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)

repo   = os.environ.get("MODEL_REPO")
target = os.environ.get("MODEL_DIR", "/opt/models/sd15-onnx")

print(f"[model] downloading full snapshot: {repo}")
snapshot_download(
    repo_id=repo,
    local_dir=target,
    local_dir_use_symlinks=False
)

# Verify required ONNX graphs exist (Diffusers ONNX layout)
req = [
    ("unet", "model.onnx"),
    ("vae_decoder", "model.onnx"),
    ("text_encoder", "model.onnx"),
]
missing = []
for sub, fname in req:
    p = pathlib.Path(target) / sub / fname
    if not p.exists() or p.stat().st_size == 0:
        missing.append(str(p))
if missing:
    print("[FATAL] Missing required ONNX files:\n  " + "\n  ".join(missing), file=sys.stderr)
    sys.exit(2)

print("[model] ONNX files present, snapshot OK:", target)
PY

# Some community ONNX exports omit an image preprocessor config. Create a default if absent.
RUN python - <<'PY'
import os, json, pathlib
model_dir = pathlib.Path(os.environ.get("MODEL_DIR", "/opt/models/sd15-onnx"))
root_pp = model_dir / "preprocessor_config.json"
feat_pp = model_dir / "feature_extractor" / "preprocessor_config.json"
if not root_pp.exists() and not feat_pp.exists():
    cfg = {
        "_class_name": "CLIPImageProcessor",
        "do_resize": True,
        "do_center_crop": False,
        "do_rescale": True,
        "resample": 3,               # PIL.BICUBIC
        "size": 512,
        "crop_size": 512,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5]
    }
    root_pp.parent.mkdir(parents=True, exist_ok=True)
    root_pp.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"[model] created default preprocessor_config.json at {root_pp}")
else:
    print("[model] preprocessor_config.json already present")
PY

# Force offline at runtime (no Hub access in Fargate)
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# ----------------- App code -----------------
# Put your FastAPI worker at repo root as main.py
COPY main.py /app/main.py

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8080"]
