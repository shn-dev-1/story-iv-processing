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
 && pip install -r /app/requirements.txt

# ----------------- Model snapshot via ModelScope -----------------
# Where we’ll store the model at build time
# ----------------- Model snapshot via ModelScope -----------------
ENV MODEL_DIR=/opt/models/wan2.2-ti2v-5b
RUN mkdir -p "${MODEL_DIR}"

# Pull Wan 2.2 TI2V-5B from ModelScope (no HF auth needed)
RUN python - <<'PY'
import os, shutil
from pathlib import Path
from modelscope.hub.snapshot_download import snapshot_download

target = Path(os.environ.get("MODEL_DIR", "/opt/models/wan2.2-ti2v-5b"))
repo   = "Wan-AI/Wan2.2-TI2V-5B"

print(f"[modelscope] downloading {repo} -> {target}")
# ModelScope writes into a cache directory; returns the actual local path
actual = Path(snapshot_download(model_id=repo, cache_dir=str(target)))
print(f"[modelscope] download path: {actual}")

# Normalize: ensure final files live directly under MODEL_DIR
if actual != target:
    # If ModelScope created a nested dir, move its contents up into MODEL_DIR
    target.mkdir(parents=True, exist_ok=True)
    for item in actual.iterdir():
        dest = target / item.name
        if dest.exists():
            # remove existing dir/file to avoid collisions in layered builds
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        shutil.move(str(item), str(dest))
    # Remove the now-empty origin directory (if inside target)
    try:
        if actual.exists() and actual != target:
            shutil.rmtree(actual)
    except Exception as e:
        print(f"[modelscope] cleanup warning: {e}")

# Sanity check: Diffusers expects model_index.json at MODEL_DIR root
mi = target / "model_index.json"
if not mi.exists():
    raise SystemExit("[FATAL] model_index.json not found at " + str(mi))

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
