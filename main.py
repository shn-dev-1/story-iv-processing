import os, json, tempfile, threading, sys, re
from pathlib import Path
from typing import Optional, List
import numpy as np
from PIL import Image

import boto3
from fastapi import FastAPI

# Offline / cache envs
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HOME", "/opt/hfcache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/opt/hfcache/transformers")

from diffusers import OnnxStableDiffusionPipeline
from diffusers.schedulers import DPMSolverMultistepScheduler  # good CPU scheduler

# ---------- Config ----------
QUEUE_URL     = os.getenv("QUEUE_URL")   # e.g. https://sqs.us-east-1.amazonaws.com/123/imagen-jobs
AWS_REGION    = os.getenv("AWS_REGION", "us-east-1")
MODEL_DIR     = os.getenv("MODEL_DIR", "/opt/models/sd15-onnx")

if not QUEUE_URL:
    print("[ERROR] QUEUE_URL is required", file=sys.stderr)
    sys.exit(1)

s3  = boto3.client("s3", region_name=AWS_REGION)
sqs = boto3.client("sqs",  region_name=AWS_REGION)

# ---------- Helpers ----------
def _parse_s3_uri(s3_uri: str):
    assert s3_uri.startswith("s3://"), f"Invalid S3 URI: {s3_uri}"
    _, _, rest = s3_uri.partition("s3://")
    bucket, _, key = rest.partition("/")
    if not bucket or not key:
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    return bucket, key

def _upload_s3(from_path: Path, s3_uri: str):
    b, k = _parse_s3_uri(s3_uri)
    s3.upload_file(str(from_path), b, k)
    return b, k

# ---------- Load ONNX pipeline (CPU) ----------
# We load strictly from local files so this works behind a NAT-less Fargate task.
pipe = OnnxStableDiffusionPipeline.from_pretrained(
    MODEL_DIR,
    provider="CPUExecutionProvider",
    local_files_only=True,
    safety_checker=None,  # keep infra minimal; enforce your own moderation if needed
)
# Swap in a CPU-friendly scheduler (fewer steps works well)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

app = FastAPI()

@app.get("/healthz")
def healthz():
    return {"ok": True}

# ---------- Generation ----------
def generate_images(
    prompt: str,
    negative_prompt: Optional[str],
    steps: int,
    guidance: float,
    width: int,
    height: int,
    num_images: int,
    seed: Optional[int],
):
    # Diffusers ONNX allows numpy RandomState as generator for determinism. :contentReference[oaicite:3]{index=3}
    generator = np.random.RandomState(seed) if seed is not None else None

    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        height=int(height),
        width=int(width),
        num_images_per_prompt=int(num_images),
        generator=generator,
    )
    # .images is a list of PIL.Image objects
    return out.images

# ---------- Job processor ----------
def process_job(job: dict):
    """
    Expected SQS job message body (JSON):
    {
      "prompt": "a cozy cabin in the forest, golden hour",
      "image_out": "s3://my-bucket/out/job-123/sample.png",
      "seed": 1234,             // optional
      "steps": 15,              // optional (default 15)
      "guidance": 7.0,          // optional (CFG)
      "width": 512,             // optional
      "height": 512,            // optional
      "num_images": 1,          // optional, <= 4 recommended for CPU
      "negative_prompt": ""     // optional
    }
    """
    if "prompt" not in job or "image_out" not in job:
        raise ValueError("Job must include 'prompt' and 'image_out'")

    prompt = job["prompt"]
    out_s3 = job["image_out"]

    steps    = int(job.get("steps", 15))
    guidance = float(job.get("guidance", 7.0))
    width    = int(job.get("width", 512))
    height   = int(job.get("height", 512))
    nimgs    = int(job.get("num_images", 1))
    seed     = job.get("seed", None)
    neg      = job.get("negative_prompt", None)

    # Generate
    images: List[Image.Image] = generate_images(
        prompt=prompt,
        negative_prompt=neg,
        steps=steps,
        guidance=guidance,
        width=width,
        height=height,
        num_images=nimgs,
        seed=seed,
    )

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # If multiple images requested, suffix the key(s) with -i.png automatically
        if nimgs == 1:
            out_path = td / "out.png"
            images[0].save(out_path, format="PNG")
            _upload_s3(out_path, out_s3)
        else:
            # Derive prefix from provided key, append index
            b, k = _parse_s3_uri(out_s3)
            base, ext = (k, ".png") if "." not in k else tuple(k.rsplit(".", 1))
            ext = "." + ext if not ext.startswith(".") else "." + ext.split(".")[-1]
            for i, im in enumerate(images):
                p = td / f"out_{i}.png"
                im.save(p, format="PNG")
                s3.upload_file(str(p), b, f"{base}-{i}{ext}")

# ---------- Worker loop ----------
def worker_loop():
    print("[worker] SQS long-poll loop started")
    while True:
        resp = sqs.receive_message(
            QueueUrl=QUEUE_URL,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=20,        # long poll
            VisibilityTimeout=900      # SD on CPU can take time; adjust per task size
        )
        for m in resp.get("Messages", []):
            rcpt = m["ReceiptHandle"]
            try:
                job = json.loads(m["Body"])
                process_job(job)
                sqs.delete_message(QueueUrl=QUEUE_URL, ReceiptHandle=rcpt)
            except Exception as e:
                print(f"[worker] job failed: {e}", file=sys.stderr)
                # Optional: DLQ or change visibility here.

@app.on_event("startup")
def _start_worker():
    t = threading.Thread(target=worker_loop, daemon=True)
    t.start()
