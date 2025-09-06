import os, json, tempfile, threading, sys, time, logging, subprocess
from pathlib import Path
from typing import Optional, List
from datetime import datetime

import boto3
from fastapi import FastAPI

import torch
from diffusers import AutoPipelineForText2Video

# -------- logging --------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("worker")

# ---------- Config ----------
QUEUE_URL       = os.getenv("QUEUE_URL")   # e.g. https://sqs.us-east-1.amazonaws.com/123/video-jobs
AWS_REGION      = os.getenv("AWS_REGION", "us-east-1")
MODEL_DIR       = os.getenv("MODEL_DIR", "/opt/models/wan2.2-ti2v-5b")
MODEL_S3_PREFIX = os.getenv("MODEL_S3_PREFIX")  # e.g. s3://story-video-gen-model/wan2.2/
DYNAMODB_TABLE  = os.getenv("DYNAMODB_TABLE")  # required
OUTPUT_BUCKET   = os.getenv("OUTPUT_BUCKET", "story-video-data")  # configurable

if not QUEUE_URL:
    print("[ERROR] QUEUE_URL is required", file=sys.stderr)
    sys.exit(1)
if not DYNAMODB_TABLE:
    print("[ERROR] DYNAMODB_TABLE is required", file=sys.stderr)
    sys.exit(1)
if not MODEL_S3_PREFIX:
    print("[ERROR] MODEL_S3_PREFIX is required (e.g., s3://bucket/prefix/)", file=sys.stderr)
    sys.exit(1)

# Optional: keep HuggingFace offline at runtime so nothing tries to hit the hub
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# AWS clients
s3        = boto3.client("s3", region_name=AWS_REGION)
sqs       = boto3.client("sqs", region_name=AWS_REGION)
dynamodb  = boto3.client("dynamodb", region_name=AWS_REGION)

# ---------- Helpers ----------
def _iso_now_ms() -> str:
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"

def _parse_s3_uri(s3_uri: str):
    assert s3_uri.startswith("s3://"), f"Invalid S3 URI: {s3_uri}"
    _, _, rest = s3_uri.partition("s3://")
    bucket, _, key = rest.partition("/")
    if not bucket:
        raise ValueError(f"Invalid S3 URI (missing bucket): {s3_uri}")
    # key may be empty if prefix is root
    return bucket, key.rstrip("/") + ("/" if key and not key.endswith("/") else "")

def _upload_s3(from_path: Path, s3_uri: str):
    b, k = _parse_s3_uri(s3_uri)
    s3.upload_file(str(from_path), b, k)
    return b, k

def _safe_int(v, default):
    try:
        return int(v)
    except Exception:
        return default

def _safe_float(v, default):
    try:
        return float(v)
    except Exception:
        return default

# ---------- S3 model sync ----------
def _ensure_model_local_from_s3(prefix_uri: str, dest_dir: str):
    """
    Sync all objects under s3://bucket/prefix/ to dest_dir (no deletion).
    Skips downloading files that already exist with the same size.
    """
    bucket, prefix = _parse_s3_uri(prefix_uri)
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    paginator = s3.get_paginator("list_objects_v2")
    found_any = False
    log.info(f"[model] syncing from s3://{bucket}/{prefix} -> {dest}")

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        contents = page.get("Contents", [])
        for obj in contents:
            key = obj["Key"]
            if key.endswith("/"):
                # "folder" placeholder
                continue
            found_any = True
            rel = key[len(prefix):] if key.startswith(prefix) else key
            local_path = dest / rel
            local_path.parent.mkdir(parents=True, exist_ok=True)

            size_remote = obj.get("Size", None)
            if local_path.exists() and size_remote is not None:
                try:
                    if local_path.stat().st_size == size_remote:
                        continue  # skip identical size
                except Exception:
                    pass

            log.info(f"[model] download s3://{bucket}/{key} -> {local_path}")
            s3.download_file(bucket, key, str(local_path))

    if not found_any:
        raise RuntimeError(
            f"No objects under s3://{bucket}/{prefix}. "
            f"Check your MODEL_S3_PREFIX and IAM permissions."
        )

    # sanity check
    mi = Path(dest_dir) / "model_index.json"
    if not mi.exists():
        raise RuntimeError(f"model_index.json not found in {dest_dir}. "
                           f"Ensure you synced the full Diffusers repo structure.")

# ---------- DynamoDB helpers ----------
def is_task_completed(parent_id: str, task_id: str) -> bool:
    try:
        response = dynamodb.get_item(
            TableName=DYNAMODB_TABLE,
            Key={'id': {'S': parent_id}, 'task_id': {'S': task_id}}
        )
        if 'Item' in response:
            item = response['Item']
            if 'status' in item and item['status']['S'] == 'COMPLETED':
                return True
        return False
    except Exception as e:
        log.error(f"[dynamodb] Failed to check task status for {task_id}: {e}")
        raise RuntimeError(f"Unable to verify task status for {task_id}. DynamoDB check failed: {e}") from e

def update_task_status(parent_id: str, task_id: str, status: str, s3_url: str = None):
    try:
        set_parts = ['#status = :status', '#date_updated = :date_updated']
        expr_names = {'#status': 'status', '#date_updated': 'date_updated'}
        expr_values = {':status': {'S': status}, ':date_updated': {'S': _iso_now_ms()}}
        if s3_url:
            set_parts.append('#media_url = :media_url')
            expr_names['#media_url'] = 'media_url'
            expr_values[':media_url'] = {'S': s3_url}
        update_expression = f"SET {', '.join(set_parts)}"
        if status == "COMPLETED":
            update_expression += " REMOVE sparse_gsi_hash_key"
        response = dynamodb.update_item(
            TableName=DYNAMODB_TABLE,
            Key={'id': {'S': parent_id}, 'task_id': {'S': task_id}},
            UpdateExpression=update_expression,
            ExpressionAttributeNames=expr_names,
            ExpressionAttributeValues=expr_values,
            ReturnValues='UPDATED_NEW'
        )
        log.info(f"[dynamodb] Updated task {task_id} status to {status}")
        return response
    except Exception as e:
        log.error(f"[dynamodb] Failed to update task {task_id}: {e}")
        raise

# ---------- Startup: model sync + load pipeline ----------
# 1) Make sure CUDA exists (we’re on EC2 GPU)
if not torch.cuda.is_available():
    log.error("[fatal] CUDA is not available. Wan 2.2 TI2V-5B requires a GPU-equipped host (e.g., g5/g6).")
    sys.exit(1)

# 2) Pull model from S3 if missing or incomplete
try:
    need_sync = not (os.path.isdir(MODEL_DIR) and (Path(MODEL_DIR) / "model_index.json").exists())
    if need_sync:
        _ensure_model_local_from_s3(MODEL_S3_PREFIX, MODEL_DIR)
    else:
        log.info(f"[model] using existing local model at {MODEL_DIR}")
except Exception as e:
    log.error(f"[fatal] Model sync failed: {e}")
    sys.exit(1)

# 3) Load the model
device = "cuda"
dtype = torch.float16

try:
    pipe = AutoPipelineForText2Video.from_pretrained(
        MODEL_DIR,
        torch_dtype=dtype,
        local_files_only=True,  # no hub needed at runtime
    ).to(device)

    # Memory/perf tweaks
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    pipe.set_progress_bar_config(disable=True)

    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass

    log.info("[model] Wan 2.2 TI2V-5B pipeline loaded")
except Exception as e:
    log.error(f"[fatal] Failed to load Wan 2.2 TI2V-5B from {MODEL_DIR}: {e}")
    sys.exit(1)

# ---------- FastAPI ----------
app = FastAPI()

@app.get("/healthz")
def healthz():
    """Health check endpoint for ECS"""
    return {
        "ok": True,
        "timestamp": _iso_now_ms(),
        "status": "healthy",
        "model_loaded": os.path.isdir(MODEL_DIR) and (Path(MODEL_DIR) / "model_index.json").exists(),
        "cuda": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "fp16": True
    }

# ---------- Video generation ----------
def generate_video_mp4(
    prompt: str,
    negative_prompt: Optional[str],
    seconds: int,
    fps: int,
    width: int,
    height: int,
    seed: Optional[int],
) -> Path:
    """
    Generates frames with Wan 2.2 TI2V-5B and muxes into an MP4 with ffmpeg.
    Returns local path to the MP4 file.
    """
    if seconds <= 0: seconds = 5
    if fps <= 0: fps = 24
    if width <= 0 or height <= 0:
        width, height = 1280, 720

    num_frames = seconds * fps
    log.info(f"[t2v] prompt='{prompt[:80]}...', frames={num_frames}, fps={fps}, size={width}x{height}, seed={seed}")

    generator = torch.Generator(device=device)
    if seed is not None:
        try:
            generator.manual_seed(int(seed))
        except Exception:
            pass

    t0 = time.time()
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_frames=num_frames,
        height=height,
        width=width,
        generator=generator,
    )
    log.info(f"[t2v] frames generated in {time.time() - t0:.2f}s")

    frames = getattr(result, "frames", None) or getattr(result, "images", None)
    if frames is None:
        raise RuntimeError("No frames returned by the pipeline")

    from PIL import Image
    norm_frames: List[Image.Image] = []
    for fr in frames:
        if isinstance(fr, Image.Image):
            norm_frames.append(fr)
        else:
            import numpy as np
            if torch.is_tensor(fr):
                fr = fr.detach().float().clamp(0, 1).mul(255).to(torch.uint8).cpu().numpy()
            if isinstance(fr, np.ndarray):
                if fr.ndim == 3 and fr.shape[0] in (1, 3):  # CHW -> HWC
                    fr = np.transpose(fr, (1, 2, 0))
                norm_frames.append(Image.fromarray(fr.astype("uint8"), mode="RGB"))
            else:
                raise RuntimeError("Unsupported frame type from pipeline")

    td = tempfile.TemporaryDirectory()
    td_path = Path(td.name)
    for i, im in enumerate(norm_frames):
        im.convert("RGB").save(td_path / f"f_{i:05d}.png", format="PNG")

    mp4_path = td_path / "out.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(td_path / "f_%05d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-vf", f"scale={width}:{height}:flags=lanczos",
        str(mp4_path)
    ]
    log.info(f"[ffmpeg] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    log.info(f"[t2v] mp4 muxed @ {mp4_path}")
    return mp4_path

# ---------- SQS + Dynamo logic (unchanged) ----------
def process_job(job: dict):
    class SQSMessageValidationError(Exception):
        pass

    # Unwrap SNS envelope if present
    actual_job = job
    if job.get("Type") == "Notification" and "Message" in job:
        try:
            actual_job = json.loads(job["Message"])
            log.info("[sns] Extracted job from SNS notification envelope")
        except json.JSONDecodeError as e:
            raise SQSMessageValidationError(f"Failed to parse SNS Message field as JSON: {e}")

    required = ["parent_id", "task_id", "prompt"]
    missing = [f for f in required if f not in actual_job]
    if missing:
        raise SQSMessageValidationError(f"Job missing required fields: {', '.join(missing)}")

    parent_id = actual_job["parent_id"]
    task_id   = actual_job["task_id"]
    prompt    = actual_job["prompt"]
    neg       = actual_job.get("negative_prompt", None)
    seed      = actual_job.get("seed", None)
    seconds   = _safe_int(actual_job.get("seconds", 5), 5)
    fps       = _safe_int(actual_job.get("fps", 24), 24)
    width     = _safe_int(actual_job.get("width", 1280), 1280)
    height    = _safe_int(actual_job.get("height", 720), 720)

    out_s3 = f"s3://{OUTPUT_BUCKET}/{parent_id}/{task_id}.mp4"

    if is_task_completed(parent_id, task_id):
        log.info(f"[skip] Task {task_id} already completed, skipping")
        return

    update_task_status(parent_id, task_id, "IN_PROGRESS")
    log.info(f"[processing] parent_id={parent_id} task_id={task_id} prompt='{prompt[:120]}...' size={width}x{height}@{fps}fps secs={seconds}")

    t_start = time.time()
    try:
        mp4_path = generate_video_mp4(
            prompt=prompt,
            negative_prompt=neg,
            seconds=seconds,
            fps=fps,
            width=width,
            height=height,
            seed=seed,
        )
        _upload_s3(Path(mp4_path), out_s3)
        update_task_status(parent_id, task_id, "COMPLETED", out_s3)
        log.info(f"[done] Uploaded video -> {out_s3}")
    except Exception as e:
        log.error(f"[processing] failed: {e}")
        try:
            update_task_status(parent_id, task_id, "FAILED")
        except Exception as u:
            log.error(f"[processing] status update failed: {u}")
        raise
    finally:
        log.info(f"[done] Task {task_id} finished in {time.time() - t_start:.2f}s")

def worker_loop():
    print("[worker] SQS long-poll loop started")
    visibility = int(os.getenv("SQS_VISIBILITY_TIMEOUT", "3600"))
    while True:
        resp = sqs.receive_message(
            QueueUrl=QUEUE_URL,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=20,
            VisibilityTimeout=visibility
        )
        for m in resp.get("Messages", []):
            rcpt = m["ReceiptHandle"]
            try:
                job = json.loads(m.get("Body", "{}"))
                process_job(job)
                sqs.delete_message(QueueUrl=QUEUE_URL, ReceiptHandle=rcpt)
            except Exception as e:
                print(f"[worker] job failed: {e}", file=sys.stderr)
                try:
                    # best-effort FAIL status
                    aj = job
                    if isinstance(job, dict) and job.get("Type") == "Notification" and "Message" in job:
                        try:
                            aj = json.loads(job["Message"])
                        except json.JSONDecodeError:
                            pass
                    if isinstance(aj, dict) and 'parent_id' in aj and 'task_id' in aj:
                        update_task_status(aj['parent_id'], aj['task_id'], "FAILED")
                        print(f"[worker] Updated task status to FAILED for: {aj['task_id']}")
                except Exception as status_error:
                    print(f"[worker] Failed to update task status to FAILED: {status_error}", file=sys.stderr)

app = FastAPI()

@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "timestamp": _iso_now_ms(),
        "status": "healthy",
        "model_loaded": os.path.isdir(MODEL_DIR) and (Path(MODEL_DIR) / "model_index.json").exists(),
        "cuda": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "fp16": True
    }

@app.on_event("startup")
def _start_worker():
    t = threading.Thread(target=worker_loop, daemon=True)
    t.start()
