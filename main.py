import os, json, tempfile, threading, sys, time, logging, subprocess, shutil
from pathlib import Path
from typing import Optional, List, Tuple
from datetime import datetime

import boto3
from fastapi import FastAPI

import torch
from diffusers import AutoPipelineForText2Video

# -------- Custom Exceptions --------
class SQSMessageValidationError(Exception):
    """Raised when an SQS message fails validation."""
    def __init__(self, message: str, missing_fields: list = None, received_fields: dict = None):
        self.message = message
        self.missing_fields = missing_fields or []
        self.received_fields = received_fields or {}
        super().__init__(self.message)

    def __str__(self):
        return f"SQSMessageValidationError: {self.message}"

# -------- logging --------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("worker")

# ---- Offline Hub defaults (model is baked at build time) ----
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# ---------- Config ----------
QUEUE_URL       = os.getenv("QUEUE_URL")   # e.g. https://sqs.us-east-1.amazonaws.com/123/video-jobs
AWS_REGION      = os.getenv("AWS_REGION", "us-east-1")
MODEL_DIR       = os.getenv("MODEL_DIR", "/opt/models/wan2.2-ti2v-5b")
DYNAMODB_TABLE  = os.getenv("DYNAMODB_TABLE")  # required
OUTPUT_BUCKET   = os.getenv("OUTPUT_BUCKET", "story-video-data")  # configurable

if not QUEUE_URL:
    print("[ERROR] QUEUE_URL is required", file=sys.stderr)
    sys.exit(1)
if not DYNAMODB_TABLE:
    print("[ERROR] DYNAMODB_TABLE is required", file=sys.stderr)
    sys.exit(1)

# --- model preflight ---
if not (os.path.isdir(MODEL_DIR) and os.listdir(MODEL_DIR)):
    log.error(f"[fatal] MODEL_DIR '{MODEL_DIR}' not found or empty. Ensure the Docker build baked the model.")
    sys.exit(1)

# --- GPU preflight ---
if not torch.cuda.is_available():
    log.error("[fatal] CUDA is not available. Wan 2.2 TI2V-5B requires a GPU-equipped host (ECS EC2 w/ g5/g6).")
    sys.exit(1)

# --- ffmpeg preflight ---
try:
    if subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode != 0:
        log.error("[fatal] ffmpeg not found in PATH")
        sys.exit(1)
except FileNotFoundError:
    log.error("[fatal] ffmpeg not found in PATH")
    sys.exit(1)

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
    if not bucket or not key:
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    return bucket, key

def _upload_s3(from_path: Path, s3_uri: str):
    b, k = _parse_s3_uri(s3_uri)
    s3.upload_file(str(from_path), b, k)
    return b, k

def _safe_int(v, default):
    try:
        return int(v)
    except Exception:
        return default

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

# ---------- Load Wan 2.2 TI2V-5B pipeline (GPU, offline) ----------
device = "cuda"
dtype = torch.float16

try:
    pipe = AutoPipelineForText2Video.from_pretrained(
        os.environ["MODEL_DIR"], 
        local_files_only=True, 
        torch_dtype=torch.float16
        ).to(device)

    # memory/perf tuning
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

    log.info("[model] Wan 2.2 TI2V-5B pipeline loaded on CUDA with fp16")

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
        "model_loaded": os.path.isdir(MODEL_DIR) and len(os.listdir(MODEL_DIR)) > 0,
        "cuda": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "fp16": True
    }

@app.on_event("startup")
def _start_worker():
    t = threading.Thread(target=worker_loop, daemon=True)
    t.start()

# ---------- Video generation ----------
def generate_video_mp4(
    prompt: str,
    negative_prompt: Optional[str],
    seconds: int,
    fps: int,
    width: int,
    height: int,
    seed: Optional[int],
) -> Tuple[Path, Path]:
    """
    Generates frames with Wan 2.2 TI2V-5B and muxes into an MP4 with ffmpeg.
    Returns (mp4_path, workdir). Caller is responsible for deleting workdir.
    """
    # sane defaults
    if seconds <= 0:
        seconds = 5
    if fps <= 0:
        fps = 24
    if width <= 0 or height <= 0:
        width, height = 1280, 720

    # snap to even dims for encoders (720p already is)
    width  = max(256, int(width)  // 2 * 2)
    height = max(256, int(height) // 2 * 2)

    num_frames = seconds * fps
    log.info(f"[t2v] prompt='{prompt[:80]}...', frames={num_frames}, fps={fps}, size={width}x{height}, seed={seed}")

    # seed
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
    gen_s = time.time() - t0
    log.info(f"[t2v] frames generated in {gen_s:.2f}s")

    # Normalize frames to list of PIL
    from PIL import Image
    frames = getattr(result, "frames", None) or getattr(result, "images", None)
    if frames is None:
        raise RuntimeError("No frames returned by the pipeline")

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

    # Use non-context tempdir so it persists until caller deletes it
    workdir = Path(tempfile.mkdtemp(prefix="t2v_"))
    for i, im in enumerate(norm_frames):
        im.convert("RGB").save(workdir / f"f_{i:05d}.png", format="PNG")

    mp4_path = workdir / "out.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(workdir / "f_%05d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-vf", f"scale={width}:{height}:flags=lanczos",
        str(mp4_path)
    ]
    log.info(f"[ffmpeg] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    log.info(f"[t2v] mp4 muxed @ {mp4_path}")

    return mp4_path, workdir

# ---------- Job processor ----------
def process_job(job: dict):
    """
    Expected SQS job body (or SNS Message JSON):
    {
      "parent_id": "12312312",
      "task_id": "12312311",
      "prompt": "a cinematic sunset over the ocean",
      "negative_prompt": "",       // optional
      "seed": 1234,                // optional
      "seconds": 5,                // default 5
      "fps": 24,                   // default 24
      "width": 1280,               // default 1280
      "height": 720                // default 720
    }
    """
    # Unwrap SNS envelope if present
    actual_job = job
    if job.get("Type") == "Notification" and "Message" in job:
        try:
            actual_job = json.loads(job["Message"])
            log.info("[sns] Extracted job from SNS notification envelope")
        except json.JSONDecodeError as e:
            raise SQSMessageValidationError(
                message=f"Failed to parse SNS Message field as JSON: {e}",
                missing_fields=[],
                received_fields=job
            )

    # Required fields
    required = ["parent_id", "task_id", "prompt"]
    missing = [f for f in required if f not in actual_job]
    if missing:
        raise SQSMessageValidationError(
            message=f"Job missing required fields: {', '.join(missing)}",
            missing_fields=missing,
            received_fields=actual_job
        )

    parent_id = actual_job["parent_id"]
    task_id   = actual_job["task_id"]
    prompt    = actual_job["prompt"]
    neg       = actual_job.get("negative_prompt", None)
    seed      = actual_job.get("seed", None)

    seconds   = _safe_int(actual_job.get("seconds", 5), 5)
    fps       = _safe_int(actual_job.get("fps", 24), 24)
    width     = _safe_int(actual_job.get("width", 1280), 1280)
    height    = _safe_int(actual_job.get("height", 720), 720)

    # S3 output key (.mp4)
    out_s3 = f"s3://{OUTPUT_BUCKET}/{parent_id}/{task_id}.mp4"

    # Idempotency
    if is_task_completed(parent_id, task_id):
        log.info(f"[skip] Task {task_id} already completed, skipping")
        return

    update_task_status(parent_id, task_id, "IN_PROGRESS")
    log.info(f"[processing] parent_id={parent_id} task_id={task_id} prompt='{prompt[:120]}...' size={width}x{height}@{fps}fps secs={seconds}")

    t_start = time.time()
    mp4_path: Optional[Path] = None
    workdir: Optional[Path] = None
    try:
        mp4_path, workdir = generate_video_mp4(
            prompt=prompt,
            negative_prompt=neg,
            seconds=seconds,
            fps=fps,
            width=width,
            height=height,
            seed=seed,
        )
        # Upload
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
        # Always cleanup workdir
        if workdir:
            try:
                shutil.rmtree(workdir, ignore_errors=True)
            except Exception as ce:
                log.warning(f"[cleanup] Failed to remove workdir {workdir}: {ce}")
        elapsed = time.time() - t_start
        log.info(f"[done] Task {task_id} finished in {elapsed:.2f}s")

# ---------- Worker loop ----------
def worker_loop():
    print("[worker] SQS long-poll loop started")
    # TI2V-5B @ 720p can be several minutes; keep visibility generous
    visibility = int(os.getenv("SQS_VISIBILITY_TIMEOUT", "3600"))
    while True:
        resp = sqs.receive_message(
            QueueUrl=QUEUE_URL,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=20,        # long poll
            VisibilityTimeout=visibility
        )
        for m in resp.get("Messages", []):
            rcpt = m["ReceiptHandle"]
            try:
                body = m.get("Body", "{}")
                job = json.loads(body)
                process_job(job)
                sqs.delete_message(QueueUrl=QUEUE_URL, ReceiptHandle=rcpt)
            except SQSMessageValidationError as e:
                print(f"[worker] SQS message validation failed: {e}", file=sys.stderr)
                # consider DLQ or delete to avoid poison-pill loops
                # sqs.delete_message(QueueUrl=QUEUE_URL, ReceiptHandle=rcpt)
            except Exception as e:
                print(f"[worker] job failed: {e}", file=sys.stderr)
                # Try to mark FAILED if we can extract ids
                try:
                    actual_job_for_status = job
                    if isinstance(job, dict) and job.get("Type") == "Notification" and "Message" in job:
                        try:
                            actual_job_for_status = json.loads(job["Message"])
                        except json.JSONDecodeError:
                            pass
                    if isinstance(actual_job_for_status, dict) and \
                       'parent_id' in actual_job_for_status and 'task_id' in actual_job_for_status:
                        update_task_status(actual_job_for_status['parent_id'],
                                           actual_job_for_status['task_id'],
                                           "FAILED")
                        print(f"[worker] Updated task status to FAILED for: {actual_job_for_status['task_id']}")
                except Exception as status_error:
                    print(f"[worker] Failed to update task status to FAILED: {status_error}", file=sys.stderr)
