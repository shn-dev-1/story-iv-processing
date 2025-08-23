import os, json, tempfile, threading, sys, re, time, logging
from pathlib import Path
from typing import Optional, List
from datetime import datetime

import numpy as np
from PIL import Image

import boto3
from fastapi import FastAPI

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

# Offline / cache envs
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HOME", "/opt/hfcache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/opt/hfcache/transformers")

from diffusers import OnnxStableDiffusionPipeline
from diffusers.schedulers import DPMSolverMultistepScheduler  # good CPU scheduler

# ---------- Config ----------
QUEUE_URL       = os.getenv("QUEUE_URL")   # e.g. https://sqs.us-east-1.amazonaws.com/123/imagen-jobs
AWS_REGION      = os.getenv("AWS_REGION", "us-east-1")
MODEL_DIR       = os.getenv("MODEL_DIR", "/opt/models/sd15-onnx")
DYNAMODB_TABLE  = os.getenv("DYNAMODB_TABLE") # DynamoDB table name from remote state
OUTPUT_BUCKET   = os.getenv("OUTPUT_BUCKET", "story-video-data")  # configurable

if not QUEUE_URL:
    print("[ERROR] QUEUE_URL is required", file=sys.stderr)
    sys.exit(1)

if not DYNAMODB_TABLE:
    print("[ERROR] DYNAMODB_TABLE environment variable is required but not set", file=sys.stderr)
    sys.exit(1)

# --- model preflight ---
if not (os.path.isdir(MODEL_DIR) and os.listdir(MODEL_DIR)):
    log.error(f"[fatal] MODEL_DIR '{MODEL_DIR}' not found or empty. Ensure the Docker build baked the model.")
    sys.exit(1)

s3        = boto3.client("s3", region_name=AWS_REGION)
sqs       = boto3.client("sqs", region_name=AWS_REGION)
dynamodb  = boto3.client("dynamodb", region_name=AWS_REGION)

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

def _iso_now_ms() -> str:
    # RFC3339-ish with milliseconds, UTC 'Z'
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"

def _snap8(x: int) -> int:
    # Clamp to reasonable range and snap to a multiple of 8
    x = max(64, min(1024, int(x)))
    return (x // 8) * 8

# ---------- DynamoDB helpers ----------
def is_task_completed(parent_id: str, task_id: str) -> bool:
    """
    Check if a task is already completed in DynamoDB.
    """
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
    """
    Update the status of a task in DynamoDB.
    """
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

# ---------- Load ONNX pipeline (CPU, offline) ----------
pipe = OnnxStableDiffusionPipeline.from_pretrained(
    MODEL_DIR,
    provider="CPUExecutionProvider",
    local_files_only=True,
    safety_checker=None,
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

app = FastAPI()

@app.get("/healthz")
def healthz():
    """Health check endpoint for ECS"""
    return {
        "ok": True,
        "timestamp": _iso_now_ms(),
        "status": "healthy",
        "model_loaded": os.path.isdir(MODEL_DIR) and len(os.listdir(MODEL_DIR)) > 0
    }

@app.on_event("startup")
def _start_worker():
    t = threading.Thread(target=worker_loop, daemon=True)
    t.start()

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
    # Deterministic CPU RNG if seed provided
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
    return out.images  # list[PIL.Image]

# ---------- Job processor ----------
def process_job(job: dict):
    """
    Expected SQS job message body (JSON):
    {
      "parent_id": "12312312",
      "task_id": "12312311",
      "prompt": "a cozy cabin in the forest, golden hour",
      "seed": 1234,             // optional
      "steps": 15,              // optional (default 15)
      "guidance": 7.0,          // optional (CFG)
      "width": 512,             // optional
      "height": 512,            // optional
      "num_images": 1,          // optional, <= 4 recommended for CPU
      "negative_prompt": ""     // optional
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
    required_fields = ["parent_id", "task_id", "prompt"]
    missing_fields = [f for f in required_fields if f not in actual_job]
    if missing_fields:
        raise SQSMessageValidationError(
            message=f"Job missing required fields: {', '.join(missing_fields)}",
            missing_fields=missing_fields,
            received_fields=actual_job
        )

    parent_id = actual_job["parent_id"]
    task_id   = actual_job["task_id"]
    prompt    = actual_job["prompt"]

    # Derive S3 output (configurable bucket)
    out_s3 = f"s3://{OUTPUT_BUCKET}/{parent_id}/{task_id}.png"

    # Idempotency: skip if already completed
    if is_task_completed(parent_id, task_id):
        log.info(f"[skip] Task {task_id} already completed, skipping processing")
        return

    log.info(f"[processing] Task {task_id} not completed, proceeding with processing")
    update_task_status(parent_id, task_id, "IN_PROGRESS")

    steps     = int(actual_job.get("steps", 15))
    guidance  = float(actual_job.get("guidance", 7.0))
    width     = _snap8(actual_job.get("width", 512))
    height    = _snap8(actual_job.get("height", 512))
    nimgs_req = int(actual_job.get("num_images", 1))
    nimgs     = max(1, min(nimgs_req, 4))   # cap for CPU sanity
    seed      = actual_job.get("seed", None)
    neg       = actual_job.get("negative_prompt", None)

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

    # Upload to S3
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        if nimgs == 1:
            out_path = td / "out.png"
            images[0].save(out_path, format="PNG")
            _upload_s3(out_path, out_s3)
            log.info(f"[done] uploaded image -> {out_s3}")
            try:
                out_path.unlink()
            except Exception:
                pass
        else:
            b, k = _parse_s3_uri(out_s3)
            base, ext = (k, ".png") if "." not in k else tuple(k.rsplit(".", 1))
            ext = "." + ext if not ext.startswith(".") else "." + ext.split(".")[-1]
            for i, im in enumerate(images):
                p = td / f"out_{i}.png"
                im.save(p, format="PNG")
                s3.upload_file(str(p), b, f"{base}-{i}{ext}")
                try:
                    p.unlink()
                except Exception:
                    pass
            log.info(f"[done] uploaded {nimgs} images to s3://{b}/{base}-[0..{nimgs-1}]{ext}")

        # Mark complete with media URL
        update_task_status(parent_id, task_id, "COMPLETED", out_s3)
        log.info(f"[done] updated DynamoDB task {task_id} to COMPLETED")

# ---------- Worker loop ----------
def worker_loop():
    print("[worker] SQS long-poll loop started")
    while True:
        resp = sqs.receive_message(
            QueueUrl=QUEUE_URL,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=20,        # long poll
            VisibilityTimeout=900      # tune to exceed worst-case generation time
        )
        for m in resp.get("Messages", []):
            rcpt = m["ReceiptHandle"]
            try:
                # Debug (short)
                body = m.get("Body", "{}")
                job = json.loads(body)
                process_job(job)
                sqs.delete_message(QueueUrl=QUEUE_URL, ReceiptHandle=rcpt)
            except SQSMessageValidationError as e:
                print(f"[worker] SQS message validation failed: {e}", file=sys.stderr)
                if e.missing_fields:
                    print(f"[worker] Missing fields: {e.missing_fields}", file=sys.stderr)
                if e.received_fields:
                    print(f"[worker] Received fields: {list(e.received_fields.keys())}", file=sys.stderr)
                # Consider deleting or routing to DLQ to avoid poison loops
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

# Move the startup event and health check to the main app instance above
