import os, json, tempfile, threading, sys, re, time, logging
from pathlib import Path
from typing import Optional, List
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
QUEUE_URL     = os.getenv("QUEUE_URL")   # e.g. https://sqs.us-east-1.amazonaws.com/123/imagen-jobs
AWS_REGION    = os.getenv("AWS_REGION", "us-east-1")
MODEL_DIR     = os.getenv("MODEL_DIR", "/opt/models/sd15-onnx")
DYNAMODB_TABLE = os.getenv("DYNAMODB_TABLE") # DynamoDB table name from remote state

if not QUEUE_URL:
    print("[ERROR] QUEUE_URL is required", file=sys.stderr)
    sys.exit(1)

if not DYNAMODB_TABLE:
    print("[ERROR] DYNAMODB_TABLE environment variable is required but not set", file=sys.stderr)
    sys.exit(1)

s3  = boto3.client("s3", region_name=AWS_REGION)
sqs = boto3.client("sqs",  region_name=AWS_REGION)
dynamodb = boto3.client("dynamodb", region_name=AWS_REGION)

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

# ---------- DynamoDB helpers ----------
def is_task_completed(parent_id: str, task_id: str) -> bool:
    """
    Check if a task is already completed in DynamoDB.
    
    Args:
        parent_id: The partition key (parent_id)
        task_id: The sort key (task_id)
    
    Returns:
        bool: True if task is completed, False otherwise
    """
    try:
        response = dynamodb.get_item(
            TableName=DYNAMODB_TABLE,
            Key={
                'id': {'S': parent_id},
                'task_id': {'S': task_id}
            }
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
    
    Args:
        parent_id: The partition key (parent_id)
        task_id: The sort key (task_id)
        status: The status to set
        s3_url: Optional S3 URI to set in the media_url field
    """
    table_name = DYNAMODB_TABLE
    
    try:
        # Build update expression based on status and s3_url
        set_parts = ['#status = :status', '#date_updated = :date_updated']
        expression_attribute_names = {
            '#status': 'status',
            '#date_updated': 'date_updated'
        }
        expression_attribute_values = {
            ':status': {'S': status},
            ':date_updated': {'S': time.strftime('%Y-%m-%dT%H:%M:%S.%fZ', time.gmtime())}
        }
        
        # Add media_url if provided
        if s3_url:
            set_parts.append('#media_url = :media_url')
            expression_attribute_names['#media_url'] = 'media_url'
            expression_attribute_values['#media_url'] = {'S': s3_url}
        
        # Build the update expression with proper comma separation
        update_expression = f"SET {', '.join(set_parts)}"
        
        # Only remove sparse_gsi_hash_key if status is COMPLETED
        if status == "COMPLETED":
            update_expression += " REMOVE sparse_gsi_hash_key"
        
        response = dynamodb.update_item(
            TableName=table_name,
            Key={
                'id': {'S': parent_id},
                'task_id': {'S': task_id}
            },
            UpdateExpression=update_expression,
            ExpressionAttributeNames=expression_attribute_names,
            ExpressionAttributeValues=expression_attribute_values,
            ReturnValues='UPDATED_NEW'
        )
        log.info(f"[dynamodb] Updated task {task_id} status to {status}")
        return response
    except Exception as e:
        log.error(f"[dynamodb] Failed to update task {task_id}: {e}")
        raise

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
    # Handle SNS message envelope - extract the actual message
    actual_job = job
    if "Type" in job and job["Type"] == "Notification" and "Message" in job:
        try:
            # Parse the nested JSON message from SNS
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
    missing_fields = [field for field in required_fields if field not in actual_job]
    
    if missing_fields:
        raise SQSMessageValidationError(
            message=f"Job missing required fields: {', '.join(missing_fields)}",
            missing_fields=missing_fields,
            received_fields=actual_job
        )

    parent_id = actual_job["parent_id"]
    task_id = actual_job["task_id"]
    prompt = actual_job["prompt"]
    out_s3 = f"s3://story-video-data/{parent_id}/{task_id}.png"

    # Check if task is already completed to avoid duplicate processing
    if is_task_completed(parent_id, task_id):
        log.info(f"[skip] Task {task_id} already completed, skipping processing")
        return  # Exit early, message will be deleted by caller
    
    log.info(f"[processing] Task {task_id} not completed, proceeding with processing")
    
    # Update task status to IN_PROGRESS
    update_task_status(parent_id, task_id, "IN_PROGRESS")
    
    steps    = int(actual_job.get("steps", 15))
    guidance = float(actual_job.get("guidance", 7.0))
    width    = int(actual_job.get("width", 512))
    height   = int(actual_job.get("height", 512))
    nimgs    = int(actual_job.get("num_images", 1))
    seed     = actual_job.get("seed", None)
    neg      = actual_job.get("negative_prompt", None)

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
            log.info(f"[done] uploaded image -> {out_s3}")
            # Clean up local file after successful upload
            out_path.unlink()
            log.debug(f"[cleanup] deleted local file: {out_path}")
        else:
            # Derive prefix from provided key, append index
            b, k = _parse_s3_uri(out_s3)
            base, ext = (k, ".png") if "." not in k else tuple(k.rsplit(".", 1))
            ext = "." + ext if not ext.startswith(".") else "." + ext.split(".")[-1]
            for i, im in enumerate(images):
                p = td / f"out_{i}.png"
                im.save(p, format="PNG")
                s3.upload_file(str(p), b, f"{base}-{i}{ext}")
                # Clean up local file after successful upload
                p.unlink()
                log.debug(f"[cleanup] deleted local file: {p}")
            log.info(f"[done] uploaded {nimgs} images to {out_s3}")
        
        # Update DynamoDB task status to COMPLETED with S3 URI
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
            VisibilityTimeout=900      # SD on CPU can take time; adjust per task size
        )
        for m in resp.get("Messages", []):
            rcpt = m["ReceiptHandle"]
            try:
                # Debug: Log the raw message structure
                print(f"[debug] Raw SQS message: {m}")
                print(f"[debug] Message body: {m['Body']}")
                
                job = json.loads(m["Body"])
                print(f"[debug] Parsed job: {job}")
                
                process_job(job)
                sqs.delete_message(QueueUrl=QUEUE_URL, ReceiptHandle=rcpt)
            except SQSMessageValidationError as e:
                print(f"[worker] SQS message validation failed: {e}", file=sys.stderr)
                if e.missing_fields:
                    print(f"[worker] Missing fields: {e.missing_fields}", file=sys.stderr)
                if e.received_fields:
                    print(f"[worker] Received fields: {list(e.received_fields.keys())}", file=sys.stderr)
                
            except Exception as e:
                print(f"[worker] job failed: {e}", file=sys.stderr)
                
                # If validation passed but processing failed, update task status to FAILED
                try:
                    # Extract the actual job data (in case it's wrapped in SNS envelope)
                    actual_job_for_status = job
                    if "Type" in job and job["Type"] == "Notification" and "Message" in job:
                        try:
                            actual_job_for_status = json.loads(job["Message"])
                        except json.JSONDecodeError:
                            actual_job_for_status = job  # Fall back to original
                    
                    # Extract task IDs from the actual job for status update
                    if 'parent_id' in actual_job_for_status and 'task_id' in actual_job_for_status:
                        parent_id = actual_job_for_status['parent_id']
                        task_id = actual_job_for_status['task_id']
                        
                        # Update task status to FAILED
                        update_task_status(parent_id, task_id, "FAILED")
                        print(f"[worker] Updated task status to FAILED for: {task_id}")
                    else:
                        print("[worker] Could not update task status - missing required fields in job", file=sys.stderr)
                        
                except Exception as status_error:
                    print(f"[worker] Failed to update task status to FAILED: {status_error}", file=sys.stderr)

@app.on_event("startup")
def _start_worker():
    t = threading.Thread(target=worker_loop, daemon=True)
    t.start()
