"""Lambda function: runs the instant a new object lands in the S3 bucket.

This is deliberately trivial -- it does NOT run any image processing (Lambda
can't run COOL; see sprint-1-aws-fundamentals.md for why). Its only job here
is to prove the trigger wiring works, by doing two things you can verify
without guessing:
  1. logs the event (bucket, key, size) to CloudWatch Logs
  2. copies the uploaded object to a parallel "processed/" prefix in the
     same bucket, so success is visible as a real S3 object, not just a
     log line you have to go find

Deploy: paste this as the function body when creating a Lambda (Python 3.12
runtime, x86 or arm64 either is fine for something this small), OR zip it
and upload via the console/CLI. The exact console steps are in the sprint 1
walkthrough. This file's only entry point AWS calls is `handler`.

Permissions this needs (set up in the walkthrough, not by this code):
  - The S3 bucket's notification configuration must point at this function.
  - This function's execution role needs s3:GetObject and s3:PutObject on
    the bucket (read the upload, write the copy).
  - The bucket needs to be allowed to invoke this function (a resource-based
    policy on the Lambda, which the console sets up automatically when you
    add the S3 trigger through the Lambda console's "Add trigger" button).
"""

from __future__ import annotations

import json
import urllib.parse

try:
    import boto3

    s3 = boto3.client("s3")
except ImportError:
    # Real Lambda always has boto3 available in its runtime; this only
    # matters for importing this file outside Lambda (e.g. to unit test
    # parse_event below without needing the AWS SDK at all).
    s3 = None

PROCESSED_PREFIX = "processed"


def parse_event(event: dict) -> list:
    """[(bucket, key, size_bytes), ...] for every S3 object this event is about.

    A single S3 event notification can (rarely) batch more than one record,
    so this always returns a list, even for the common single-record case.
    Pulled out from handler() so it's testable with a plain dict, no mocking
    of AWS internals needed.
    """
    out = []
    for record in event.get("Records", []):
        s3_info = record.get("s3", {})
        bucket = s3_info.get("bucket", {}).get("name")
        # S3 event keys are URL-encoded (e.g. spaces become '+'); undo that
        # or a key with special characters silently fails to be found.
        raw_key = s3_info.get("object", {}).get("key", "")
        key = urllib.parse.unquote_plus(raw_key)
        size = s3_info.get("object", {}).get("size", 0)
        if bucket and key:
            out.append((bucket, key, size))
    return out


def processed_key(original_key: str) -> str:
    """sessions/x/frames/a.jpg -> processed/sessions/x/frames/a.jpg"""
    return f"{PROCESSED_PREFIX}/{original_key}"


def handler(event, context):
    uploads = parse_event(event)
    results = []
    for bucket, key, size in uploads:
        print(f"triggered by s3://{bucket}/{key} ({size} bytes)")
        if key.startswith(f"{PROCESSED_PREFIX}/"):
            # Copying into processed/ would itself trigger this function again
            # if the notification isn't scoped to skip that prefix -- this
            # guard makes the function safe even if that scoping is missed.
            print(f"  skipping: already under {PROCESSED_PREFIX}/")
            continue
        dest_key = processed_key(key)
        s3.copy_object(Bucket=bucket, CopySource={"Bucket": bucket, "Key": key}, Key=dest_key)
        print(f"  copied to s3://{bucket}/{dest_key}")
        results.append({"source": key, "dest": dest_key})
    return {"statusCode": 200, "body": json.dumps({"processed": results})}
