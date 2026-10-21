"""Lambda function: archives every live frame the Pi uploads and queues it for processing.

The Pi (upload_live_frame.py) keeps overwriting one key, live/latest_frame.jpg, with
x-amz-meta-session-start / frame-time / frame-seq metadata. Each overwrite fires this
function, which:
  1. copies *that exact upload* to live/<session-start>/<frame-time>.jpg
  2. sends {bucket, key, session_start, frame_time, frame_seq} to an SQS queue, which
     lld_aws.py (on EC2) consumes to process frames as they arrive

Why the versionId matters: the Pi uploads ~2 frames/s, and by the time this function runs
latest_frame.jpg may already hold the next frame. With bucket versioning on, every S3 event
carries the versionId of the upload that fired it, so the copy targets that exact version --
no duplicated or skipped frames. Without versioning it falls back to copying whatever is
current (and logs a warning).

Config (Lambda environment variables):
  QUEUE_URL  -- SQS queue URL; a FIFO queue (.fifo) is recommended so frames stay in order and
                retried invocations don't double-queue. Unset = archive only, no queueing.

Permissions this needs (set up in phase2/live-pipeline-setup.md, not by this code):
  - S3 notification on the bucket: ObjectCreated, prefix filter = live/latest_frame.jpg
    (the exact key -- otherwise the archive copies under live/ would re-trigger this)
  - execution role: s3:GetObject + s3:GetObjectVersion on live/latest_frame.jpg,
    s3:PutObject on live/*, sqs:SendMessage on the queue
"""

from __future__ import annotations

import json
import os
import urllib.parse
from datetime import datetime, timezone

try:
    import boto3

    s3 = boto3.client("s3")
    sqs = boto3.client("sqs")
except ImportError:
    # Real Lambda always has boto3; this only matters for importing this file in tests.
    s3 = None
    sqs = None

LIVE_KEY = "live/latest_frame.jpg"
LIVE_PREFIX = "live"
UNKNOWN_SESSION = "unknown-session"  # frames uploaded without session-start metadata


def parse_event(event: dict) -> list:
    """[{bucket, key, version_id, event_time}, ...] for every S3 object this event is about."""
    out = []
    for record in event.get("Records", []):
        s3_info = record.get("s3", {})
        bucket = s3_info.get("bucket", {}).get("name")
        obj = s3_info.get("object", {})
        # S3 event keys are URL-encoded (e.g. spaces become '+')
        key = urllib.parse.unquote_plus(obj.get("key", ""))
        if bucket and key:
            out.append({
                "bucket": bucket,
                "key": key,
                "version_id": obj.get("versionId"),  # only present when versioning is on
                "event_time": record.get("eventTime"),
            })
    return out


def event_time_to_epoch(event_time: str | None) -> float:
    """'2026-10-20T14:05:12.512Z' -> epoch seconds; now() if missing/unparseable."""
    if event_time:
        for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
            try:
                return datetime.strptime(event_time, fmt).replace(tzinfo=timezone.utc).timestamp()
            except ValueError:
                pass
    return datetime.now(timezone.utc).timestamp()


def frame_name(frame_time: float) -> str:
    """1792505112.512 -> '20261020T140512.512Z.jpg' (UTC, sorts chronologically as text)."""
    dt = datetime.fromtimestamp(frame_time, timezone.utc)
    return f"{dt.strftime('%Y%m%dT%H%M%S')}.{dt.microsecond // 1000:03d}Z.jpg"


def archive_key(session_start: str, frame_time: float) -> str:
    return f"{LIVE_PREFIX}/{session_start}/{frame_name(frame_time)}"


def frame_info(metadata: dict, event_time: str | None) -> tuple:
    """(session_start, frame_time, frame_seq) from the object's metadata, with fallbacks."""
    session_start = metadata.get("session-start") or UNKNOWN_SESSION
    try:
        frame_time = float(metadata["frame-time"])
    except (KeyError, ValueError):
        frame_time = event_time_to_epoch(event_time)
    try:
        frame_seq = int(metadata["frame-seq"])
    except (KeyError, ValueError):
        frame_seq = None
    return session_start, frame_time, frame_seq


def queue_message(queue_url: str, body: dict) -> dict:
    """send_message kwargs; FIFO queues additionally need a group and a dedup id."""
    kwargs = {"QueueUrl": queue_url, "MessageBody": json.dumps(body)}
    if queue_url.endswith(".fifo"):
        kwargs["MessageGroupId"] = body["session_start"]  # in-order within a session
        kwargs["MessageDeduplicationId"] = body["key"][-128:]  # Lambda retries won't double-queue
    return kwargs


def handler(event, context):
    queue_url = os.environ.get("QUEUE_URL")
    results = []
    for rec in parse_event(event):
        bucket, key, version_id = rec["bucket"], rec["key"], rec["version_id"]
        if key != LIVE_KEY:
            # Guard in case the notification filter is wider than the exact key:
            # our own archive copies under live/ must never be re-archived.
            print(f"skipping s3://{bucket}/{key}: not {LIVE_KEY}")
            continue

        version = {"VersionId": version_id} if version_id else {}
        if not version_id:
            print("WARNING: no versionId in event (bucket versioning off?) -- copying the current "
                  "object, which may already be a newer frame")

        head = s3.head_object(Bucket=bucket, Key=key, **version)
        session_start, frame_time, frame_seq = frame_info(head.get("Metadata", {}), rec["event_time"])

        dest = archive_key(session_start, frame_time)
        # metadata is copied along by default (MetadataDirective=COPY)
        s3.copy_object(Bucket=bucket, Key=dest, CopySource={"Bucket": bucket, "Key": key, **version})
        print(f"archived s3://{bucket}/{key} (version {version_id}) -> {dest}")

        body = {"bucket": bucket, "key": dest, "session_start": session_start,
                "frame_time": frame_time, "frame_seq": frame_seq}
        if queue_url:
            sqs.send_message(**queue_message(queue_url, body))
        results.append(body)

    return {"statusCode": 200, "body": json.dumps({"archived": results})}
