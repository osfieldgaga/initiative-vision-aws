"""Run the lld.py lane pipeline on a session's frames stored in S3.

Reads frames from s3://BUCKET/FRAMES_PREFIX, runs lld.process_frame on each,
and uploads per-session results to:

    s3://BUCKET/OUTPUT_PREFIX/<session start timestamp>/overlay.mp4
    s3://BUCKET/OUTPUT_PREFIX/<session start timestamp>/bev.mp4
    s3://BUCKET/OUTPUT_PREFIX/<session start timestamp>/metrics.json

OUTPUT_PREFIX must not be under sessions/ -- that prefix fires the on-upload Lambda.

Credentials: on EC2 attach an instance role; boto3 picks it up automatically. It needs
s3:ListBucket on arn:aws:s3:::BUCKET, s3:GetObject on FRAMES_PREFIX*, s3:PutObject on OUTPUT_PREFIX/*.

Watching it live: the script serves the overlay, BEV and metrics on 127.0.0.1:LIVE_PORT
(see live_view.py). From your laptop:

    ssh -L 8080:localhost:8080 ec2-user@<server>
    open http://localhost:8080

LIVE_PORT=0 disables it. LIVE_HOLD=1 keeps the page up after the run until Ctrl+C.
OpenCV windows still open if a display is available (DISPLAY set), otherwise headless.
"""
import json
import os
import tempfile
import threading
import time
from pathlib import Path, PurePosixPath

import boto3
import cv2 as cv
import numpy as np

import lld
from live_view import LiveView


## Config -- fill these in
BUCKET = "<your-bucket-name>"
FRAMES_PREFIX = "sessions/<session_id>/frames/"
OUTPUT_PREFIX = "lld"
WEIGHTS_PATH = lld.DEFAULT_WEIGHTS
REGION = "us-east-1"

FPS = 2.0
SHOW = bool(os.environ.get("DISPLAY"))
LIVE_PORT = int(os.environ.get("LIVE_PORT", 8080))  # 0 = no live view
LIVE_HOLD = bool(os.environ.get("LIVE_HOLD"))
FRAME_EXTENSIONS = (".jpg", ".jpeg", ".png")


def list_frame_keys(s3, bucket, prefix):
    """All frame keys under prefix, sorted (frame names sort chronologically).
    Paginated: a single list_objects_v2 call returns at most 1000 keys."""
    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].lower().endswith(FRAME_EXTENSIONS):
                keys.append(obj["Key"])
    keys.sort()
    return keys


def read_frame(s3, bucket, key):
    """Download one frame straight into memory and decode it to BGR. None if undecodable."""
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return cv.imdecode(np.frombuffer(body, np.uint8), cv.IMREAD_COLOR)


def session_id(frames_prefix):
    """sessions/20261001T170034Z-d015/frames/ -> 20261001T170034Z-d015"""
    return PurePosixPath(frames_prefix.rstrip("/")).parent.name


def session_timestamp(frames_prefix):
    """sessions/20261001T170034Z-d015/frames/ -> 20261001T170034Z (when the session started)"""
    return session_id(frames_prefix).split("-")[0]


def output_key(name, frames_prefix, output_prefix=OUTPUT_PREFIX):
    return f"{output_prefix}/{session_timestamp(frames_prefix)}/{name}"


def run(s3, bucket, frames_prefix, model, device, work_dir, show=SHOW, live=None):
    """Process every frame under frames_prefix and upload the videos + metrics.
    Returns the metrics document that was uploaded."""
    keys = list_frame_keys(s3, bucket, frames_prefix)
    if not keys:
        raise SystemExit(f"no frames found under s3://{bucket}/{frames_prefix}")
    print(f"{len(keys)} frames under s3://{bucket}/{frames_prefix}")

    work_dir = Path(work_dir)
    overlay_path = work_dir / "overlay.mp4"
    bev_path = work_dir / "bev.mp4"

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(str(overlay_path), fourcc, FPS, (lld.MODEL_SIZE, lld.MODEL_SIZE))  # size must match every frame you write, exactly
    bev = cv.VideoWriter(str(bev_path), fourcc, FPS, (lld.canvas_w, lld.canvas_h))

    frames = []
    started = time.time()
    try:
        for index, key in enumerate(keys):
            frame = read_frame(s3, bucket, key)
            if frame is None:
                print(f"Frame {index}: could not decode {key}, skipping")
                continue

            metrics, overlay, warped = lld.process_frame(frame, model, device)
            out.write(overlay)
            bev.write(warped)

            frames.append({"index": index, "key": key, "frame": PurePosixPath(key).name, **metrics})
            print(f"Frame {index}/{len(keys)} | {lld.format_metrics(metrics)}")

            if live:
                live.update(overlay, warped, metrics, index=index, total=len(keys), frame=PurePosixPath(key).name)

            if show:
                cv.imshow('YOLOPv2 Panoptic Perception Overlay', overlay)
                cv.imshow("Bird Eye's View - Lane Mask", warped)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    print("stopped early with q, uploading what was processed")
                    break
    finally:
        # mp4 is only finalized on release -- uploading before this gives an unplayable file
        out.release()
        bev.release()
        if show:
            cv.destroyAllWindows()
        if live:
            live.finish()

        doc = {
            "session_id": session_id(frames_prefix),
            "session_start": session_timestamp(frames_prefix),
            "source": f"s3://{bucket}/{frames_prefix}",
            "frames_total": len(keys),
            "frames_processed": len(frames),
            "frames_valid": sum(f["lane_valid"] for f in frames),
            "processing_seconds": round(time.time() - started, 2),
            "device": str(device),
            "params": {
                "lane_threshold": lld.LANE_THRESHOLD,
                "tolerance_px": lld.TOLERANCE,
                "degree": lld.DEGREE,
                "lookahead_y": lld.LOOKAHEAD_Y,
                "departure_threshold_m": lld.DEPARTURE_THRESHOLD,
                "fps": FPS,
            },
            "frames": frames,
        }

        if frames:
            upload(s3, bucket, frames_prefix, overlay_path, bev_path, doc)
        else:
            print("no frames processed, nothing uploaded")

    return doc


def upload(s3, bucket, frames_prefix, overlay_path, bev_path, doc):
    for path in (overlay_path, bev_path):
        key = output_key(path.name, frames_prefix)
        s3.upload_file(str(path), bucket, key, ExtraArgs={"ContentType": "video/mp4"})
        print(f"uploaded s3://{bucket}/{key}")

    key = output_key("metrics.json", frames_prefix)
    s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(doc, indent=2).encode(), ContentType="application/json")
    print(f"uploaded s3://{bucket}/{key}")


def main():
    if "<" in BUCKET or "<" in FRAMES_PREFIX:
        raise SystemExit("fill in BUCKET and FRAMES_PREFIX at the top of lld_aws.py first")
    if FRAMES_PREFIX.startswith(OUTPUT_PREFIX.rstrip("/") + "/") or OUTPUT_PREFIX.startswith("sessions"):
        raise SystemExit("OUTPUT_PREFIX must be separate from sessions/ (it would re-trigger the on-upload Lambda)")

    s3 = boto3.client("s3", region_name=REGION)
    model, device = lld.load_model(WEIGHTS_PATH)
    print(f"model on {device}, display: {SHOW}")

    live = None
    if LIVE_PORT:
        try:
            live = LiveView(port=LIVE_PORT, departure_threshold=lld.DEPARTURE_THRESHOLD).start()
        except OSError as e:
            print(f"live view disabled, could not bind port {LIVE_PORT}: {e}")

    try:
        with tempfile.TemporaryDirectory(prefix="lld_") as work_dir:
            run(s3, BUCKET, FRAMES_PREFIX, model, device, work_dir, live=live)

        if live and LIVE_HOLD:
            print("run finished, live view still up -- Ctrl+C to exit")
            try:
                threading.Event().wait()
            except KeyboardInterrupt:
                pass
    finally:
        if live:
            live.stop()


if __name__ == "__main__":
    main()
