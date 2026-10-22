"""Run the lld.py lane pipeline in the cloud on frames stored in S3.

Two modes:

    python lld_aws.py live            # default: process frames as the Pi uploads them
    python lld_aws.py replay <prefix> # re-run a stored session, e.g. live/20261020T140512Z/
                                      # or sessions/20261001T170034Z-d015/frames/

Live mode: the Pi overwrites live/latest_frame.jpg; the on_live_frame Lambda archives each
upload to live/<session start>/<frame time>.jpg and queues its key in SQS. This script
reads that queue, processes each frame, and writes per-session results:

    s3://BUCKET/OUTPUT_PREFIX/<session start>/overlay.mp4   (overlay_part2.mp4, ... if resumed)
    s3://BUCKET/OUTPUT_PREFIX/<session start>/bev.mp4
    s3://BUCKET/OUTPUT_PREFIX/<session start>/metrics.json  (checkpointed every CHECKPOINT_EVERY frames)

A session is finalized (videos uploaded) when frames from a new session arrive, or after
SESSION_IDLE_S without any frame. Restarting this script mid-session resumes it: metrics are
appended and the videos continue as a new part. Setup steps: phase2/live-pipeline-setup.md.

Watching it: the script serves the overlay, BEV and metrics on 127.0.0.1:LIVE_PORT
(see live_view.py). From your laptop:

    ssh -L 8080:localhost:8080 ec2-user@<server>
    open http://localhost:8080

LIVE_PORT=0 disables it. LIVE_HOLD=1 keeps the page up after a replay until Ctrl+C.
OpenCV windows still open if a display is available (DISPLAY set), otherwise headless.

Videos are written as mp4v (what OpenCV can always write) and converted to H.264 with ffmpeg
when the session is finalized, so they play in a browser. Without ffmpeg (sudo apt install ffmpeg)
they're uploaded as mp4v.

CloudWatch: per-frame lag, its hops, inference and processing time go to the InitiativeVision
namespace every METRICS_FLUSH_S (put_metric_data with value lists, so p50/p95 work). Needs
cloudwatch:PutMetricData on the instance role; LLD_METRICS=0 turns it off.

Public demo: LIVE_TOKEN_SECRET turns on the live page's signed-token check (see live_view.py),
and LLD_IDLE_STOP_MIN=N shuts the instance down (= stopped on EC2, so not billed) after N minutes
with no frame processed and no page request. SIGTERM (systemctl stop, shutdown) finalizes the
current session before exiting, like Ctrl+C.

Credentials: on EC2 attach an instance role; boto3 picks it up automatically.
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path, PurePosixPath

import boto3
import cv2 as cv
import numpy as np

import lld
import signs
from live_view import LiveView


## Config
BUCKET = "initiative-vision-1"
QUEUE_URL = os.environ.get("LLD_QUEUE_URL", "")  # SQS queue the Pi (or the Lambda) feeds, required for live
FRAMES_PREFIX = "live/<session_start>/"  # default for replay mode
OUTPUT_PREFIX = "lld"
WEIGHTS_PATH = os.environ.get("LLD_WEIGHTS", lld.DEFAULT_WEIGHTS)  # .onnx -> cv.dnn, .pt -> TorchScript
READ_SIGNS = os.environ.get("LLD_SIGNS", "1") != "0"  # stop / speed-limit signs (signs.py); LLD_SIGNS=0 = off
REGION = "us-east-1"

FPS = 2.0
SHOW = bool(os.environ.get("DISPLAY"))
LIVE_PORT = int(os.environ.get("LIVE_PORT", 8080))  # 0 = no live view
LIVE_HOLD = bool(os.environ.get("LIVE_HOLD"))
LIVE_TOKEN_SECRET = os.environ.get("LIVE_TOKEN_SECRET") or None
IDLE_STOP_MIN = float(os.environ.get("LLD_IDLE_STOP_MIN", 0))  # 0 = never
FRAME_EXTENSIONS = (".jpg", ".jpeg", ".png")

SESSION_IDLE_S = 30      # no frames for this long -> session finalized
CHECKPOINT_EVERY = 50    # frames between metrics.json uploads during a session
BACKLOG_EVERY = 10       # frames between queue-depth checks (shown on the live page)
RECEIVE_BATCH = 10       # SQS messages per receive (max 10)
# Live mode: a frame older than this is skipped if a newer one is already waiting, so the live
# view stays current when processing can't keep up. Skipped frames are still archived in S3 --
# `replay` processes everything. 0 = never skip (complete metrics, but the lag can grow).
MAX_LAG_S = float(os.environ.get("LLD_MAX_LAG_S", 10))
VISIBILITY_S = 120       # a received batch must be processed within this, or SQS re-delivers it

PUBLISH_METRICS = os.environ.get("LLD_METRICS", "1") != "0"
METRICS_NAMESPACE = "InitiativeVision"
METRICS_FLUSH_S = 10     # seconds between put_metric_data calls


## S3 helpers

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


def is_missing(error):
    """True if a boto3 error means 'object does not exist'."""
    code = getattr(error, "response", {}).get("Error", {}).get("Code")
    return code in ("NoSuchKey", "404", "NotFound")


def session_id(frames_prefix):
    """sessions/20261001T170034Z-d015/frames/ -> 20261001T170034Z-d015
       live/20261020T140512Z/                -> 20261020T140512Z"""
    path = PurePosixPath(frames_prefix.rstrip("/"))
    return path.parent.name if path.name == "frames" else path.name


def session_timestamp(session):
    """20261001T170034Z-d015 -> 20261001T170034Z (when the session started)"""
    return session.split("-")[0]


## Per-session outputs

class SessionOutput:
    """Videos + metrics for one session, uploaded to OUTPUT_PREFIX/<session start>/.

    resume=True picks up an existing metrics.json (live mode after a restart): known frames
    are skipped and the videos continue as overlay_partN.mp4 / bev_partN.mp4.
    resume=False starts over and overwrites (replay mode).
    """

    def __init__(self, s3, bucket, session, source, device, work_dir, resume=True):
        self.s3 = s3
        self.bucket = bucket
        self.session = session
        self.folder = session_timestamp(session)
        self.prefix = f"{OUTPUT_PREFIX}/{self.folder}"
        self.source = source
        self.device = str(device)
        self.started = time.time()
        self.last_frame_at = time.time()
        self.finalized = False

        self.frames, self.videos, self.skipped = self._load_existing() if resume else ([], [], 0)
        self.seen = {f["key"] for f in self.frames}
        self.new_frames = 0

        part = len(self.videos) + 1
        suffix = "" if part == 1 else f"_part{part}"
        local_dir = Path(work_dir) / self.folder
        local_dir.mkdir(parents=True, exist_ok=True)
        self.video_paths = {
            "overlay": local_dir / f"overlay{suffix}.mp4",
            "bev": local_dir / f"bev{suffix}.mp4",
        }
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        # size must match every frame you write, exactly
        self.writers = {
            "overlay": cv.VideoWriter(str(self.video_paths["overlay"]), fourcc, FPS, (lld.MODEL_SIZE, lld.MODEL_SIZE)),
            "bev": cv.VideoWriter(str(self.video_paths["bev"]), fourcc, FPS, (lld.canvas_w, lld.canvas_h)),
        }
        if self.frames:
            print(f"session {session}: resuming after {len(self.frames)} frames, video part {part}")

    def _load_existing(self):
        try:
            body = self.s3.get_object(Bucket=self.bucket, Key=f"{self.prefix}/metrics.json")["Body"].read()
        except Exception as e:
            if is_missing(e):
                return [], [], 0
            raise
        doc = json.loads(body)
        return doc.get("frames", []), doc.get("videos", []), doc.get("frames_skipped", 0)

    def add(self, key, metrics, overlay, warped, **extra):
        """Record one processed frame. False if this key was already recorded (re-delivery)."""
        if key in self.seen:
            return False
        self.seen.add(key)
        self.writers["overlay"].write(overlay)
        self.writers["bev"].write(warped)
        self.frames.append({"index": len(self.frames), "key": key, "frame": PurePosixPath(key).name,
                            **extra, **metrics})
        self.new_frames += 1
        self.last_frame_at = time.time()
        if self.new_frames % CHECKPOINT_EVERY == 0:
            try:
                self.upload_metrics()
            except Exception as e:
                print(f"checkpoint upload failed, retrying at the next one: {e}")  # boto3 already retried
        return True

    def doc(self):
        return {
            "session_id": self.session,
            "session_start": self.folder,
            "source": self.source,
            "frames_processed": len(self.frames),
            "frames_valid": sum(f["lane_valid"] for f in self.frames),
            "frames_skipped": self.skipped,  # live mode, too old (see MAX_LAG_S); replay to fill in
            "videos": self.videos,
            "device": self.device,
            "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "params": {
                "lane_threshold": lld.LANE_THRESHOLD,
                "tolerance_px": lld.TOLERANCE,
                "degree": lld.DEGREE,
                "lookahead_y": lld.LOOKAHEAD_Y,
                "departure_threshold_m": lld.DEPARTURE_THRESHOLD,
                "fps": FPS,
            },
            "frames": self.frames,
        }

    def upload_metrics(self):
        key = f"{self.prefix}/metrics.json"
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=json.dumps(self.doc(), indent=2).encode(),
                           ContentType="application/json")
        return key

    def finalize(self):
        """Close the videos (mp4 is only playable after release) and upload everything."""
        if self.finalized:
            return
        self.finalized = True
        for writer in self.writers.values():
            writer.release()

        if not self.new_frames:
            print(f"session {self.session}: no new frames, nothing uploaded")
            return

        part = {"frames": self.new_frames}
        for name, path in self.video_paths.items():
            to_h264(path)
            key = f"{self.prefix}/{path.name}"
            self.s3.upload_file(str(path), self.bucket, key, ExtraArgs={"ContentType": "video/mp4"})
            part[name] = key
            print(f"uploaded s3://{self.bucket}/{key}")
        self.videos.append(part)
        print(f"uploaded s3://{self.bucket}/{self.upload_metrics()}")


def to_h264(path):
    """Re-encode an mp4v file to H.264 in place (browsers don't play mp4v). True if converted.
    yuv420p + faststart: plays everywhere and starts before the whole file has downloaded."""
    if not shutil.which("ffmpeg"):
        print(f"ffmpeg not found, uploading {Path(path).name} as mp4v (won't play in browsers)")
        return False
    tmp = Path(path).with_suffix(".h264.mp4")
    result = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-crf", "23", "-preset", "veryfast", "-movflags", "+faststart", str(tmp)],
        capture_output=True, text=True)
    if result.returncode != 0 or not tmp.exists():
        print(f"H.264 conversion of {Path(path).name} failed, uploading mp4v: {result.stderr.strip()[-300:]}")
        tmp.unlink(missing_ok=True)
        return False
    os.replace(tmp, path)
    return True


## CloudWatch

class MetricsPublisher:
    """Collects per-frame values and sends them to CloudWatch every METRICS_FLUSH_S, from a
    background thread so the frame loop never waits on it. Values are sent as lists, which keeps
    the distribution: dashboards can show p50 / p95, not just averages. A failure (no permission,
    network) is printed once and never stops processing."""

    UNITS = {"FramesProcessed": "Count", "FramesSkipped": "Count", "LaneValid": "Percent",
             "Backlog": "Count"}

    def __init__(self, cloudwatch, namespace=METRICS_NAMESPACE, flush_s=METRICS_FLUSH_S,
                 dimensions=(("Pipeline", "live"),)):
        self.cw = cloudwatch
        self.namespace = namespace
        self.flush_s = flush_s
        self.dimensions = [{"Name": k, "Value": v} for k, v in dimensions]
        self.lock = threading.Lock()
        self.values = {}
        self.warned = False
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def add(self, name, value):
        if value is None:
            return
        with self.lock:
            self.values.setdefault(name, []).append(float(value))

    def record_frame(self, timing, lag_s, metrics, backlog=None):
        """One processed frame: the end-to-end lag, its hops, and the processing breakdown."""
        self.add("LagSeconds", lag_s)
        for key, name in (("capture_to_write_s", "CaptureToWriteSeconds"), ("pi_to_s3_s", "PiToS3Seconds"),
                          ("s3_to_queue_s", "S3ToQueueSeconds"), ("queue_s", "QueueSeconds"),
                          ("download_s", "DownloadSeconds"), ("infer_s", "InferenceSeconds"),
                          ("signs_s", "SignsSeconds"), ("total_s", "ProcessingSeconds")):
            self.add(name, timing.get(key))
        self.add("FramesProcessed", 1)
        self.add("LaneValid", 100.0 if metrics.get("lane_valid") else 0.0)
        self.add("Backlog", backlog)

    def datums(self, values):
        out = []
        for name, vals in values.items():
            unit = self.UNITS.get(name, "Seconds")
            if unit == "Count" and name.startswith("Frames"):  # counts: one summed datum
                out.append({"MetricName": name, "Dimensions": self.dimensions, "Value": sum(vals), "Unit": unit})
                continue
            for i in range(0, len(vals), 150):  # put_metric_data takes at most 150 values per datum
                out.append({"MetricName": name, "Dimensions": self.dimensions,
                            "Values": vals[i:i + 150], "Unit": unit})
        return out

    def flush(self):
        with self.lock:
            values, self.values = self.values, {}
        data = self.datums(values)
        for i in range(0, len(data), 1000):  # at most 1000 datums per call
            try:
                self.cw.put_metric_data(Namespace=self.namespace, MetricData=data[i:i + 1000])
            except Exception as e:
                if not self.warned:
                    print(f"CloudWatch metrics not sent (will keep trying quietly): {e}")
                    self.warned = True

    def _run(self):
        while not self.stop.wait(self.flush_s):
            self.flush()

    def close(self):
        self.stop.set()
        self.flush()


## Idle auto-stop

class IdleStop:
    """Shuts the machine down after idle_s without activity: a processed frame (touch()) or a
    live-page request (live.last_request). On EC2 a shutdown stops the instance."""

    def __init__(self, idle_s, live=None, shutdown=("sudo", "-n", "shutdown", "-h", "now"), check_s=30):
        self.idle_s, self.live, self.shutdown, self.check_s = idle_s, live, list(shutdown), check_s
        self.last = time.time()
        self.stop = threading.Event()
        self.fired = False

    def touch(self):
        self.last = time.time()

    def idle_for(self, now=None):
        last = max(self.last, self.live.last_request if self.live else 0.0)
        return (now or time.time()) - last

    def check(self, now=None):
        """Shut down if idle for too long; True if it did."""
        if self.fired or self.idle_for(now) < self.idle_s:
            return False
        self.fired = True
        print(f"idle for {self.idle_for(now) / 60:.0f} min: shutting down")
        subprocess.run(self.shutdown, check=False)
        return True

    def start(self):
        def run():
            while not self.stop.wait(self.check_s):
                self.check()
        threading.Thread(target=run, daemon=True).start()
        return self


def _sigterm(signum, frame):
    raise KeyboardInterrupt  # same clean exit as Ctrl+C: the current session is finalized


## Display helpers

def show_frame(overlay, warped):
    """OpenCV windows (needs a display). True if q was pressed."""
    cv.imshow('YOLOPv2 Panoptic Perception Overlay', overlay)
    cv.imshow("Bird Eye's View - Lane Mask", warped)
    return cv.waitKey(1) & 0xFF == ord('q')


## Modes

def run_replay(s3, bucket, frames_prefix, model, device, work_dir, show=SHOW, live=None, sign_reader=None):
    """Process every frame under frames_prefix in order. Returns the uploaded metrics doc."""
    keys = list_frame_keys(s3, bucket, frames_prefix)
    if not keys:
        raise SystemExit(f"no frames found under s3://{bucket}/{frames_prefix}")
    print(f"{len(keys)} frames under s3://{bucket}/{frames_prefix}")

    out = SessionOutput(s3, bucket, session_id(frames_prefix), f"s3://{bucket}/{frames_prefix}",
                        device, work_dir, resume=False)
    try:
        for index, key in enumerate(keys):
            frame = read_frame(s3, bucket, key)
            if frame is None:
                print(f"Frame {index}: could not decode {key}, skipping")
                continue

            metrics, overlay, warped = lld.process_frame(frame, model, device, sign_reader)
            out.add(key, metrics, overlay, warped)
            print(f"Frame {index}/{len(keys)} | {lld.format_metrics(metrics)}")

            if live:
                live.update(overlay, warped, metrics, session=out.session, index=index, total=len(keys),
                            frame=PurePosixPath(key).name)
            if show and show_frame(overlay, warped):
                print("stopped early with q, uploading what was processed")
                break
    finally:
        # also runs on a crash or Ctrl+C, so partial results still get uploaded
        out.finalize()
        if show:
            cv.destroyAllWindows()
        if live:
            live.finish()
    return out.doc()


def queue_depth(sqs, queue_url):
    """Frames waiting in the queue (approximate), or None if it can't be read."""
    try:
        attrs = sqs.get_queue_attributes(QueueUrl=queue_url, AttributeNames=["ApproximateNumberOfMessages"])
        return int(attrs["Attributes"]["ApproximateNumberOfMessages"])
    except Exception:
        return None  # e.g. role lacks sqs:GetQueueAttributes -- not worth stopping for


def lag_breakdown(body, received_at, started_at):
    """Split a frame's lag into hops, from the timestamps the Pi and the Lambda put in the message.
    pi_to_s3 uses the Pi's clock against AWS's, so a Pi clock error shows up there."""
    uploaded_at, queued_at = body.get("uploaded_at"), body.get("queued_at")
    if uploaded_at is None or queued_at is None:
        return {}  # message from an older Lambda version
    hops = {}
    written_at = body.get("written_at")  # direct-mode Pi: when the recorder wrote the preview file
    if written_at is not None:
        hops["capture_to_write_s"] = round(written_at - body["frame_time"], 3)  # encode + disk backlog
    return {
        **hops,
        "pi_to_s3_s": round(uploaded_at - (written_at if written_at is not None else body["frame_time"]), 3),
        "s3_to_queue_s": round(queued_at - uploaded_at, 3),        # S3 event -> Lambda -> SQS
        "queue_s": round(received_at - queued_at, 3),              # sitting in SQS
        "batch_wait_s": round(started_at - received_at, 3),        # behind earlier frames of the same batch
    }


def newer_frame_waiting(sqs, queue_url, position, messages):
    """True if a newer frame than messages[position] is already available: a later message in
    this batch (FIFO, so later = newer), or anything still in the queue."""
    if position < len(messages) - 1:
        return True
    return bool(queue_depth(sqs, queue_url))


def run_live(s3, sqs, queue_url, model, device, work_dir, show=SHOW, live=None, max_receives=None, sign_reader=None,
             cw_metrics=None, idle=None):
    """Process frames from the SQS queue as they arrive, forever (or max_receives receive calls).
    Returns the last session's metrics doc."""
    current = None
    receives = 0
    processed = 0
    backlog = None
    stop = False
    print(f"waiting for frames on {queue_url}")

    try:
        while not stop and (max_receives is None or receives < max_receives):
            receives += 1
            resp = sqs.receive_message(QueueUrl=queue_url, MaxNumberOfMessages=RECEIVE_BATCH,
                                       WaitTimeSeconds=1 if show else 20,  # long poll: fewer empty (billed) receives
                                       VisibilityTimeout=VISIBILITY_S)
            messages = resp.get("Messages", [])
            received_at = time.time()

            if not messages:
                if current and time.time() - current.last_frame_at > SESSION_IDLE_S:
                    print(f"session {current.session}: no frames for {SESSION_IDLE_S}s, finalizing")
                    current.finalize()
                    current = None
                    if live:
                        live.finish()
                if show and cv.waitKey(1) & 0xFF == ord('q'):
                    stop = True
                continue

            for position, msg in enumerate(messages):
                body = json.loads(msg["Body"])
                bucket, key, session = body["bucket"], body["key"], body["session_start"]

                if current is None or current.session != session:
                    if current:
                        current.finalize()
                        current = None
                    try:
                        current = SessionOutput(s3, bucket, session, f"s3://{bucket}/live/{session}/",
                                                device, work_dir, resume=True)
                    except Exception as e:
                        # couldn't check for an existing metrics.json, so can't safely resume yet;
                        # leave the message queued and retry after VISIBILITY_S
                        print(f"session {session}: could not start, will retry: {e}")
                        continue

                if key in current.seen:
                    print(f"already processed {key}, skipping re-delivery")
                    sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=msg["ReceiptHandle"])
                    continue

                age = time.time() - body["frame_time"]
                if MAX_LAG_S and age > MAX_LAG_S and newer_frame_waiting(sqs, queue_url, position, messages):
                    current.skipped += 1
                    if cw_metrics:
                        cw_metrics.add("FramesSkipped", 1)
                    print(f"{session}: skipping {PurePosixPath(key).name}, {age:.1f}s old and newer frames are waiting")
                    sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=msg["ReceiptHandle"])
                    continue

                t0 = time.perf_counter()
                try:
                    frame = read_frame(s3, bucket, key)
                except Exception as e:
                    if not is_missing(e):
                        # transient (network/throttling): leave it, SQS re-delivers after VISIBILITY_S
                        print(f"could not read {key}, will retry: {e}")
                        continue
                    frame = None
                if frame is None:
                    print(f"{key} missing or undecodable, dropping")
                    sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=msg["ReceiptHandle"])
                    continue

                t1 = time.perf_counter()
                metrics, overlay, warped = lld.process_frame(frame, model, device, sign_reader)
                t2 = time.perf_counter()
                lag = time.time() - body["frame_time"]  # capture on the Pi -> processed here
                current.add(key, metrics, overlay, warped, frame_time=body["frame_time"],
                            frame_seq=body.get("frame_seq"), lag_s=round(lag, 3))
                sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=msg["ReceiptHandle"])
                t3 = time.perf_counter()

                # where each frame's time goes: S3 download, model, sign reading, the rest of process_frame,
                # outputs + delete
                infer = metrics.get("infer_s", 0.0)
                signs_s = metrics.get("signs_s", 0.0)
                timing = {"download_s": round(t1 - t0, 3), "infer_s": round(infer, 3), "signs_s": round(signs_s, 3),
                          "post_s": round(t2 - t1 - infer - signs_s, 3), "output_s": round(t3 - t2, 3),
                          "total_s": round(t3 - t0, 3)}
                timing.update(lag_breakdown(body, received_at, started_at=time.time() - (t3 - t0)))

                processed += 1
                if idle:
                    idle.touch()
                if processed % BACKLOG_EVERY == 1:
                    backlog = queue_depth(sqs, queue_url)
                if cw_metrics:
                    cw_metrics.record_frame(timing, round(lag, 3), metrics,
                                            backlog=backlog if processed % BACKLOG_EVERY == 1 else None)

                print(f"{session} #{len(current.frames) - 1} | lag {lag:.1f}s | backlog {backlog} | "
                      f"dl {timing['download_s']:.2f} infer {timing['infer_s']:.2f} signs {timing['signs_s']:.2f} "
                      f"post {timing['post_s']:.2f} "
                      f"out {timing['output_s']:.2f} = {timing['total_s']:.2f}s | {lld.format_metrics(metrics)}")
                if live:
                    live.update(overlay, warped, metrics, session=session, index=len(current.frames) - 1,
                                total=None, frame=PurePosixPath(key).name, lag_s=round(lag, 2),
                                timing=timing, backlog=backlog, skipped=current.skipped)
                if show and show_frame(overlay, warped):
                    stop = True
                    break
    finally:
        # Ctrl+C or crash: upload what this session has so far; unprocessed messages stay queued
        if current:
            current.finalize()
        if show:
            cv.destroyAllWindows()
        if live:
            live.finish()
    return current.doc() if current else None


def main(argv=None):
    parser = argparse.ArgumentParser(description="lane pipeline on frames in S3")
    sub = parser.add_subparsers(dest="mode")
    sub.add_parser("live", help="process frames from the SQS queue as they arrive (default)")
    replay = sub.add_parser("replay", help="process a stored session")
    replay.add_argument("prefix", nargs="?", default=FRAMES_PREFIX,
                        help="e.g. live/20261020T140512Z/ or sessions/<id>/frames/")
    args = parser.parse_args(argv)
    mode = args.mode or "live"
    signal.signal(signal.SIGTERM, _sigterm)  # systemctl stop / shutdown: finalize like Ctrl+C

    if OUTPUT_PREFIX.rstrip("/") in ("sessions", "live"):
        raise SystemExit("OUTPUT_PREFIX must be separate from sessions/ and live/")
    if mode == "live" and not QUEUE_URL:
        raise SystemExit("set LLD_QUEUE_URL to the SQS queue URL, e.g. "
                         "https://sqs.us-east-1.amazonaws.com/<account-id>/lld-frames.fifo")
    if mode == "replay" and "<" in args.prefix:
        raise SystemExit("give a prefix: python lld_aws.py replay live/<session_start>/")

    s3 = boto3.client("s3", region_name=REGION)
    model, device = lld.load_model(WEIGHTS_PATH)
    sign_reader = signs.SignReader() if READ_SIGNS else None
    if isinstance(model, cv.dnn.Net):
        threads = f"OpenCV {cv.__version__}, cv threads {cv.getNumThreads()}"
    else:
        import torch
        threads = f"torch threads {torch.get_num_threads()}"
    print(f"model {WEIGHTS_PATH} on {device}, {threads}, cpus {os.cpu_count()}, display: {SHOW}")

    live = None
    if LIVE_PORT:
        try:
            live = LiveView(port=LIVE_PORT, departure_threshold=lld.DEPARTURE_THRESHOLD,
                            token_secret=LIVE_TOKEN_SECRET).start()
            if LIVE_TOKEN_SECRET:
                print("live view: signed-token access check on")
        except OSError as e:
            print(f"live view disabled, could not bind port {LIVE_PORT}: {e}")

    try:
        with tempfile.TemporaryDirectory(prefix="lld_") as work_dir:
            if mode == "live":
                sqs = boto3.client("sqs", region_name=REGION)
                cw_metrics = None
                if PUBLISH_METRICS:
                    cw_metrics = MetricsPublisher(boto3.client("cloudwatch", region_name=REGION)).start()
                    print(f"CloudWatch metrics: namespace {METRICS_NAMESPACE}, every {METRICS_FLUSH_S}s")
                idle = None
                if IDLE_STOP_MIN:
                    idle = IdleStop(IDLE_STOP_MIN * 60, live).start()
                    print(f"idle auto-stop: shutting down after {IDLE_STOP_MIN:g} min without frames or viewers")
                try:
                    run_live(s3, sqs, QUEUE_URL, model, device, work_dir, live=live, sign_reader=sign_reader,
                             cw_metrics=cw_metrics, idle=idle)
                finally:
                    if cw_metrics:
                        cw_metrics.close()
            else:
                run_replay(s3, BUCKET, args.prefix, model, device, work_dir, live=live, sign_reader=sign_reader)

        if live and LIVE_HOLD:
            print("run finished, live view still up -- Ctrl+C to exit")
            threading.Event().wait()
    except KeyboardInterrupt:
        print("stopped")
    finally:
        if live:
            live.stop()


if __name__ == "__main__":
    main()
