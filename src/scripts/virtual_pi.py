#!/usr/bin/env python3
"""Virtual Pi: replays a recorded drive in real time and uploads live frames like the real Pi.

    python src/scripts/virtual_pi.py <session_dir> [--start 120] [--speed 1] [--no-upload] [--direct] [--size 640]
    python src/scripts/virtual_pi.py ../initiative-vision/sessions/20260927T184255Z-fddd

The session is a Pi recording (frames/ + frames.csv with seq,filename,timestamp_ns). Frames are
shown in a window at their recorded pace, as the camera would produce them. Separately, an
uploader thread does what upload_live_frame.py does: every POLL_S it checks for a new frame and,
at most once per MIN_UPLOAD_INTERVAL_S, PUTs the current one to s3://<bucket>/live/latest_frame.jpg
(JPEG q70, 1280x1280) with the metadata the Lambda reads:

    session-start  this run's start, UTC %Y%m%dT%H%M%SZ (a new session each run)
    frame-time     epoch seconds when the frame was "captured" = shown here, so lag is real
    frame-seq      the frame's seq in the recording

From there it goes through the real S3 -> Lambda -> SQS -> EC2 path. Put this window next to the
live page and the gap between them is the end-to-end lag. The window shows the frame undistorted
and upright (lld.undistort_frame) with an overlay (seq, clock, last upload); what's uploaded is
the raw frame, as the Pi sends it. --size 640 downscales it first (INTER_AREA) to cut upload
time; lld.py scales the camera calibration to whatever size arrives.

--direct skips the S3 event + Lambda hop (~1 s of S3 event delivery): the frame is PUT straight
to its archive key live/<session-start>/<frame time>.jpg and the SQS message the Lambda would
have sent is sent from here, in the same format, so lld_aws.py needs no change. Needs the queue
URL (--queue-url or LLD_QUEUE_URL) and s3:PutObject on live/* + sqs:SendMessage on the queue.

Keys: q / Esc quit, space pause (uploads pause too).
Needs boto3 and AWS credentials allowed to s3:PutObject on live/latest_frame.jpg.
"""

from __future__ import annotations

import argparse
import csv
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import os
import sys

import cv2 as cv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "lambda"))
import lld  # noqa: E402
from on_live_frame import archive_key, queue_message  # noqa: E402  (same key + message as the Lambda)

BUCKET = "initiative-vision-1"
LIVE_KEY = "live/latest_frame.jpg"
POLL_S = 0.1                 # same as the Pi uploader
MIN_UPLOAD_INTERVAL_S = 1.0  # at most 1 upload/s, same as the Pi
JPEG_QUALITY = 70
DISPLAY_SIZE = 640


def load_frames(session_dir):
    """[(seq, path, t_s)] from frames.csv, t_s = seconds since the first frame."""
    session_dir = Path(session_dir)
    with open(session_dir / "frames.csv", newline="") as f:
        rows = [(int(r["seq"]), session_dir / r["filename"], int(r["timestamp_ns"])) for r in csv.DictReader(f)]
    t0 = rows[0][2]
    return [(seq, path, (ts - t0) / 1e9) for seq, path, ts in rows]


def index_at(frames, t, start=0):
    """Index of the last frame recorded at or before t (seconds into the drive), searching from start."""
    i = start
    while i + 1 < len(frames) and frames[i + 1][2] <= t:
        i += 1
    return i


def session_start_name(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def upload_metadata(session_start, frame_time, frame_seq):
    return {"session-start": session_start, "frame-time": f"{frame_time:.3f}", "frame-seq": str(frame_seq)}


class Uploader(threading.Thread):
    """Uploads the latest published frame at most once per MIN_UPLOAD_INTERVAL_S."""

    def __init__(self, session_start, bucket, enabled, queue_url=None):
        super().__init__(daemon=True)
        self.session_start, self.bucket, self.enabled = session_start, bucket, enabled
        self.queue_url = queue_url  # set = direct mode
        self.s3 = self.sqs = None
        if enabled:
            import boto3
            self.s3 = boto3.client("s3")
            if queue_url:
                self.sqs = boto3.client("sqs")
        self.lock = threading.Lock()
        self.latest = None   # (seq, frame, frame_time)
        self.status = "no upload yet"
        self.stop = threading.Event()
        self.paused = False

    def publish(self, seq, frame, frame_time):
        with self.lock:
            self.latest = (seq, frame, frame_time)

    def run(self):
        last_seq, last_start = None, 0.0
        while not self.stop.is_set():
            time.sleep(POLL_S)
            with self.lock:
                latest = self.latest
            if self.paused or latest is None or latest[0] == last_seq:
                continue
            if time.time() - last_start < MIN_UPLOAD_INTERVAL_S:
                continue
            seq, frame, frame_time = latest
            last_seq, last_start = seq, time.time()
            ok, jpg = cv.imencode(".jpg", frame, [cv.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not ok:
                continue
            t = time.time()
            try:
                self.status = self._send(jpg.tobytes(), seq, frame_time, t)
                print(f"{self.status}  age when done {time.time() - frame_time:.2f}s")
            except Exception as e:  # keep going through network drops, like the Pi
                self.status = f"upload failed #{seq}: {type(e).__name__}"
                print(f"{self.status}: {e}")


    def _send(self, jpg, seq, frame_time, t):
        """Upload one frame (and in direct mode queue it); returns a status line."""
        kb = len(jpg) // 1024
        if not self.enabled:
            return f"#{seq} {kb} KB [dry run]"
        meta = upload_metadata(self.session_start, frame_time, seq)
        if not self.queue_url:
            self.s3.put_object(Bucket=self.bucket, Key=LIVE_KEY, Body=jpg, ContentType="image/jpeg", Metadata=meta)
            return f"uploaded #{seq} ({kb} KB, put {time.time() - t:.2f}s)"
        key = archive_key(self.session_start, frame_time)
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=jpg, ContentType="image/jpeg", Metadata=meta)
        uploaded_at = time.time()
        body = {"bucket": self.bucket, "key": key, "session_start": self.session_start,
                "frame_time": frame_time, "frame_seq": seq,
                "uploaded_at": uploaded_at, "queued_at": time.time()}
        self.sqs.send_message(**queue_message(self.queue_url, body))
        return (f"direct #{seq} ({kb} KB, put {uploaded_at - t:.2f}s, "
                f"send {time.time() - uploaded_at:.2f}s)")


def display_frame(frame):
    """Undistorted, upright, DISPLAY_SIZE wide -- what the live page's overlay starts from."""
    upright = lld.undistort_frame(frame)
    h, w = upright.shape[:2]
    return cv.resize(upright, (DISPLAY_SIZE, round(h * DISPLAY_SIZE / w)), interpolation=cv.INTER_AREA)


def draw_overlay(img, seq, drive_t, status, paused):
    lines = [f"#{seq}  drive {drive_t:7.1f}s  {datetime.now().strftime('%H:%M:%S.%f')[:-4]}", status]
    if paused:
        lines.append("PAUSED")
    for i, text in enumerate(lines):
        y = 26 + 24 * i
        cv.putText(img, text, (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(img, text, (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session_dir")
    parser.add_argument("--start", type=float, default=0.0, help="seconds into the drive to start from")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed (uploads stay <= 1/s)")
    parser.add_argument("--bucket", default=BUCKET)
    parser.add_argument("--no-upload", action="store_true", help="show and pace only, nothing sent to S3")
    parser.add_argument("--direct", action="store_true",
                        help="PUT to the archive key and send the SQS message from here (no S3 event / Lambda)")
    parser.add_argument("--queue-url", default=os.environ.get("LLD_QUEUE_URL"), help="for --direct")
    parser.add_argument("--size", type=int, default=None,
                        help="upload frames resized to SIZE x SIZE (default: as recorded, 1280)")
    parser.add_argument("--duration", type=float, default=None, help="stop after this many seconds")
    parser.add_argument("--headless", action="store_true", help="no window (for testing)")
    args = parser.parse_args(argv)

    if args.direct and not args.no_upload and not args.queue_url:
        parser.error("--direct needs --queue-url or LLD_QUEUE_URL")
    frames = load_frames(args.session_dir)
    run_start = time.time()
    session_start = session_start_name(run_start)
    uploader = Uploader(session_start, args.bucket, enabled=not args.no_upload,
                        queue_url=args.queue_url if args.direct else None)
    uploader.start()
    mode = ("dry run" if args.no_upload else
            f"direct: s3://{args.bucket}/live/{session_start}/ + {args.queue_url}" if args.direct else
            f"uploading to s3://{args.bucket}/{LIVE_KEY}")
    print(f"{len(frames)} frames, {frames[-1][2]:.0f}s of driving; session {session_start}; {mode}; "
          f"upload size {f'{args.size}x{args.size}' if args.size else 'as recorded'}")

    i = index_at(frames, args.start)
    drive_t0, wall_t0 = frames[i][2], time.monotonic()
    shown, pause_started = None, None
    try:
        while True:
            now = time.monotonic()
            if args.duration and now - wall_t0 >= args.duration:
                break
            if pause_started is None:
                i = index_at(frames, drive_t0 + (now - wall_t0) * args.speed, i)  # skip frames if behind
            seq, path, drive_t = frames[i]
            if seq != shown:
                frame = cv.imread(str(path))
                if frame is not None:
                    captured = time.time()
                    upload = frame if not args.size else cv.resize(frame, (args.size, args.size), interpolation=cv.INTER_AREA)
                    uploader.publish(seq, upload, captured)
                    shown = seq
                    if not args.headless:
                        view = display_frame(frame)
                        draw_overlay(view, seq, drive_t, uploader.status, pause_started is not None)
                        cv.imshow("virtual pi", view)
            if i == len(frames) - 1:
                print("end of recording")
                break
            key = cv.waitKey(5) & 0xFF if not args.headless else (time.sleep(0.005) or 255)
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                if pause_started is None:
                    pause_started = time.monotonic()
                else:  # shift the clock so playback resumes where it paused
                    wall_t0 += time.monotonic() - pause_started
                    pause_started = None
                uploader.paused = pause_started is not None
    finally:
        uploader.stop.set()
        uploader.join(timeout=5)
        cv.destroyAllWindows()
    print(f"stopped after {time.time() - run_start:.0f}s")


if __name__ == "__main__":
    main()
