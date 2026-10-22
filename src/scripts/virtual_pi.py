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

Next to the frame, a CAN panel shows what the car was doing at that moment: speed, RPM,
steering wheel angle, accelerator and brake pedal, and gear (PRNDL). It's read from the session's
decoded.csv (same monotonic clock as the frames). Make it once with the Pi repo's decoder:

    PYTHONPATH=src python3 -m ivision.can.decode <session_dir> --dbc dbc/gm_global_a_powertrain.dbc ...

(the standard command with its --skip blacklist is in the initiative-vision README). Without
decoded.csv the panel says so and everything else works.

Keys: q / Esc quit, space pause (uploads pause too), a / d or left / right arrow seek 10 s back /
forward. Click the timeline at the bottom to jump anywhere in the drive.
Needs boto3 and AWS credentials allowed to s3:PutObject on live/latest_frame.jpg.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import math
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import os
import sys

import cv2 as cv
import numpy as np

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
SEEK_S = 10
PANEL_W = 300
TIMELINE_H = 34

# CAN signals shown in the panel: name -> (decoded.csv column(s), scale). Columns come from the GM
# global-A DBC; the first column that exists is used. Pedal scales follow openpilot's GM carstate
# (AcceleratorPedal2 / 254, BrakePedalPos / 0xd0) -- shown as %, not independently verified.
# SteeringWheelAngle: positive = turning left (checked against the camera: the scene slides right).
# TurnSignals: 1 = left, 2 = right (openpilot's convention; in this drive 2 was followed by right
# turns). It's the stalk state, steady for the whole signal, so the panel does the flashing.
CAN_SIGNALS = {
    "speed_mph": (("VehicleSpeed",), 1.0),
    "rpm": (("EngineRPM",), 1.0),
    "steering_deg": (("SteeringWheelAngle",), 1.0),
    "accel_pct": (("AcceleratorPedal2", "AcceleratorPedal"), 100 / 254),
    "brake_pct": (("BrakePedalPos",), 100 / 0xD0),
    "gear": (("PRNDL2", "PRNDL"), None),  # text
    "blinker": (("TurnSignals",), 1.0),
}
BLINK_HZ = 1.5
GEARS = "PRNDL"


def load_frames(session_dir):
    """[(seq, path, t_s)] from frames.csv, t_s = seconds since the first frame."""
    session_dir = Path(session_dir)
    with open(session_dir / "frames.csv", newline="") as f:
        rows = [(int(r["seq"]), session_dir / r["filename"], int(r["timestamp_ns"])) for r in csv.DictReader(f)]
    t0 = rows[0][2]
    return [(seq, path, (ts - t0) / 1e9) for seq, path, ts in rows]


def first_timestamp_ns(session_dir):
    """timestamp_ns of the session's first frame: t = 0 for load_frames and CanTrack."""
    with open(Path(session_dir) / "frames.csv", newline="") as f:
        return int(next(csv.DictReader(f))["timestamp_ns"])


class CanTrack:
    """The CAN_SIGNALS values from a session's decoded.csv, looked up by drive time.

    decoded.csv rows are full snapshots (each column forward-filled with its last decoded value),
    so the state at time t is simply the last row at or before t."""

    def __init__(self, path, t0_ns):
        self.available = Path(path).exists()
        self.t, self.values = np.zeros(0), {}
        if not self.available:
            return
        with open(path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            cols = {}
            for name, (candidates, _) in CAN_SIGNALS.items():
                col = next((c for c in candidates if c in header), None)
                if col is not None:
                    cols[name] = header.index(col)
            it = header.index("timestamp_ns")
            t, raw = [], {name: [] for name in cols}
            for row in reader:
                t.append(int(row[it]))
                for name, i in cols.items():
                    raw[name].append(row[i])
        self.t = (np.array(t, dtype=np.int64) - t0_ns) / 1e9
        for name, vals in raw.items():
            scale = CAN_SIGNALS[name][1]
            if scale is None:
                self.values[name] = vals
            else:
                self.values[name] = np.array([float(v) * scale if v else np.nan for v in vals])

    def at(self, t):
        """{signal: value or None} as of t seconds into the drive (None before it was ever seen)."""
        i = int(np.searchsorted(self.t, t, side="right")) - 1
        out = {name: None for name in CAN_SIGNALS}
        if i < 0:
            return out
        for name, vals in self.values.items():
            v = vals[i]
            if isinstance(v, str):
                out[name] = v or None
            elif not np.isnan(v):
                out[name] = float(v)
        return out


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


def _text(img, text, org, scale=0.5, color=(220, 220, 220), thick=1):
    cv.putText(img, text, org, cv.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv.LINE_AA)


def _bar(img, x, y, w, h, frac, color, vertical=False):
    """Outlined bar filled to frac (0..1): left to right, or bottom to top if vertical."""
    cv.rectangle(img, (x, y), (x + w, y + h), (90, 90, 90), 1)
    frac = min(max(frac or 0.0, 0.0), 1.0)
    if vertical:
        cv.rectangle(img, (x + 1, y + h - int((h - 2) * frac) - 1), (x + w - 1, y + h - 1), color, -1)
    else:
        cv.rectangle(img, (x + 1, y + 1), (x + 1 + int((w - 2) * frac), y + h - 1), color, -1)


def draw_blinkers(p, blinker, t):
    """Left / right arrows top right; the active one flashes at BLINK_HZ (t = drive time, so a
    paused replay holds still)."""
    lit = blinker in (1.0, 2.0) and int(t * BLINK_HZ * 2) % 2 == 0
    green, dim = (60, 220, 60), (70, 70, 70)
    for side, pts in ((1.0, [(200, 22), (224, 8), (224, 36)]), (2.0, [(284, 22), (260, 8), (260, 36)])):
        on = lit and blinker == side
        cv.fillPoly(p, [np.array(pts)], green if on else (28, 28, 28), cv.LINE_AA)
        cv.polylines(p, [np.array(pts)], True, green if on else dim, 1, cv.LINE_AA)


def draw_can_panel(height, v, available=True, t=0.0):
    """The CAN dashboard as a PANEL_W x height image, from CanTrack.at() values at drive time t."""
    p = np.full((height, PANEL_W, 3), 28, np.uint8)
    grey, white, accent = (150, 150, 150), (240, 240, 240), (60, 170, 255)
    _text(p, "CAN", (16, 28), 0.6, grey)
    if not available:
        _text(p, "no decoded.csv for this session", (16, 60), 0.45, grey)
        _text(p, "run ivision.can.decode (see --help)", (16, 82), 0.45, grey)
        return p

    draw_blinkers(p, v.get("blinker"), t)

    # speed
    speed = v["speed_mph"]
    _text(p, "--" if speed is None else f"{speed:.0f}", (16, 100), 2.2, white, 3)
    _text(p, "mph", (150, 72), 0.6, grey)
    if speed is not None:
        _text(p, f"{speed * 1.609344:.0f} km/h", (150, 98), 0.5, grey)

    # gear: P R N D L, the current one highlighted
    gear = v["gear"]
    for k, g in enumerate(GEARS):
        x0 = 16 + k * 54
        active = gear is not None and gear[:1] == g and gear != "Shifting"
        cv.rectangle(p, (x0, 122), (x0 + 44, 162), accent if active else (70, 70, 70), -1 if active else 1)
        _text(p, g, (x0 + 13, 152), 0.8, (20, 20, 20) if active else grey, 2)
    if gear and (gear == "Shifting" or len(gear) > 1):
        _text(p, gear, (16, 182), 0.45, accent)

    # RPM
    rpm = v["rpm"]
    _text(p, "RPM", (16, 206), 0.45, grey)
    _text(p, "--" if rpm is None else f"{rpm:.0f}", (70, 206), 0.5, white)
    _bar(p, 16, 214, 268, 14, None if rpm is None else rpm / 6000, (80, 200, 255))

    # steering wheel: positive angle = left = counterclockwise on screen
    angle = v["steering_deg"]
    cx, cy, r = 150, 318, 62
    cv.circle(p, (cx, cy), r, (200, 200, 200), 3, cv.LINE_AA)
    a = math.radians(angle or 0.0)
    for spoke in (0.0, math.radians(120), math.radians(-120)):  # bottom, lower-left, lower-right spokes
        # (0, r) below the centre, rotated counterclockwise on screen by a
        cv.line(p, (cx, cy), (int(cx + r * math.sin(a + spoke)), int(cy + r * math.cos(a + spoke))),
                (160, 160, 160), 2, cv.LINE_AA)
    top = (int(cx - r * math.sin(a)), int(cy - r * math.cos(a)))  # the wheel's 12 o'clock mark
    cv.circle(p, top, 7, accent, -1, cv.LINE_AA)
    _text(p, "STEERING", (16, 252), 0.45, grey)
    if angle is not None:
        side = "L" if angle > 0.5 else "R" if angle < -0.5 else ""
        _text(p, f"{abs(angle):.0f} deg {side}", (16, 400), 0.55, white)

    # pedals
    for k, (name, label, color) in enumerate((("accel_pct", "ACCEL", (90, 200, 90)),
                                              ("brake_pct", "BRAKE", (70, 70, 230)))):
        x0 = 60 + k * 120
        pct = v[name]
        _bar(p, x0, 425, 56, 150, None if pct is None else pct / 100, color, vertical=True)
        _text(p, label, (x0 + 2, 597), 0.45, grey)
        _text(p, "--" if pct is None else f"{pct:.0f}%", (x0 + 6, 620), 0.5, white)
    return p


def draw_timeline(width, drive_t, total_t):
    """Progress bar across the bottom; clicking it seeks (see on_mouse in main)."""
    bar = np.full((TIMELINE_H, width, 3), 18, np.uint8)
    x0, x1 = 90, width - 16
    _text(bar, f"{int(drive_t // 60):02d}:{int(drive_t % 60):02d}", (12, 22), 0.5, (220, 220, 220))
    cv.rectangle(bar, (x0, 13), (x1, 21), (70, 70, 70), -1)
    for m in range(1, int(total_t // 60) + 1):  # a tick per minute
        x = x0 + int((x1 - x0) * m * 60 / total_t)
        cv.line(bar, (x, 9), (x, 25), (110, 110, 110), 1)
    pos = x0 + int((x1 - x0) * min(drive_t / total_t, 1.0)) if total_t else x0
    cv.rectangle(bar, (x0, 13), (pos, 21), (60, 170, 255), -1)
    cv.circle(bar, (pos, 17), 7, (240, 240, 240), -1, cv.LINE_AA)
    return bar


def draw_overlay(img, seq, drive_t, status, paused):
    lines = [f"#{seq}  drive {drive_t:7.1f}s  {datetime.now().strftime('%H:%M:%S.%f')[:-4]}", status]
    if paused:
        lines.append("PAUSED")
    width = max(cv.getTextSize(t, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] for t in lines)
    box = img[4:12 + 22 * len(lines), 4:20 + width]
    box[:] = (box * 0.35).astype(img.dtype)  # darken behind the text so it reads on any frame
    for i, text in enumerate(lines):
        _text(img, text, (12, 24 + 22 * i), 0.5, (255, 255, 255))


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
    times = [f[2] for f in frames]
    total_t = frames[-1][2]
    can = None
    if not args.headless:
        print("loading CAN (decoded.csv) ...")
        can = CanTrack(Path(args.session_dir) / "decoded.csv", first_timestamp_ns(args.session_dir))
        if not can.available:
            print("no decoded.csv: the CAN panel stays empty (see --help to make one)")
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
    seek_to = []  # set by the timeline click handler, read by the loop

    def seek(t):
        nonlocal i, drive_t0, wall_t0, pause_started, shown
        t = min(max(t, 0.0), total_t)
        i = max(0, bisect.bisect_right(times, t) - 1)
        drive_t0, wall_t0, shown = frames[i][2], time.monotonic(), None
        if pause_started is not None:
            pause_started = wall_t0

    window = "virtual pi"
    if not args.headless:
        cv.namedWindow(window, cv.WINDOW_AUTOSIZE)

        def on_mouse(event, x, y, flags, param):
            height = param["height"]
            if event == cv.EVENT_LBUTTONDOWN and y >= height and param["width"]:
                frac = (x - 90) / (param["width"] - 16 - 90)
                seek_to.append(min(max(frac, 0.0), 1.0) * total_t)
        mouse = {"height": 10 ** 9, "width": 0}
        cv.setMouseCallback(window, on_mouse, mouse)

    started = wall_t0
    try:
        while True:
            now = time.monotonic()
            if args.duration and now - started >= args.duration:
                break
            if seek_to:
                seek(seek_to.pop())
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
                        panel = draw_can_panel(view.shape[0], can.at(drive_t), can.available, drive_t)
                        screen = np.hstack([view, panel])
                        screen = np.vstack([screen, draw_timeline(screen.shape[1], drive_t, total_t)])
                        mouse["height"], mouse["width"] = view.shape[0], screen.shape[1]
                        cv.imshow(window, screen)
            if i == len(frames) - 1:
                print("end of recording")
                break
            key = cv.waitKeyEx(5) if not args.headless else (time.sleep(0.005) or -1)
            if key in (ord("q"), 27):
                break
            if key in (ord("a"), 65361, 2424832):    # a / left arrow (GTK/Qt, Windows codes)
                seek(frames[i][2] - SEEK_S)
            elif key in (ord("d"), 65363, 2555904):  # d / right arrow
                seek(frames[i][2] + SEEK_S)
            if key == ord(" "):
                if pause_started is None:
                    pause_started = time.monotonic()
                else:  # shift the clock so playback resumes where it paused
                    wall_t0 += time.monotonic() - pause_started
                    pause_started = None
                uploader.paused = pause_started is not None
                shown = None  # redraw now, so PAUSED shows up
    finally:
        uploader.stop.set()
        uploader.join(timeout=5)
        cv.destroyAllWindows()
    print(f"stopped after {time.time() - run_start:.0f}s")


if __name__ == "__main__":
    main()
