#!/usr/bin/env python3
"""Reproducible evaluation of the sign reader (signs.py): real stop signs, false alarms, synthetic
speed limits.

    python src/scripts/eval_signs.py --sessions ../initiative-vision/sessions/* test_sessions/2026*
    python src/scripts/eval_signs.py --sessions ... --size 0      # full 1280 frames instead of 640

Frames go through what the live pipeline does: resize the raw frame to --size (640 = the Pi's
uploads), lld.undistort_frame, then SignReader.detect. Three parts:

1. Real stop signs: every frame of --stop-session with seq in --stop-range (an approach to a stop
   sign); reports in how many frames STOP was found and how wide the sign was.
2. False alarms: every --fp-every-th frame of all --sessions, except the stop approach; any sign
   reported there is a false alarm (the local footage has no other stop or speed-limit signs).
3. Synthetic speed limits: the public-domain MUTCD R2-1 sign (data/eval/r2-1.png, FHWA via
   Wikimedia Commons) with its number re-drawn for each value 15..85 (the template's own "50" is
   kept as is), pasted into random real frames: random width, slight rotation, blur, the camera's
   purple cast and a JPEG q70 round trip. Each paste counts as read correctly, misread (a different
   value at that spot -- the costly error) or missed, bucketed by sign width in processed pixels.

Writes data/eval/signs_results.json. Template source and license: data/eval/README.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2 as cv
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "scripts"))
import lld  # noqa: E402
import signs  # noqa: E402
from virtual_pi import load_frames  # noqa: E402

TEMPLATE = ROOT / "data" / "eval" / "r2-1.png"
STOP_SESSION = "20260927T184255Z-fddd"
STOP_RANGE = (4000, 4120)
NUMBER_BOX = (95, 615, 865, 1065)   # x1, y1, x2, y2 of the "50" in the 960 x 1200 template
WIDTH_BUCKETS = [(30, 40), (40, 50), (50, 70), (70, 85), (85, 100), (100, 140)]
CAMERA_CAST = np.array([1.08, 0.92, 1.06])  # BGR gains: this camera's purple tint


def prepare(raw, size):
    """Raw frame -> what SignReader sees in the live pipeline."""
    if size:
        raw = cv.resize(raw, (size, size), interpolation=cv.INTER_AREA)
    return lld.undistort_frame(raw)


def load_template():
    img = cv.imread(str(TEMPLATE), cv.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"missing {TEMPLATE} (see the module docstring)")
    if img.ndim == 3 and img.shape[2] in (2, 4):  # gray+alpha / BGRA: composite on white
        alpha = img[:, :, -1:].astype(np.float32) / 255
        color = img[:, :, :1] if img.shape[2] == 2 else img[:, :, :3]
        img = (color * alpha + 255 * (1 - alpha)).astype(np.uint8)
    if img.ndim == 2 or img.shape[2] == 1:
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    return img


def sign_with_value(template, value):
    """The R2-1 template showing `value`. 50 is the template's own artwork; other values are
    drawn in OpenCV's bold Hershey font, stretched to the original number's box."""
    if value == 50:
        return template.copy()
    img = template.copy()
    x1, y1, x2, y2 = NUMBER_BOX
    img[y1:y2, x1:x2] = 255
    # draw large on a scratch canvas, crop to the ink, then fit it to the original digits' box:
    # highway digits are tall and narrow, Hershey's are wide, so the fit stretches them upright
    text, font, scale, thick = str(value), cv.FONT_HERSHEY_DUPLEX, 8.0, 24
    (tw, th), base = cv.getTextSize(text, font, scale, thick)
    canvas = np.full((th + base + 2 * thick, tw + 2 * thick), 255, np.uint8)
    cv.putText(canvas, text, (thick, th + thick), font, scale, 0, thick, cv.LINE_AA)
    ys, xs = np.where(canvas < 128)
    ink = canvas[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    bw, bh = int((x2 - x1) * 0.98), int((y2 - y1) * 0.92)
    ink = cv.resize(ink, (bw, bh), interpolation=cv.INTER_AREA)
    ox, oy = x1 + ((x2 - x1) - bw) // 2, y1 + ((y2 - y1) - bh) // 2
    img[oy:oy + bh, ox:ox + bw] = cv.cvtColor(ink, cv.COLOR_GRAY2BGR)
    return img


def paste(frame, sign, width, rng):
    """Paste a sign `width` px wide at a random spot in the top 2/3 of frame (in place);
    returns its box. Slight rotation, blur scaled to size, camera cast."""
    h = int(round(width * sign.shape[0] / sign.shape[1]))
    small = cv.resize(sign, (width, h), interpolation=cv.INTER_AREA)
    angle = rng.uniform(-3, 3)
    m = cv.getRotationMatrix2D((width / 2, h / 2), angle, 1.0)
    small = cv.warpAffine(small, m, (width, h), borderMode=cv.BORDER_REPLICATE)
    small = cv.GaussianBlur(small, (0, 0), rng.uniform(0.4, 1.2))
    small = np.clip(small * CAMERA_CAST * rng.uniform(0.75, 1.0), 0, 255).astype(np.uint8)
    fh, fw = frame.shape[:2]
    x = int(rng.integers(0, fw - width))
    y = int(rng.integers(int(fh * 0.05), int(fh * signs.SEARCH_TOP) - h))
    frame[y:y + h, x:x + width] = small
    return [x, y, x + width, y + h]


def overlaps(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy > 0.3 * (a[2] - a[0]) * (a[3] - a[1])


def eval_stop(reader, session_dir, seq_range, size):
    frames = [f for f in load_frames(session_dir) if seq_range[0] <= f[0] <= seq_range[1]]
    widths = []
    for seq, path, _ in frames:
        found = [s for s in reader.detect(prepare(cv.imread(str(path)), size)) if s["type"] == "stop"]
        if found:
            widths.append(found[0]["box"][2] - found[0]["box"][0])
    return {"session": Path(session_dir).name, "seq_range": list(seq_range), "frames": len(frames),
            "frames_with_stop": len(widths),
            "sign_width_px": {"min": min(widths), "median": float(np.median(widths)), "max": max(widths)} if widths else None}


def eval_false_alarms(reader, session_dirs, every, size, exclude):
    checked, alarms = 0, []
    for d in session_dirs:
        for seq, path, _ in load_frames(d)[::every]:
            if Path(d).name == exclude[0] and exclude[1][0] - 100 <= seq <= exclude[1][1] + 100:
                continue
            img = cv.imread(str(path))
            if img is None:
                continue
            checked += 1
            for s in reader.detect(prepare(img, size)):
                alarms.append({"session": Path(d).name, "seq": seq, "type": s["type"], "value": s["value"], "text": s["text"]})
    return {"frames_checked": checked, "false_alarms": len(alarms), "examples": alarms[:20]}


def eval_speed(reader, template, session_dirs, n, seeds, size):
    backgrounds = [p for d in session_dirs for _, p, _ in load_frames(d)[::50]]
    values = [int(v) for v in signs.SPEED_VALUES]
    signs_by_value = {v: sign_with_value(template, v) for v in values}
    buckets = {f"{a}-{b}": {"n": 0, "correct": 0, "misread": 0, "missed": 0} for a, b in WIDTH_BUCKETS}
    misreads = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        for _ in range(n // seeds):
            img = cv.imread(str(backgrounds[int(rng.integers(len(backgrounds)))]))
            if img is None:
                continue
            frame = prepare(img, size)
            value = values[int(rng.integers(len(values)))]
            lo, hi = WIDTH_BUCKETS[int(rng.integers(len(WIDTH_BUCKETS)))]
            width = int(rng.integers(lo, hi))
            box = paste(frame, signs_by_value[value], width, rng)
            ok, buf = cv.imencode(".jpg", frame, [cv.IMWRITE_JPEG_QUALITY, 70])
            frame = cv.imdecode(buf, cv.IMREAD_COLOR)
            here = [s for s in reader.detect(frame) if s["type"] == "speed_limit" and overlaps(box, s["box"])]
            b = buckets[f"{lo}-{hi}"]
            b["n"] += 1
            if any(s["value"] == value for s in here):
                b["correct"] += 1
            elif here:
                b["misread"] += 1
                misreads.append({"value": value, "read": here[0]["value"], "width": width, "seed": seed})
            else:
                b["missed"] += 1
    for b in buckets.values():
        b["correct_rate"] = round(b["correct"] / b["n"], 3) if b["n"] else None
    return {"signs": sum(b["n"] for b in buckets.values()), "seeds": seeds, "by_width_px": buckets,
            "misreads": misreads}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sessions", nargs="+", required=True, help="session dirs (frames/ + frames.csv)")
    parser.add_argument("--size", type=int, default=640, help="resize raw frames first (0 = as recorded)")
    parser.add_argument("--stop-session", default=None, help=f"default: the one named {STOP_SESSION} in --sessions")
    parser.add_argument("--fp-every", type=int, default=15)
    parser.add_argument("--synthetic", type=int, default=2000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--out", default=str(ROOT / "data" / "eval" / "signs_results.json"))
    args = parser.parse_args(argv)

    sessions = [d for d in args.sessions if (Path(d) / "frames.csv").exists()]
    stop_session = args.stop_session or next((d for d in sessions if Path(d).name == STOP_SESSION), None)
    reader = signs.SignReader(str(ROOT / signs.OCR_WEIGHTS))
    results = {"opencv": cv.__version__, "size": args.size or "as recorded", "date": time.strftime("%Y-%m-%d")}

    t = time.time()
    if stop_session:
        results["stop"] = eval_stop(reader, stop_session, STOP_RANGE, args.size)
        s = results["stop"]
        print(f"stop: STOP in {s['frames_with_stop']}/{s['frames']} frames of the approach; widths {s['sign_width_px']}")
    results["false_alarms"] = eval_false_alarms(reader, sessions, args.fp_every, args.size, (STOP_SESSION, STOP_RANGE))
    fa = results["false_alarms"]
    print(f"false alarms: {fa['false_alarms']} in {fa['frames_checked']} frames")
    results["speed_limit_synthetic"] = eval_speed(reader, load_template(), sessions, args.synthetic, args.seeds, args.size)
    sp = results["speed_limit_synthetic"]
    print(f"speed limits, {sp['signs']} synthetic signs:")
    for k, b in sp["by_width_px"].items():
        print(f"  {k:>8} px: {b['correct']}/{b['n']} correct, {b['misread']} misread, {b['missed']} missed")
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"{time.time() - t:.0f} s; written to {args.out}")


if __name__ == "__main__":
    main()
