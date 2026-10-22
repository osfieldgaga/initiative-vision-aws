#!/usr/bin/env python3
"""Make a shareable copy of a recorded session: faces and number plates blurred, no location, no VIN.

    python src/scripts/privacy_package.py <session_dir> <out_dir> [--start 60] [--end 600]
    python src/scripts/privacy_package.py test_sessions/20261022T164127Z-084b judge_data
    python src/scripts/privacy_package.py ... --size 640     # ~5x smaller download, what the Pi uploads anyway

Writes <out_dir>/<session>/ with what virtual_pi.py needs, and nothing else:

    frames/       every frame in the window, faces and plate areas blurred (JPEG q90 like the recorder,
                  or --size 640: blurred at full resolution first, then saved at 640 x 640, q85)
    frames.csv    the ledger, trimmed to the window
    decoded.csv   decoded CAN, trimmed, with the GPS columns removed
    session.json  copied as is (camera and system settings, no host or user names)
    PRIVACY.md    what was removed and blurred, with counts

Not copied: can.csv (raw CAN contains the VIN and the OnStar GPS), system.csv.

How frames are blurred (in the upright, rotated frame, then rotated back so the files stay raw):
  - faces: YuNet (OpenCV model zoo, cv.FaceDetectorYN) at full resolution. Tuned for recall: in
    testing it also fires on wheels and lamps, which costs nothing here.
  - plates: no plate detector worked on this footage (the zoo's LPD-YuNet is trained on Chinese
    plates and found none), so the lower half of every vehicle box from YOLOPv2
    (lld.decode_detections) at least MIN_VEHICLE_W px wide is blurred instead: full width, since a
    car seen side-on has its plate at one end of the box. Smaller vehicles are too far away for a
    readable plate. On 50 frames, YOLOPv2 still found 93% of the vehicles in the blurred frames and
    lane_valid was unchanged, so the frames still work for the pipeline.

Also writes <out_dir>/<session>-review/: before/after pairs of frames where something was blurred.
Look through them before publishing -- this is a safety net, not a guarantee. Re-running resumes
(frames already written are skipped). About 0.5 s per frame on a laptop CPU, mostly YOLOPv2.

Weights (data/weights/privacy/, not in git):
  curl -L -o data/weights/privacy/face_detection_yunet_2023mar.onnx \\
    https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx
plus data/weights/yolopv2.onnx (see the README).
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import cv2 as cv
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lld  # noqa: E402

FACE_WEIGHTS = "data/weights/privacy/face_detection_yunet_2023mar.onnx"
FACE_SCORE = 0.7          # YuNet's default is 0.9; lower = more blurred false positives, fewer misses
FACE_PAD = 0.3            # grow face boxes by this fraction on each side
MIN_VEHICLE_W = 60        # px at full resolution (1280); a plate on a smaller car is a few px wide
PLATE_ZONE = (0.0, 0.5, 1.0, 1.0)    # x1, y1, x2, y2 as fractions of the vehicle box: lower half
DROP_COLUMNS = ("GPSLatitude", "GPSLongitude")
JPEG_QUALITY = 90
REVIEW_MAX = 40           # before/after pairs written for review


def blur_region(img, x1, y1, x2, y2):
    """Irreversibly blur img[y1:y2, x1:x2] in place: pixelate, then smooth the blocks."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return False
    roi = img[y1:y2, x1:x2]
    small = cv.resize(roi, (max(1, (x2 - x1) // 8), max(1, (y2 - y1) // 8)), interpolation=cv.INTER_AREA)
    roi[:] = cv.GaussianBlur(cv.resize(small, (x2 - x1, y2 - y1), interpolation=cv.INTER_NEAREST), (0, 0), 3)
    return True


class Anonymizer:
    def __init__(self, face_weights=FACE_WEIGHTS, vehicle_weights=lld.DEFAULT_WEIGHTS):
        self.face = cv.FaceDetectorYN.create(face_weights, "", (320, 320), FACE_SCORE)
        # default engine: the fastest on x86 laptops (lld uses the classic one, fastest on Graviton)
        self.net = cv.dnn.readNetFromONNX(vehicle_weights)

    def regions(self, upright):
        """[(kind, x1, y1, x2, y2)] to blur in an upright BGR frame."""
        h, w = upright.shape[:2]
        out = []
        self.face.setInputSize((w, h))
        _, faces = self.face.detect(upright)
        for x, y, fw, fh in ([] if faces is None else faces[:, :4]):
            px, py = fw * FACE_PAD, fh * FACE_PAD
            out.append(("face", x - px, y - py, x + fw + px, y + fh + py))

        model_in = cv.resize(upright, (lld.MODEL_SIZE, lld.MODEL_SIZE), interpolation=cv.INTER_AREA)
        _, _, det_maps, _ = lld._infer_dnn(self.net, model_in)
        sx, sy = w / lld.MODEL_SIZE, h / lld.MODEL_SIZE
        for obj in lld.decode_detections(det_maps):
            x1, y1, x2, y2 = obj["box"]
            x1, x2, y1, y2 = x1 * sx, x2 * sx, y1 * sy, y2 * sy
            if x2 - x1 < MIN_VEHICLE_W:
                continue
            bw, bh = x2 - x1, y2 - y1
            zx1, zy1, zx2, zy2 = PLATE_ZONE
            out.append(("plate", x1 + bw * zx1, y1 + bh * zy1, x1 + bw * zx2, y1 + bh * zy2))
        return out

    def anonymize(self, raw):
        """Raw (rotated) camera frame -> (blurred raw frame, regions). Detection runs upright."""
        upright = cv.rotate(raw, cv.ROTATE_90_CLOCKWISE)
        regions = [r for r in self.regions(upright) if blur_region(upright, *r[1:])]
        return cv.rotate(upright, cv.ROTATE_90_COUNTERCLOCKWISE), regions


def read_frames(session_dir):
    with open(Path(session_dir) / "frames.csv", newline="") as f:
        return list(csv.DictReader(f))


def window(rows, start_s, end_s):
    """frames.csv rows between start_s and end_s seconds after the first frame."""
    t0 = int(rows[0]["timestamp_ns"])
    keep = [r for r in rows if start_s <= (int(r["timestamp_ns"]) - t0) / 1e9 <= (end_s if end_s is not None else 1e18)]
    if not keep:
        raise SystemExit("no frames in that window")
    return keep


def copy_decoded(src, dst, t_from_ns, t_to_ns):
    """decoded.csv rows in [t_from_ns, t_to_ns] without DROP_COLUMNS. Returns (rows, dropped columns).
    Rows are snapshots (forward-filled), so the first kept row already carries every known value."""
    with open(src, newline="") as fin, open(dst, "w", newline="") as fout:
        reader = csv.reader(fin)
        header = next(reader)
        keep = [i for i, c in enumerate(header) if c not in DROP_COLUMNS]
        dropped = [c for c in header if c in DROP_COLUMNS]
        it = header.index("timestamp_ns")
        writer = csv.writer(fout)
        writer.writerow([header[i] for i in keep])
        n = 0
        for row in reader:
            if t_from_ns <= int(row[it]) <= t_to_ns:
                writer.writerow([row[i] for i in keep])
                n += 1
    return n, dropped


def write_review(review_dir, name, before, after, regions):
    marked = cv.rotate(before, cv.ROTATE_90_CLOCKWISE)
    for kind, x1, y1, x2, y2 in regions:
        cv.rectangle(marked, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255) if kind == "face" else (0, 200, 255), 2)
    pair = np.hstack([marked, cv.rotate(after, cv.ROTATE_90_CLOCKWISE)])
    cv.imwrite(str(review_dir / name), cv.resize(pair, (pair.shape[1] // 2, pair.shape[0] // 2)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session_dir")
    parser.add_argument("out_dir")
    parser.add_argument("--start", type=float, default=0.0, help="seconds after the first frame")
    parser.add_argument("--end", type=float, default=None, help="seconds after the first frame (default: the end)")
    parser.add_argument("--size", type=int, default=0,
                        help="save frames at SIZE x SIZE (detection still runs at full resolution); 0 = as recorded")
    parser.add_argument("--quality", type=int, default=None, help=f"JPEG quality (default {JPEG_QUALITY}, 85 with --size)")
    args = parser.parse_args(argv)

    src = Path(args.session_dir)
    out = Path(args.out_dir) / src.name
    review = Path(args.out_dir) / f"{src.name}-review"
    (out / "frames").mkdir(parents=True, exist_ok=True)
    review.mkdir(parents=True, exist_ok=True)

    rows = window(read_frames(src), args.start, args.end)
    t_from, t_to = int(rows[0]["timestamp_ns"]), int(rows[-1]["timestamp_ns"])
    with open(out / "frames.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} frames, {(t_to - t_from) / 1e9:.0f} s, -> {out}")

    can_rows, dropped = 0, []
    if (src / "decoded.csv").exists():
        can_rows, dropped = copy_decoded(src / "decoded.csv", out / "decoded.csv", t_from, t_to)
        print(f"decoded.csv: {can_rows} rows, removed {', '.join(dropped) or 'nothing'}")
    else:
        print("no decoded.csv in the session: the virtual Pi's CAN panel will be empty")
    if (src / "session.json").exists():
        shutil.copy2(src / "session.json", out / "session.json")

    quality = args.quality or (85 if args.size else JPEG_QUALITY)
    anonymizer = Anonymizer()
    counts = {"face": 0, "plate": 0}
    frames_blurred = reviewed = skipped = 0
    started = time.time()
    for k, row in enumerate(rows):
        dst = out / row["filename"]
        if dst.exists():
            skipped += 1
            continue
        raw = cv.imread(str(src / row["filename"]))
        if raw is None:
            print(f"unreadable, left out: {row['filename']}")
            continue
        clean, regions = anonymizer.anonymize(raw)
        saved = cv.resize(clean, (args.size, args.size), interpolation=cv.INTER_AREA) if args.size else clean
        cv.imwrite(str(dst), saved, [cv.IMWRITE_JPEG_QUALITY, quality])
        for kind, *_ in regions:
            counts[kind] += 1
        if regions:
            frames_blurred += 1
            # spread the review samples over the drive, faces first (rarer and more sensitive)
            if reviewed < REVIEW_MAX and (any(r[0] == "face" for r in regions) or frames_blurred % 25 == 1):
                write_review(review, Path(row["filename"]).name, raw, clean, regions)
                reviewed += 1
        if (k + 1) % 100 == 0:
            rate = (time.time() - started) / (k + 1 - skipped)
            print(f"  {k + 1}/{len(rows)}  faces {counts['face']}  plate areas {counts['plate']}  "
                  f"~{rate * (len(rows) - k - 1) / 60:.0f} min left", flush=True)

    (out / "PRIVACY.md").write_text(f"""# Privacy processing

Made by `src/scripts/privacy_package.py` on {time.strftime('%Y-%m-%d')} from session `{src.name}`.

- **Removed:** raw `can.csv` (contains the vehicle's VIN and the OnStar GPS position) and `system.csv`.
- **decoded.csv:** columns removed: {', '.join(dropped) or 'none'}; {can_rows} rows kept.
- **Frames:** {len(rows)} frames, {(t_to - t_from) / 1e9:.0f} s of driving, saved {f"at {args.size} x {args.size}" if args.size else "at recorded size"}, JPEG q{quality} (blurring was done at full resolution).
  - Faces (YuNet, OpenCV model zoo, score >= {FACE_SCORE}): {counts['face']} regions blurred (includes false positives such as wheels; blurring them is harmless).
  - Number plates: the lower half of every vehicle box at least {MIN_VEHICLE_W} px wide (YOLOPv2): {counts['plate']} regions blurred.
  - {frames_blurred} frames had at least one region blurred in this run.
- Blurring is irreversible (pixelated, then smoothed) and was applied before the JPEG was written.
- Automatic detection can miss things; check the before/after samples in the review folder before publishing.
""")
    print(f"done: faces {counts['face']}, plate areas {counts['plate']}, {frames_blurred} frames blurred, "
          f"{skipped} already done; review samples in {review}")


if __name__ == "__main__":
    main()
