#!/usr/bin/env python3
"""Time the OpenCV image operations the pipeline runs around inference, one frame at a time.

    python src/scripts/ops_bench.py [frame.jpg] [--size 640] [--runs 50]    # from the repo root

Same operations, sizes and order as lld.process_frame and signs.SignReader.detect on an
uploaded frame (640x640 by default). The input frame matters a little for decode/encode and
Canny/contours, so use a real one (any frame from a session); without one a synthetic road-like
image is used. Prints the median per operation and the sum.

For COOL vs stock on a COOL AMI, run it in COOL's venv, then in a stock venv with PYTHONPATH
cleared (see phase2/ec2-compute-benchmark.md): env -u PYTHONPATH ~/stock/bin/python ...
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2 as cv
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lld  # noqa: E402


def synthetic(size):
    img = np.full((size, size, 3), (150, 120, 110), np.uint8)
    img[size * 6 // 10:] = (70, 70, 70)
    for x in (size // 4, size * 3 // 4):
        cv.line(img, (x, size), (size // 2, size * 6 // 10), (240, 240, 240), max(2, size // 100))
    return img


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("frame", nargs="?")
    parser.add_argument("--size", type=int, default=640, help="uploaded frame size (0 = as read)")
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args(argv)

    raw = cv.imread(args.frame) if args.frame else synthetic(1280)
    if args.size:
        raw = cv.resize(raw, (args.size, args.size), interpolation=cv.INTER_AREA)
    jpg = cv.imencode(".jpg", raw, [cv.IMWRITE_JPEG_QUALITY, 70])[1]

    h, w = raw.shape[:2]
    mtx, new_mtx, roi = lld._undistort_params(w, h)
    map1, map2 = cv.initUndistortRectifyMap(mtx, lld.imx219_dist, None, new_mtx, (w, h), cv.CV_16SC2)
    x, y, rw, rh = roi
    upright = lld.undistort_frame(raw)
    model_in = cv.resize(upright, (lld.MODEL_SIZE, lld.MODEL_SIZE), interpolation=cv.INTER_AREA)
    mask = np.zeros((lld.MODEL_SIZE, lld.MODEL_SIZE), np.uint8)
    cv.line(mask, (160, 639), (300, 480), 255, 6)
    cv.line(mask, (480, 639), (340, 480), 255, 6)
    top = upright[:int(upright.shape[0] * 2 / 3)]
    hsv = cv.cvtColor(top, cv.COLOR_BGR2HSV)
    gray = cv.cvtColor(top, cv.COLOR_BGR2GRAY)
    red = cv.inRange(hsv, (0, 110, 60), (10, 255, 255))
    color = np.zeros_like(model_in)

    ops = [
        ("imdecode (JPEG q70)", lambda: cv.imdecode(jpg, cv.IMREAD_COLOR)),
        ("undistort, cv.undistort (maps rebuilt each call)", lambda: cv.undistort(raw, mtx, lld.imx219_dist, None, new_mtx)),
        ("undistort, cached maps + remap (as lld does)", lambda: cv.remap(raw, map1, map2, cv.INTER_LINEAR)),
        ("rotate 90", lambda: cv.rotate(raw[y:y + rh, x:x + rw], cv.ROTATE_90_CLOCKWISE)),
        ("resize to 640 (INTER_AREA)", lambda: cv.resize(upright, (640, 640), interpolation=cv.INTER_AREA)),
        ("blobFromImage", lambda: cv.dnn.blobFromImage(model_in, scalefactor=1 / 255.0, swapRB=True)),
        ("warpPerspective (lane mask -> BEV)", lambda: cv.warpPerspective(mask, lld.M, (lld.canvas_w, lld.canvas_h))),
        ("findContours (BEV mask)", lambda: cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)),
        ("addWeighted x2 (overlay)", lambda: cv.addWeighted(cv.addWeighted(model_in, 1.0, color, 0.3, 0), 1.0, color, 0.6, 0)),
        ("cvtColor BGR2HSV (signs)", lambda: cv.cvtColor(top, cv.COLOR_BGR2HSV)),
        ("cvtColor BGR2GRAY (signs)", lambda: cv.cvtColor(top, cv.COLOR_BGR2GRAY)),
        ("inRange x2 + morph close (signs)", lambda: cv.morphologyEx(
            cv.inRange(hsv, (0, 110, 60), (10, 255, 255)) | cv.inRange(hsv, (170, 110, 60), (180, 255, 255)),
            cv.MORPH_CLOSE, np.ones((5, 5), np.uint8))),
        ("Canny + dilate (signs)", lambda: cv.dilate(cv.Canny(gray, 50, 150), np.ones((3, 3), np.uint8))),
        ("findContours (signs, RETR_LIST)", lambda: cv.findContours(red, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)),
        ("imencode overlay (JPEG q80, live page)", lambda: cv.imencode(".jpg", model_in, [cv.IMWRITE_JPEG_QUALITY, 80])),
    ]

    print(f"OpenCV {cv.__version__}, threads {cv.getNumThreads()}, frame {w}x{h}, {args.runs} runs")
    total = 0.0
    for name, fn in ops:
        fn()  # warm-up
        t = []
        for _ in range(args.runs):
            s = time.perf_counter()
            fn()
            t.append(time.perf_counter() - s)
        ms = statistics.median(t) * 1000
        if "cv.undistort" not in name:  # lld uses the cached-maps variant
            total += ms
        print(f"  {name:48s} {ms:7.2f} ms")
    print(f"  {'sum (as the pipeline runs today)':48s} {total:7.2f} ms")


if __name__ == "__main__":
    main()
