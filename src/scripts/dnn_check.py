#!/usr/bin/env python3
"""Compare cv.dnn engine/target configurations for YOLOPv2 on real frames.

    python src/scripts/dnn_check.py <frames_dir> [n]      # from the repo root

For each config: median inference time (interleaved frame by frame, so run order can't bias it),
and against the new-engine FP32 reference: worst lane / drivable-area mask IoU, how often the
detection count matches, and the largest difference in the raw detection maps.
"""
import statistics
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lld  # noqa: E402

CONFIGS = {
    "new/cpu": (cv.dnn.ENGINE_NEW, cv.dnn.DNN_TARGET_CPU),
    "classic/cpu": (cv.dnn.ENGINE_CLASSIC, cv.dnn.DNN_TARGET_CPU),
    "classic/cpu_fp16": (cv.dnn.ENGINE_CLASSIC, cv.dnn.DNN_TARGET_CPU_FP16),
}
REF = "new/cpu"


def iou(a, b):
    a, b = a > 0, b > 0
    u = np.logical_or(a, b).sum()
    return 1.0 if u == 0 else float(np.logical_and(a, b).sum() / u)


def main():
    frames_dir, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 20
    paths = sorted(p for p in Path(frames_dir).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    paths = paths[::max(1, len(paths) // n)][:n]
    print(f"OpenCV {cv.__version__}, threads {cv.getNumThreads()}, {len(paths)} frames from {frames_dir}")

    nets = {}
    for name, (engine, target) in CONFIGS.items():
        net = cv.dnn.readNetFromONNX(lld.DEFAULT_WEIGHTS, engine=engine)
        net.setPreferableTarget(target)
        nets[name] = net

    others = [k for k in nets if k != REF]
    times = {k: [] for k in nets}
    ll_iou, da_iou = {k: [] for k in others}, {k: [] for k in others}
    det_same, max_diff = {k: 0 for k in others}, {k: 0.0 for k in others}
    for i, p in enumerate(paths):
        frame = cv.imread(str(p))
        if frame is None:
            continue
        model_in = cv.resize(frame, (lld.MODEL_SIZE, lld.MODEL_SIZE), interpolation=cv.INTER_AREA)
        out = {name: lld._infer_dnn(net, model_in) for name, net in nets.items()}
        if i > 0:  # first frame = warm-up
            for name in nets:
                times[name].append(out[name][3])
        r_da, r_ll, r_det, _ = out[REF]
        r_count = len(lld.decode_detections(r_det))
        for name in others:
            da, ll, det, _ = out[name]
            ll_iou[name].append(iou(r_ll, ll))
            da_iou[name].append(iou(r_da, da))
            det_same[name] += len(lld.decode_detections(det)) == r_count
            max_diff[name] = max(max_diff[name], max(float(np.abs(a - b).max()) for a, b in zip(r_det, det)))

    for name in nets:
        line = f"{name:18s} infer median {statistics.median(times[name]):.3f}s"
        if name in others:
            line += (f" | lane IoU min {min(ll_iou[name]):.4f} | drivable IoU min {min(da_iou[name]):.4f}"
                     f" | det count same {det_same[name]}/{len(paths)} | det map max diff {max_diff[name]:.4f}")
        print(line)


if __name__ == "__main__":
    main()
