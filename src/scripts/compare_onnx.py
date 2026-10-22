#!/usr/bin/env python3
"""Check the ONNX model on cv.dnn against the TorchScript original, on real frames.

    python src/scripts/compare_onnx.py <frames_dir> [--n 50]
    python src/scripts/compare_onnx.py ../initiative-vision/sessions/20260927T184255Z-fddd/frames

For n frames spread evenly over the folder, runs both backends and reports:
  - lane-line and drivable-area mask IoU (cv.dnn vs torch)
  - detections: how often the box count matches and the worst box IoU between matched boxes
  - how often the final metrics agree (lane_valid, departure) and the largest offset difference
  - inference time per backend

Needs torch (for the reference). Exits non-zero if the lane-mask IoU drops below --min-iou.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lld  # noqa: E402


def iou(a, b):
    a, b = a > 0, b > 0
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("frames_dir")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--torch-weights", default="data/weights/yolopv2.pt")
    parser.add_argument("--onnx-weights", default="data/weights/yolopv2.onnx")
    parser.add_argument("--min-iou", type=float, default=0.95)
    args = parser.parse_args(argv)

    import torch
    t_model, t_device = lld.load_model(args.torch_weights, device=torch.device("cpu"))
    d_model, d_device = lld.load_model(args.onnx_weights)

    frames = sorted(p for p in Path(args.frames_dir).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    frames = frames[::max(1, len(frames) // args.n)][:args.n]

    ll_ious, da_ious, t_times, d_times, offset_diffs = [], [], [], [], []
    valid_agree = departure_agree = det_count_agree = 0
    box_ious = []
    for path in frames:
        frame = cv.imread(str(path))
        if frame is None:
            continue
        model_in = cv.resize(frame, (lld.MODEL_SIZE, lld.MODEL_SIZE), interpolation=cv.INTER_AREA)
        t_da, t_ll, t_det, _ = lld._infer_torch(t_model, t_device, model_in)
        d_da, d_ll, d_det, _ = lld._infer_dnn(d_model, model_in)
        ll_ious.append(iou(t_ll, d_ll))
        da_ious.append(iou(t_da, d_da))

        t_objs, d_objs = lld.decode_detections(t_det), lld.decode_detections(d_det)
        det_count_agree += len(t_objs) == len(d_objs)
        for t_obj in t_objs:  # best-matching cv.dnn box for each torch box
            box_ious.append(max((box_iou(t_obj["box"], d["box"]) for d in d_objs), default=0.0))

        t_metrics, _, _ = lld.process_frame(frame, t_model, t_device)
        d_metrics, _, _ = lld.process_frame(frame, d_model, d_device)
        t_times.append(t_metrics["infer_s"])
        d_times.append(d_metrics["infer_s"])
        valid_agree += t_metrics["lane_valid"] == d_metrics["lane_valid"]
        departure_agree += t_metrics["departure"] == d_metrics["departure"]
        if t_metrics["offset_m"] is not None and d_metrics["offset_m"] is not None:
            offset_diffs.append(abs(t_metrics["offset_m"] - d_metrics["offset_m"]))

    n = len(ll_ious)
    print(f"{n} frames from {args.frames_dir}  (OpenCV {cv.__version__}, torch {torch.__version__})")
    print(f"lane mask IoU       mean {statistics.mean(ll_ious):.4f}  min {min(ll_ious):.4f}")
    print(f"drivable mask IoU   mean {statistics.mean(da_ious):.4f}  min {min(da_ious):.4f}")
    print(f"detection count     agrees {det_count_agree}/{n}, {len(box_ious)} torch boxes")
    if box_ious:
        print(f"detection box IoU   mean {statistics.mean(box_ious):.4f}  min {min(box_ious):.4f}")
    print(f"lane_valid agrees   {valid_agree}/{n}")
    print(f"departure agrees    {departure_agree}/{n}")
    if offset_diffs:
        print(f"offset |diff|       max {max(offset_diffs):.4f} m over {len(offset_diffs)} frames with both")
    print(f"inference torch     median {statistics.median(t_times):.3f} s")
    print(f"inference cv.dnn    median {statistics.median(d_times):.3f} s")
    return 0 if min(ll_ious) >= args.min_iou else 1


if __name__ == "__main__":
    sys.exit(main())
