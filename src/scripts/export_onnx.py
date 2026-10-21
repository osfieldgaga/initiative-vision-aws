#!/usr/bin/env python3
"""Export the TorchScript YOLOPv2 to ONNX for OpenCV's DNN module (cv.dnn).

Only the two segmentation heads lld.py uses are kept (drivable area, lane lines). The full model
is exported first, then the detection outputs are removed and the graph pruned to what da/ll
need. (Wrapping the ScriptModule to return only da/ll doesn't work: torch can't trace into it.)

    python src/scripts/export_onnx.py                          # data/weights/yolopv2.pt -> data/weights/yolopv2_seg.onnx
    python src/scripts/export_onnx.py --weights a.pt --out b.onnx

Outputs (fixed 1x3x640x640 float input, RGB, 0..1):
    da  1x2x640x640  drivable-area scores (argmax over dim 1)
    ll  1x1x640x640  lane-line probability (lld.LANE_THRESHOLD applies to it)

Run once on a machine with torch (it isn't needed at inference time). Check the result
against TorchScript with src/scripts/compare_onnx.py.
"""

from __future__ import annotations

import argparse
import os

import onnx
import onnx.utils
import torch


# the model returns ([3 detection maps], [3 anchor grids]), da, ll; the exporter turns the
# detection maps into one output and flattens the rest
FLAT_OUTPUTS = ["det", "grid0", "grid1", "grid2", "da", "ll"]
KEEP = ["da", "ll"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", default="data/weights/yolopv2.pt")
    parser.add_argument("--out", default="data/weights/yolopv2_seg.onnx")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args(argv)

    model = torch.jit.load(args.weights, map_location="cpu").eval()
    dummy = torch.zeros(1, 3, 640, 640)
    full = args.out + ".full"
    with torch.no_grad():
        torch.onnx.export(
            model, (dummy,), full,
            input_names=["images"], output_names=FLAT_OUTPUTS,
            opset_version=args.opset,
            dynamo=False,  # the TorchScript-based exporter; the dynamo one can't take a ScriptModule
        )

    # keep only da/ll and the nodes that feed them
    onnx.utils.extract_model(full, args.out, ["images"], KEEP)
    os.remove(full)
    onnx.checker.check_model(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
