#!/usr/bin/env python3
"""Export the TorchScript YOLOPv2 to ONNX for OpenCV's DNN module (cv.dnn).

Kept: the two segmentation heads (drivable area, lane lines) and the three raw detection maps.
The full model is exported first; the exporter packs the detection maps into an ONNX Sequence,
which cv.dnn can't read, so they're re-exposed as plain outputs and the sequence and the
constant anchor grids (hardcoded in lld.ANCHORS) are dropped. (Wrapping the ScriptModule to
return only what we need doesn't work: torch can't trace into it.)

    python src/scripts/export_onnx.py                          # data/weights/yolopv2.pt -> data/weights/yolopv2.onnx
    python src/scripts/export_onnx.py --weights a.pt --out b.onnx

Outputs (fixed 1x3x640x640 float input, RGB, 0..1):
    da     1x2x640x640  drivable-area scores (argmax over dim 1)
    ll     1x1x640x640  lane-line probability (lld.LANE_THRESHOLD applies to it)
    det8   1x255x80x80  raw detection maps at stride 8/16/32, 3 anchors x (xywh, obj, 80 classes),
    det16  1x255x40x40  before sigmoid -- decoded by lld.decode_detections
    det32  1x255x20x20

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
DET_OUTPUTS = ["det8", "det16", "det32"]
KEEP = ["da", "ll"] + DET_OUTPUTS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", default="data/weights/yolopv2.pt")
    parser.add_argument("--out", default="data/weights/yolopv2.onnx")
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

    # the "det" output is a SequenceConstruct of the three maps: expose its inputs instead
    model_proto = onnx.load(full)
    graph = model_proto.graph
    seq = next(n for n in graph.node if "det" in n.output)
    for name, src in zip(DET_OUTPUTS, seq.input):
        graph.node.append(onnx.helper.make_node("Identity", [src], [name]))
    graph.node.remove(seq)
    size = 640
    shapes = {"det8": size // 8, "det16": size // 16, "det32": size // 32}
    kept = [o for o in graph.output if o.name in ("da", "ll")]
    del graph.output[:]
    graph.output.extend(kept)
    for name in DET_OUTPUTS:
        n = shapes[name]
        graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [1, 255, n, n]))
    onnx.save(model_proto, full)

    # keep only those outputs and the nodes that feed them
    onnx.utils.extract_model(full, args.out, ["images"], KEEP)
    os.remove(full)
    onnx.checker.check_model(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
