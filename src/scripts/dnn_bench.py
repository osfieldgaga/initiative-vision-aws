#!/usr/bin/env python3
"""Time YOLOPv2 (data/weights/yolopv2.onnx) on cv.dnn for each engine/target combination.

    python src/scripts/dnn_bench.py          # from the repo root; stop lld_aws.py first

Random 1x3x640x640 input (a conv net's cost doesn't depend on the pixels), one warm-up pass,
then the median and min of 5 forward passes requesting all outputs, as lld._infer_dnn does.
Results are recorded in phase2/ec2-compute-benchmark.md.
"""
import time

import cv2 as cv
import numpy as np

WEIGHTS = "data/weights/yolopv2.onnx"
OUTPUTS = ["da", "ll", "det8", "det16", "det32"]
RUNS = 5


def bench(x, engine, target, name):
    try:
        net = cv.dnn.readNetFromONNX(WEIGHTS, engine=engine)
        net.setPreferableTarget(target)
        net.setInput(x)
        net.forward(OUTPUTS)  # warm-up
        t = []
        for _ in range(RUNS):
            s = time.perf_counter()
            net.setInput(x)
            net.forward(OUTPUTS)
            t.append(time.perf_counter() - s)
        print(f"{name:18s} median {sorted(t)[RUNS // 2]:.3f}s  min {min(t):.3f}s")
    except Exception as e:
        print(f"{name:18s} failed: {str(e).strip().splitlines()[-1][:150]}")


def main():
    x = np.random.rand(1, 3, 640, 640).astype(np.float32)
    print("OpenCV", cv.__version__, "threads", cv.getNumThreads())
    for engine, ename in ((cv.dnn.ENGINE_NEW, "new"), (cv.dnn.ENGINE_CLASSIC, "classic")):
        for target, tname in ((cv.dnn.DNN_TARGET_CPU, "cpu"), (cv.dnn.DNN_TARGET_CPU_FP16, "cpu_fp16")):
            bench(x, engine, target, f"{ename}/{tname}")


if __name__ == "__main__":
    main()
