#!/usr/bin/env python3
"""Time one OpenCV operation on a real frame, using whatever OpenCV is
currently active in this Python environment, and print the result.

Run this TWICE on the COOL EC2 instance, once per environment, to get the
actual comparison -- this script itself doesn't switch environments, since
that's an OS-level thing (which venv you sourced), not something Python can
do to itself mid-run:

    # 1) COOL's accelerated build:
    source /opt/cool/venvs/py312/bin/activate      # path per the COOL AMI's docs; confirm the
                                                    # exact path once you're on the instance --
                                                    # this is the one detail not yet verified
                                                    # first-hand, only from the marketplace listing
    python3 time_cool_vs_stock.py one_frame.jpg
    deactivate

    # 2) stock OpenCV, same machine, same frame, for a fair comparison:
    pip install opencv-python-headless
    python3 time_cool_vs_stock.py one_frame.jpg

Compare the two "mean_ms" numbers by hand -- this script does not claim to
know which environment it's running in, on purpose: that claim belongs to
you, made once you've actually confirmed which venv is active, not to a
script guessing at it.

Operations timed: resize and adaptiveThreshold, both explicitly named in
COOL's own documentation as accelerated. Picked for exactly that reason --
so a difference (or lack of one) is a real signal, not noise from testing
something COOL never claimed to speed up.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time


def load_gray(path: str):
    import cv2

    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"could not read {path} as an image (wrong path, or not a decodable image file)")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def time_op(fn, repeats: int) -> dict:
    fn()  # warm-up: first call pays one-time setup cost, not representative
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times.sort()
    return {
        "mean_ms": round(sum(times) / len(times) * 1000, 3),
        "median_ms": round(statistics.median(times) * 1000, 3),
        "min_ms": round(times[0] * 1000, 3),
        "max_ms": round(times[-1] * 1000, 3),
    }


def run_benchmark(image_path: str, repeats: int = 50) -> dict:
    import cv2

    gray = load_gray(image_path)
    h, w = gray.shape[:2]

    results = {
        "opencv_version": cv2.__version__,
        "image_size": [w, h],
        "resize_half": time_op(lambda: cv2.resize(gray, (w // 2, h // 2)), repeats),
        "adaptive_threshold": time_op(
            lambda: cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),
            repeats,
        ),
    }
    return results


def print_results(results: dict) -> None:
    print(f"OpenCV version: {results['opencv_version']}")
    print(f"Image size: {results['image_size'][0]}x{results['image_size'][1]}")
    for name in ("resize_half", "adaptive_threshold"):
        r = results[name]
        print(f"  {name}: mean {r['mean_ms']} ms, median {r['median_ms']} ms "
              f"(min {r['min_ms']}, max {r['max_ms']})")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image_path", help="a real frame, e.g. one from a Phase 1 session's frames/ folder")
    parser.add_argument("--repeats", type=int, default=50, help="timed repeats per operation (default 50)")
    args = parser.parse_args(argv)

    try:
        import cv2  # noqa: F401
    except ImportError:
        print("cv2 (OpenCV) is not importable in this Python environment.\n"
              "If you meant to test COOL: source /opt/cool/venvs/<version>/bin/activate first.\n"
              "If you meant to test stock OpenCV: pip install opencv-python-headless", file=sys.stderr)
        return 2

    results = run_benchmark(args.image_path, repeats=args.repeats)
    print_results(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
