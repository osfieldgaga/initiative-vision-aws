"""Tests for lld: decode_detections on synthetic detection maps, and calibration scaling for
smaller frames. No model needed."""

import sys
import unittest
from pathlib import Path

import cv2 as cv
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import lld  # noqa: E402

OFF = -20.0  # logit that sigmoids to ~0


def empty_maps():
    return [np.full((1, 255, lld.MODEL_SIZE // s, lld.MODEL_SIZE // s), OFF, np.float32) for s in lld.DET_STRIDES]


def put(maps, scale, anchor, gx, gy, obj_logit, cls=3, cls_logit=5.0):
    """One prediction with x/y/w/h logits 0, i.e. centered in its cell at exactly the anchor size."""
    c = anchor * 85
    maps[scale][0, c:c + 4, gy, gx] = 0.0
    maps[scale][0, c + 4, gy, gx] = obj_logit
    maps[scale][0, c + 5 + cls, gy, gx] = cls_logit


class DecodeDetectionsTest(unittest.TestCase):
    def test_nothing_above_threshold(self):
        self.assertEqual(lld.decode_detections(empty_maps()), [])

    def test_box_position_and_size(self):
        maps = empty_maps()
        put(maps, scale=0, anchor=0, gx=10, gy=20, obj_logit=5.0)
        (obj,) = lld.decode_detections(maps)
        # center (10.5, 20.5) cells * stride 8 = (84, 164); size = anchor 12x16
        self.assertEqual(obj["box"], [78.0, 156.0, 90.0, 172.0])
        self.assertEqual(obj["label"], "vehicle")
        self.assertAlmostEqual(obj["score"], 0.987, places=3)

    def test_nms_drops_overlapping_duplicate_and_sorts_nearest_first(self):
        maps = empty_maps()
        # two neighbouring stride-32 cells, 142x110 boxes 32 px apart (IoU ~0.63): one survives
        put(maps, scale=2, anchor=0, gx=5, gy=5, obj_logit=5.0)
        put(maps, scale=2, anchor=0, gx=6, gy=5, obj_logit=3.0)
        # a separate, lower (nearer) box
        put(maps, scale=0, anchor=0, gx=60, gy=70, obj_logit=5.0)
        objs = lld.decode_detections(maps)
        self.assertEqual(len(objs), 2)
        self.assertGreater(objs[0]["box"][3], objs[1]["box"][3])
        self.assertEqual(objs[1]["box"][0], 5.5 * 32 - 71)  # the higher-scoring duplicate kept


class UndistortScaleTest(unittest.TestCase):
    def test_half_size_frame_undistorts_like_the_full_one(self):
        # a checkerboard at the calibration size, and the same board at 640x640
        yy, xx = np.mgrid[0:1280, 0:1280]
        big = np.repeat((((xx // 80) + (yy // 80)) % 2 * 255).astype(np.uint8)[:, :, None], 3, axis=2)
        small = cv.resize(big, (640, 640), interpolation=cv.INTER_AREA)

        ub, us = lld.undistort_frame(big), lld.undistort_frame(small)
        self.assertAlmostEqual(us.shape[0] / ub.shape[0], 0.5, delta=0.01)
        self.assertAlmostEqual(us.shape[1] / ub.shape[1], 0.5, delta=0.01)
        a = cv.resize(ub, (320, 320), interpolation=cv.INTER_AREA).astype(float)
        b = cv.resize(us, (320, 320), interpolation=cv.INTER_AREA).astype(float)
        self.assertLess(np.abs(a - b).mean(), 10)  # unscaled intrinsics give ~100: a different warp


if __name__ == "__main__":
    unittest.main()
