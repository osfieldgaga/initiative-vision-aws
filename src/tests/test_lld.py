"""Tests for lld.decode_detections -- synthetic detection maps, no model needed."""

import sys
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
