"""Tests for signs.py: the CTC scorer against brute force, and the full reader on drawn signs.

The reader tests need the CRNN weights (signs.OCR_WEIGHTS) and are skipped without them.
"""

import itertools
import sys
import unittest
from pathlib import Path

import cv2 as cv
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import signs  # noqa: E402

HAVE_WEIGHTS = (ROOT / signs.OCR_WEIGHTS).exists()


def collapse(path):
    """CTC: merge repeats, then drop blanks (0)."""
    out = [c for k, c in enumerate(path) if k == 0 or c != path[k - 1]]
    return [c for c in out if c != 0]


class CtcTest(unittest.TestCase):
    def test_matches_brute_force_sum_over_alignments(self):
        rng = np.random.default_rng(0)
        T, classes = 6, 1 + len(signs.OCR_VOCAB)
        allowed = [0, signs.OCR_VOCAB.index("s") + 1, signs.OCR_VOCAB.index("t") + 1]
        lp = np.full((T, classes), -np.inf)
        raw = rng.normal(size=(T, 3))
        lp[:, allowed] = raw - np.log(np.exp(raw).sum(axis=1, keepdims=True))

        for word in ("st", "ts", "s", "ss"):
            target = [signs.OCR_VOCAB.index(c) + 1 for c in word]
            brute = np.logaddexp.reduce([
                sum(lp[t, c] for t, c in enumerate(path))
                for path in itertools.product(allowed, repeat=T) if collapse(list(path)) == target
            ])
            self.assertAlmostEqual(signs._ctc_logp(lp, word), brute, places=6, msg=word)

    def test_greedy_collapses_repeats_and_blanks(self):
        lp = np.full((5, 1 + len(signs.OCR_VOCAB)), -10.0)
        for t, c in enumerate("ss-to"):  # "-" = blank
            lp[t, 0 if c == "-" else signs.OCR_VOCAB.index(c) + 1] = 0.0
        self.assertEqual(signs._greedy(lp)[0], "sto")


def scene():
    """Gray-blue 'sky' over a darker 'road', the size of an undistorted frame."""
    img = np.full((1200, 1200, 3), (150, 120, 110), np.uint8)
    img[700:] = (70, 70, 70)
    return img


def draw_stop(img, cx, cy, r):
    angles = np.deg2rad(22.5 + 45 * np.arange(8))
    pts = np.stack([cx + r * np.cos(angles), cy + r * np.sin(angles)], axis=1).astype(np.int32)
    cv.fillPoly(img, [pts], (30, 30, 200))
    scale = r / 38
    (tw, th), _ = cv.getTextSize("STOP", cv.FONT_HERSHEY_SIMPLEX, scale, max(1, int(3 * scale)))
    cv.putText(img, "STOP", (cx - tw // 2, cy + th // 2), cv.FONT_HERSHEY_SIMPLEX, scale,
               (255, 255, 255), max(1, int(3 * scale)), cv.LINE_AA)


def draw_speed(img, x, y, w, value):
    h = int(w * 1.25)
    cv.rectangle(img, (x, y), (x + w, y + h), (245, 245, 245), -1)
    cv.rectangle(img, (x + 3, y + 3), (x + w - 3, y + h - 3), (0, 0, 0), 2)
    for text, cy, scale, thick in (("SPEED", 0.22, w / 140, 2), ("LIMIT", 0.42, w / 140, 2), (str(value), 0.82, w / 50, 5)):
        (tw, _), _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv.putText(img, text, (x + (w - tw) // 2, y + int(h * cy)), cv.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick, cv.LINE_AA)


@unittest.skipUnless(HAVE_WEIGHTS, f"needs {signs.OCR_WEIGHTS}")
class SignReaderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reader = signs.SignReader(str(ROOT / signs.OCR_WEIGHTS))

    def test_empty_scene(self):
        self.assertEqual(self.reader.detect(scene()), [])

    def test_stop_sign(self):
        img = scene()
        draw_stop(img, 900, 400, 45)
        found = self.reader.detect(cv.GaussianBlur(img, (0, 0), 1.0))
        self.assertEqual([s["type"] for s in found], ["stop"])
        x1, y1, x2, y2 = found[0]["box"]
        self.assertTrue(x1 < 900 < x2 and y1 < 400 < y2)

    def test_speed_limit_sign(self):
        img = scene()
        draw_speed(img, 850, 300, 90, 45)
        found = self.reader.detect(cv.GaussianBlur(img, (0, 0), 1.0))
        self.assertEqual([(s["type"], s["value"]) for s in found], [("speed_limit", 45)])

    def test_small_speed_sign_is_not_read(self):
        # below SPEED_MIN_WIDTH digits are too small to tell 5 from 6: wait until it's closer
        img = scene()
        draw_speed(img, 850, 300, 40, 55)
        self.assertEqual(self.reader.detect(img), [])


if __name__ == "__main__":
    unittest.main()
