"""Tests for src/scripts/privacy_package.py. The model-free parts always run; the full package
run needs the YuNet and YOLOPv2 weights and is skipped without them."""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import cv2 as cv
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "scripts"))
import privacy_package as pp  # noqa: E402

HAVE_WEIGHTS = (ROOT / pp.FACE_WEIGHTS).exists() and (ROOT / pp.lld.DEFAULT_WEIGHTS).exists()


def fake_session(d, n=5):
    """A session folder with n frames 0.5 s apart, a decoded.csv with GPS, and a raw can.csv."""
    (d / "frames").mkdir()
    rows = []
    for k in range(n):
        name = f"frames/{k:08d}_{1_000_000_000 + k * 500_000_000}.jpg"
        img = np.random.default_rng(k).integers(0, 255, (200, 200, 3), dtype=np.uint8)
        cv.imwrite(str(d / name), img)
        rows.append({"seq": k, "filename": name, "timestamp_ns": 1_000_000_000 + k * 500_000_000})
    with open(d / "frames.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seq", "filename", "timestamp_ns"])
        w.writeheader()
        w.writerows(rows)
    with open(d / "decoded.csv", "w", newline="") as f:
        f.write("timestamp_ns,seq,message,VehicleSpeed,GPSLatitude,GPSLongitude\n")
        for k in range(n):
            f.write(f"{1_000_000_000 + k * 500_000_000},{k},m,{k},42.1,-83.2\n")
    (d / "can.csv").write_text("seq,timestamp_ns,can_id,data\n1,1,0x514,4a4b4c\n")  # VIN-bearing raw CAN
    (d / "session.json").write_text("{}")


class ModelFreeTest(unittest.TestCase):
    def test_blur_region_changes_only_that_region(self):
        img = np.random.default_rng(0).integers(0, 255, (100, 100, 3), dtype=np.uint8)
        before = img.copy()
        self.assertTrue(pp.blur_region(img, 20, 30, 60, 70))
        self.assertFalse(np.array_equal(img[30:70, 20:60], before[30:70, 20:60]))
        mask = np.ones((100, 100), bool)
        mask[30:70, 20:60] = False
        self.assertTrue(np.array_equal(img[mask], before[mask]))
        self.assertFalse(pp.blur_region(img, 150, 150, 160, 160))  # off the image: nothing

    def test_window_and_decoded_copy_drop_gps(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            fake_session(d)
            rows = pp.window(pp.read_frames(d), 0.5, 1.5)
            self.assertEqual([int(r["seq"]) for r in rows], [1, 2, 3])
            n, dropped = pp.copy_decoded(d / "decoded.csv", d / "out.csv",
                                         int(rows[0]["timestamp_ns"]), int(rows[-1]["timestamp_ns"]))
            self.assertEqual((n, dropped), (3, ["GPSLatitude", "GPSLongitude"]))
            header = (d / "out.csv").read_text().splitlines()[0]
            self.assertEqual(header, "timestamp_ns,seq,message,VehicleSpeed")


@unittest.skipUnless(HAVE_WEIGHTS, "needs the YuNet and YOLOPv2 weights")
class PackageTest(unittest.TestCase):
    def test_package_contents(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, out = Path(tmp) / "S1", Path(tmp) / "out"
            src.mkdir()
            fake_session(src, n=3)
            import os
            cwd = os.getcwd()
            os.chdir(ROOT)  # weights paths are relative to the repo root
            try:
                pp.main([str(src), str(out)])
            finally:
                os.chdir(cwd)
            pkg = out / "S1"
            self.assertEqual(sorted(p.name for p in pkg.iterdir()),
                             ["PRIVACY.md", "decoded.csv", "frames", "frames.csv", "session.json"])
            self.assertEqual(len(list((pkg / "frames").iterdir())), 3)
            self.assertNotIn("GPS", (pkg / "decoded.csv").read_text())


if __name__ == "__main__":
    unittest.main()
