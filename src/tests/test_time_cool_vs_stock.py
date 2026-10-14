"""Tests for phase2/scripts/time_cool_vs_stock.py.

Needs opencv-python(-headless) and numpy installed (not the standard-library-
only bar most of this project holds itself to -- this script's whole job is
timing real OpenCV calls, so it has to import the real thing). Uses a small
synthetic image, not a real session frame, so these tests never depend on
any particular session existing on disk.
"""

import importlib.util
import tempfile
import unittest
from pathlib import Path

try:
    import cv2
    import numpy as np
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "time_cool_vs_stock.py"
_spec = importlib.util.spec_from_file_location("time_cool_vs_stock", _SCRIPT)
tcs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tcs)


@unittest.skipUnless(HAVE_CV2, "opencv-python(-headless) not installed here")
class TimeOpTest(unittest.TestCase):
    def test_reports_the_expected_stats_with_sane_ordering(self):
        calls = {"n": 0}

        def fn():
            calls["n"] += 1

        r = tcs.time_op(fn, repeats=10)
        self.assertEqual(set(r), {"mean_ms", "median_ms", "min_ms", "max_ms"})
        self.assertLessEqual(r["min_ms"], r["median_ms"])
        self.assertLessEqual(r["median_ms"], r["max_ms"])
        # 1 warm-up call + `repeats` timed calls
        self.assertEqual(calls["n"], 11)

    def test_all_values_are_non_negative(self):
        r = tcs.time_op(lambda: None, repeats=5)
        for v in r.values():
            self.assertGreaterEqual(v, 0)


@unittest.skipUnless(HAVE_CV2, "opencv-python(-headless) not installed here")
class LoadGrayTest(unittest.TestCase):
    def test_loads_and_converts_to_single_channel(self):
        with tempfile.TemporaryDirectory() as d:
            path = f"{d}/test.jpg"
            color = np.zeros((40, 60, 3), dtype=np.uint8)
            color[:, :, 2] = 255  # a solid red image, BGR order (matches the real camera path)
            cv2.imwrite(path, color)
            gray = tcs.load_gray(path)
        self.assertEqual(gray.ndim, 2)
        self.assertEqual(gray.shape, (40, 60))

    def test_missing_file_gives_a_clear_error_not_a_none_crash_later(self):
        with self.assertRaises(ValueError):
            tcs.load_gray("/nonexistent/path/does_not_exist.jpg")


@unittest.skipUnless(HAVE_CV2, "opencv-python(-headless) not installed here")
class RunBenchmarkTest(unittest.TestCase):
    def test_runs_both_named_operations_and_reports_real_size(self):
        with tempfile.TemporaryDirectory() as d:
            path = f"{d}/test.jpg"
            rng = np.random.default_rng(0)
            cv2.imwrite(path, rng.integers(0, 255, (80, 100, 3), dtype=np.uint8))
            results = tcs.run_benchmark(path, repeats=5)
        self.assertEqual(results["image_size"], [100, 80])  # [w, h]
        self.assertIn("resize_half", results)
        self.assertIn("adaptive_threshold", results)
        self.assertEqual(results["opencv_version"], cv2.__version__)

    def test_print_results_does_not_raise(self):
        with tempfile.TemporaryDirectory() as d:
            path = f"{d}/test.jpg"
            cv2.imwrite(path, np.zeros((30, 30, 3), dtype=np.uint8))
            results = tcs.run_benchmark(path, repeats=3)
        tcs.print_results(results)  # just needs to not raise


class MainTest(unittest.TestCase):
    def test_missing_cv2_gives_a_clear_message(self):
        import sys
        from unittest import mock

        real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

        def blocked_import(name, *a, **kw):
            if name == "cv2":
                raise ImportError("No module named 'cv2'")
            return real_import(name, *a, **kw)

        with mock.patch("builtins.__import__", side_effect=blocked_import):
            code = tcs.main(["some_image.jpg"])
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
