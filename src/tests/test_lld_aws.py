"""Tests for lld_aws.py -- no real AWS and no model needed.

S3 and SQS are small in-memory fakes; lld.process_frame is replaced by a stub that returns
blank images of the right sizes, so these run in milliseconds.
"""

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import cv2 as cv
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import lld  # noqa: E402

try:
    import lld_aws  # noqa: E402
except ImportError:
    # boto3 isn't needed with the fakes below; stand in for it only while importing lld_aws,
    # so other test modules in the same run still see the real (or missing) boto3
    with mock.patch.dict(sys.modules, {"boto3": types.ModuleType("boto3")}):
        import lld_aws  # noqa: E402


class MissingError(Exception):
    """Shaped like botocore's ClientError for a missing key."""
    def __init__(self):
        super().__init__("NoSuchKey")
        self.response = {"Error": {"Code": "NoSuchKey"}}


class FakeS3:
    def __init__(self, objects=None):
        self.objects = dict(objects or {})
        self.puts = []  # every key written, in order

    def get_paginator(self, name):
        fake = self

        class Paginator:
            def paginate(self, Bucket, Prefix):
                keys = sorted(k for k in fake.objects if k.startswith(Prefix))
                for i in range(0, max(len(keys), 1), 2):  # 2 keys per page, to exercise pagination
                    yield {"Contents": [{"Key": k} for k in keys[i:i + 2]]} if keys else {}
        return Paginator()

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise MissingError()
        return {"Body": types.SimpleNamespace(read=lambda: self.objects[Key])}

    def put_object(self, Bucket, Key, Body, ContentType=None):
        self.objects[Key] = Body
        self.puts.append(Key)

    def upload_file(self, path, Bucket, Key, ExtraArgs=None):
        self.objects[Key] = Path(path).read_bytes()
        self.puts.append(Key)

    def metrics(self, folder):
        return json.loads(self.objects[f"lld/{folder}/metrics.json"])


class FakeSQS:
    def __init__(self, bodies):
        self.queue = [{"Body": json.dumps(b), "ReceiptHandle": f"rh{i}"} for i, b in enumerate(bodies)]
        self.deleted = []

    def receive_message(self, QueueUrl, MaxNumberOfMessages, **kwargs):
        batch, self.queue = self.queue[:MaxNumberOfMessages], self.queue[MaxNumberOfMessages:]
        return {"Messages": batch} if batch else {}

    def delete_message(self, QueueUrl, ReceiptHandle):
        self.deleted.append(ReceiptHandle)


def jpeg():
    return cv.imencode(".jpg", np.zeros((64, 64, 3), np.uint8))[1].tobytes()


def fake_process_frame(frame, model, device):
    metrics = {"offset_m": 0.1, "left_off_m": -1.7, "right_off_m": 1.9, "road_width_m": 3.6,
               "left_present": True, "right_present": True, "left_coeff": None, "right_coeff": None,
               "lane_valid": True, "departure": None, "departure_text": "In lane"}
    overlay = np.zeros((lld.MODEL_SIZE, lld.MODEL_SIZE, 3), np.uint8)
    warped = np.zeros((lld.canvas_h, lld.canvas_w, 3), np.uint8)
    return metrics, overlay, warped


def live_msg(session, n, t0=1792505112.0):
    return {"bucket": "b", "key": f"live/{session}/{n:04d}.jpg", "session_start": session,
            "frame_time": t0 + n * 0.5, "frame_seq": n}


class Base(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(lld, "process_frame", fake_process_frame)
        p.start()
        self.addCleanup(p.stop)
        self.work = tempfile.TemporaryDirectory()
        self.addCleanup(self.work.cleanup)


class SessionIdTest(unittest.TestCase):
    def test_recorder_session_layout(self):
        self.assertEqual(lld_aws.session_id("sessions/20261001T170034Z-d015/frames/"), "20261001T170034Z-d015")

    def test_live_layout(self):
        self.assertEqual(lld_aws.session_id("live/20261020T140512Z/"), "20261020T140512Z")
        self.assertEqual(lld_aws.session_id("live/20261020T140512Z"), "20261020T140512Z")

    def test_timestamp_drops_suffix(self):
        self.assertEqual(lld_aws.session_timestamp("20261001T170034Z-d015"), "20261001T170034Z")
        self.assertEqual(lld_aws.session_timestamp("20261020T140512Z"), "20261020T140512Z")


class ReplayTest(Base):
    def test_processes_all_frames_and_uploads(self):
        s3 = FakeS3({f"live/S1/{i:04d}.jpg": jpeg() for i in range(5)} | {"live/S1/notes.txt": b"x"})
        doc = lld_aws.run_replay(s3, "b", "live/S1/", None, "cpu", self.work.name, show=False)
        self.assertEqual(doc["frames_processed"], 5)
        self.assertIn("lld/S1/overlay.mp4", s3.objects)
        self.assertIn("lld/S1/bev.mp4", s3.objects)
        self.assertEqual(s3.metrics("S1")["frames_processed"], 5)

    def test_replay_starts_over_instead_of_resuming(self):
        s3 = FakeS3({f"live/S1/{i:04d}.jpg": jpeg() for i in range(3)})
        lld_aws.run_replay(s3, "b", "live/S1/", None, "cpu", self.work.name, show=False)
        lld_aws.run_replay(s3, "b", "live/S1/", None, "cpu", self.work.name, show=False)
        doc = s3.metrics("S1")
        self.assertEqual(doc["frames_processed"], 3)
        self.assertEqual(len(doc["videos"]), 1)


class LiveTest(Base):
    def run_live(self, s3, sqs, receives=5):
        with mock.patch.object(lld_aws, "SESSION_IDLE_S", 0):
            return lld_aws.run_live(s3, sqs, "q.fifo", None, "cpu", self.work.name, show=False,
                                    max_receives=receives)

    def test_processes_queue_and_finalizes_session(self):
        msgs = [live_msg("S1", i) for i in range(7)]
        s3 = FakeS3({m["key"]: jpeg() for m in msgs})
        sqs = FakeSQS(msgs)
        self.run_live(s3, sqs)

        doc = s3.metrics("S1")
        self.assertEqual(doc["frames_processed"], 7)
        self.assertEqual([f["frame_seq"] for f in doc["frames"]], list(range(7)))
        self.assertIn("lag_s", doc["frames"][0])
        self.assertEqual(doc["videos"], [{"frames": 7, "overlay": "lld/S1/overlay.mp4", "bev": "lld/S1/bev.mp4"}])
        self.assertEqual(len(sqs.deleted), 7)

    def test_new_session_finalizes_the_previous_one(self):
        msgs = [live_msg("S1", 0), live_msg("S1", 1), live_msg("S2", 0)]
        s3 = FakeS3({m["key"]: jpeg() for m in msgs})
        self.run_live(s3, FakeSQS(msgs))
        self.assertEqual(s3.metrics("S1")["frames_processed"], 2)
        self.assertEqual(s3.metrics("S2")["frames_processed"], 1)

    def test_redelivered_message_is_not_processed_twice(self):
        msgs = [live_msg("S1", 0), live_msg("S1", 1), live_msg("S1", 1)]
        s3 = FakeS3({m["key"]: jpeg() for m in msgs})
        sqs = FakeSQS(msgs)
        self.run_live(s3, sqs)
        self.assertEqual(s3.metrics("S1")["frames_processed"], 2)
        self.assertEqual(len(sqs.deleted), 3)  # the duplicate is still removed from the queue

    def test_missing_frame_is_dropped_not_retried(self):
        msgs = [live_msg("S1", 0), live_msg("S1", 1)]
        s3 = FakeS3({msgs[0]["key"]: jpeg()})  # second frame's object doesn't exist
        sqs = FakeSQS(msgs)
        self.run_live(s3, sqs)
        self.assertEqual(s3.metrics("S1")["frames_processed"], 1)
        self.assertEqual(len(sqs.deleted), 2)

    def test_transient_read_error_leaves_message_for_retry(self):
        msgs = [live_msg("S1", 0)]
        s3 = FakeS3({msgs[0]["key"]: jpeg()})
        sqs = FakeSQS(msgs)
        real_get = s3.get_object

        def flaky_get(Bucket, Key):
            if Key.endswith(".jpg"):
                raise ConnectionError("network down")
            return real_get(Bucket=Bucket, Key=Key)

        with mock.patch.object(s3, "get_object", side_effect=flaky_get):
            self.run_live(s3, sqs, receives=1)
        self.assertEqual(sqs.deleted, [])

    def test_session_start_error_leaves_message_for_retry(self):
        msgs = [live_msg("S1", 0)]
        s3 = FakeS3({msgs[0]["key"]: jpeg()})
        sqs = FakeSQS(msgs)
        with mock.patch.object(s3, "get_object", side_effect=ConnectionError("network down")):
            self.run_live(s3, sqs, receives=1)
        self.assertEqual(sqs.deleted, [])
        self.assertNotIn("lld/S1/metrics.json", s3.objects)

    def test_restart_resumes_session_as_a_new_video_part(self):
        first = [live_msg("S1", i) for i in range(3)]
        second = [live_msg("S1", i) for i in range(2, 6)]  # frame 2 re-delivered after the restart
        s3 = FakeS3({m["key"]: jpeg() for m in first + second})
        self.run_live(s3, FakeSQS(first))
        self.run_live(s3, FakeSQS(second))

        doc = s3.metrics("S1")
        self.assertEqual(doc["frames_processed"], 6)
        self.assertEqual([f["index"] for f in doc["frames"]], list(range(6)))
        self.assertEqual([v["frames"] for v in doc["videos"]], [3, 3])
        self.assertIn("lld/S1/overlay_part2.mp4", s3.objects)

    def test_checkpoints_metrics_during_a_session(self):
        msgs = [live_msg("S1", i) for i in range(5)]
        s3 = FakeS3({m["key"]: jpeg() for m in msgs})
        with mock.patch.object(lld_aws, "CHECKPOINT_EVERY", 2):
            self.run_live(s3, FakeSQS(msgs))
        # 2 checkpoints (after frames 2 and 4) + the final upload
        self.assertEqual(s3.puts.count("lld/S1/metrics.json"), 3)


if __name__ == "__main__":
    unittest.main()
