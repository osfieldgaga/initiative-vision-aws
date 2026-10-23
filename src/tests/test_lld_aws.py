"""Tests for lld_aws.py -- no real AWS and no model needed.

S3 and SQS are small in-memory fakes; lld.process_frame is replaced by a stub that returns
blank images of the right sizes, so these run in milliseconds.
"""

import json
import subprocess
import sys
import tempfile
import time
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

    def get_queue_attributes(self, QueueUrl, AttributeNames):
        return {"Attributes": {"ApproximateNumberOfMessages": str(len(self.queue))}}


def jpeg():
    return cv.imencode(".jpg", np.zeros((64, 64, 3), np.uint8))[1].tobytes()


def fake_process_frame(frame, model, device, sign_reader=None):
    metrics = {"offset_m": 0.1, "left_off_m": -1.7, "right_off_m": 1.9, "road_width_m": 3.6,
               "left_present": True, "right_present": True, "left_coeff": None, "right_coeff": None,
               "lane_valid": True, "departure": None, "departure_text": "In lane"}
    overlay = np.zeros((lld.MODEL_SIZE, lld.MODEL_SIZE, 3), np.uint8)
    warped = np.zeros((lld.canvas_h, lld.canvas_w, 3), np.uint8)
    return metrics, overlay, warped


def live_msg(session, n, t0=None):
    """Queue message for frame n; captured just now unless t0 (epoch seconds) is given."""
    t0 = time.time() if t0 is None else t0
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


class LagBreakdownTest(unittest.TestCase):
    def test_splits_lag_into_hops(self):
        body = {"frame_time": 100.0, "uploaded_at": 101.5, "queued_at": 101.8}
        got = lld_aws.lag_breakdown(body, received_at=102.0, started_at=102.25)
        self.assertEqual(got, {"pi_to_s3_s": 1.5, "s3_to_queue_s": 0.3, "queue_s": 0.2, "batch_wait_s": 0.25})

    def test_capture_to_write_split_out_when_the_pi_sends_it(self):
        body = {"frame_time": 100.0, "written_at": 100.4, "uploaded_at": 101.5, "queued_at": 101.5}
        got = lld_aws.lag_breakdown(body, received_at=101.6, started_at=101.6)
        self.assertEqual((got["capture_to_write_s"], got["pi_to_s3_s"]), (0.4, 1.1))

    def test_old_lambda_messages_have_no_breakdown(self):
        self.assertEqual(lld_aws.lag_breakdown({"frame_time": 100.0}, 102.0, 102.0), {})


class StaleFrameTest(Base):
    def run_live(self, s3, sqs, max_lag=10, receives=3):
        with mock.patch.object(lld_aws, "SESSION_IDLE_S", 0), mock.patch.object(lld_aws, "MAX_LAG_S", max_lag):
            lld_aws.run_live(s3, sqs, "q.fifo", None, "cpu", self.work.name, show=False, max_receives=receives)

    def test_backlog_of_old_frames_is_skipped_down_to_the_newest(self):
        msgs = [live_msg("S1", i, t0=time.time() - 100) for i in range(6)]
        s3 = FakeS3({m["key"]: jpeg() for m in msgs})
        sqs = FakeSQS(msgs)
        self.run_live(s3, sqs)
        doc = s3.metrics("S1")
        self.assertEqual([f["frame_seq"] for f in doc["frames"]], [5])  # only the newest
        self.assertEqual(doc["frames_skipped"], 5)
        self.assertEqual(len(sqs.deleted), 6)

    def test_old_frame_is_still_processed_when_nothing_newer_waits(self):
        # e.g. the Pi's clock is off and every frame *looks* old -- must not skip everything
        msgs = [live_msg("S1", 0, t0=time.time() - 100)]
        s3 = FakeS3({m["key"]: jpeg() for m in msgs})
        self.run_live(s3, FakeSQS(msgs))
        doc = s3.metrics("S1")
        self.assertEqual(doc["frames_processed"], 1)
        self.assertEqual(doc["frames_skipped"], 0)

    def test_old_frame_is_skipped_when_newer_ones_are_still_queued(self):
        msgs = [live_msg("S1", i, t0=time.time() - 100) for i in range(12)]  # more than one batch
        s3 = FakeS3({m["key"]: jpeg() for m in msgs})
        self.run_live(s3, FakeSQS(msgs))
        doc = s3.metrics("S1")
        self.assertEqual([f["frame_seq"] for f in doc["frames"]], [11])
        self.assertEqual(doc["frames_skipped"], 11)

    def test_fresh_frames_are_never_skipped(self):
        msgs = [live_msg("S1", i) for i in range(6)]
        s3 = FakeS3({m["key"]: jpeg() for m in msgs})
        self.run_live(s3, FakeSQS(msgs))
        self.assertEqual(s3.metrics("S1")["frames_processed"], 6)

    def test_max_lag_zero_processes_everything(self):
        msgs = [live_msg("S1", i, t0=time.time() - 100) for i in range(6)]
        s3 = FakeS3({m["key"]: jpeg() for m in msgs})
        self.run_live(s3, FakeSQS(msgs), max_lag=0)
        doc = s3.metrics("S1")
        self.assertEqual(doc["frames_processed"], 6)
        self.assertEqual(doc["frames_skipped"], 0)

    def test_skipped_count_survives_a_restart(self):
        old = [live_msg("S1", i, t0=time.time() - 100) for i in range(3)]
        s3 = FakeS3({m["key"]: jpeg() for m in old})
        self.run_live(s3, FakeSQS(old))
        new = [live_msg("S1", 10)]
        s3.objects[new[0]["key"]] = jpeg()
        self.run_live(s3, FakeSQS(new))
        doc = s3.metrics("S1")
        self.assertEqual(doc["frames_skipped"], 2)
        self.assertEqual(doc["frames_processed"], 2)


class FormatMetricsTest(unittest.TestCase):
    def test_missing_lanes_show_na_not_a_number(self):
        # a missing side used to come out as a ~-4 m offset; now the values it needs are None
        metrics = {"offset_m": None, "left_off_m": None, "right_off_m": 1.9, "road_width_m": None,
                   "lane_valid": False, "departure_text": "Departure: n/a"}
        text = lld.format_metrics(metrics)
        self.assertIn("Offset: n/a", text)
        self.assertIn("L off: n/a | R off: 1.90 m", text)
        self.assertIn("Road width: n/a", text)


if __name__ == "__main__":
    unittest.main()


class H264Test(unittest.TestCase):
    def write_mp4v(self, path):
        w = cv.VideoWriter(str(path), cv.VideoWriter_fourcc(*"mp4v"), 2, (64, 64))
        for i in range(4):
            w.write(np.full((64, 64, 3), i * 60, np.uint8))
        w.release()

    @unittest.skipUnless(lld_aws.shutil.which("ffmpeg") and lld_aws.shutil.which("ffprobe"), "needs ffmpeg")
    def test_converts_to_h264_in_place(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "overlay.mp4"
            self.write_mp4v(path)
            self.assertTrue(lld_aws.to_h264(path))
            codec = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                    "stream=codec_name", "-of", "csv=p=0", str(path)],
                                   capture_output=True, text=True).stdout.strip()
            self.assertEqual(codec, "h264")
            self.assertEqual(sorted(p.name for p in Path(d).iterdir()), ["overlay.mp4"])  # no temp file left

    def test_without_ffmpeg_leaves_the_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "overlay.mp4"
            self.write_mp4v(path)
            before = path.read_bytes()
            with mock.patch.object(lld_aws.shutil, "which", return_value=None):
                self.assertFalse(lld_aws.to_h264(path))
            self.assertEqual(path.read_bytes(), before)


class MetricsPublisherTest(unittest.TestCase):
    def test_values_are_lists_counts_are_summed(self):
        pub = lld_aws.MetricsPublisher(mock.Mock())
        for lag in (0.8, 0.9):
            pub.record_frame({"infer_s": 0.2, "total_s": 0.3, "pi_to_s3_s": 0.4}, lag, {"lane_valid": lag < 0.85})
        pub.add("FramesSkipped", 1)
        pub.flush()
        data = {d["MetricName"]: d for d in pub.cw.put_metric_data.call_args.kwargs["MetricData"]}
        self.assertEqual(pub.cw.put_metric_data.call_args.kwargs["Namespace"], "InitiativeVision")
        self.assertEqual(data["LagSeconds"]["Values"], [0.8, 0.9])
        self.assertEqual(data["InferenceSeconds"]["Unit"], "Seconds")
        self.assertEqual(data["LaneValid"]["Values"], [100.0, 0.0])
        self.assertEqual(data["FramesProcessed"]["Value"], 2)
        self.assertEqual(data["FramesSkipped"]["Value"], 1)
        self.assertNotIn("CaptureToWriteSeconds", data)  # not in this timing: nothing sent

    def test_long_lists_are_split_and_buffer_is_emptied(self):
        pub = lld_aws.MetricsPublisher(mock.Mock())
        for i in range(400):
            pub.add("LagSeconds", i)
        pub.flush()
        sent = pub.cw.put_metric_data.call_args.kwargs["MetricData"]
        self.assertEqual([len(d["Values"]) for d in sent], [150, 150, 100])
        pub.flush()
        self.assertEqual(pub.cw.put_metric_data.call_count, 1)  # nothing new: no call

    def test_failure_never_raises(self):
        cw = mock.Mock()
        cw.put_metric_data.side_effect = RuntimeError("AccessDenied")
        pub = lld_aws.MetricsPublisher(cw)
        pub.add("LagSeconds", 1.0)
        pub.flush()  # printed once, no exception
        pub.add("LagSeconds", 1.0)
        pub.flush()
        self.assertTrue(pub.warned)


class LiveTokenTest(unittest.TestCase):
    def test_token_round_trip_expiry_and_tampering(self):
        from live_view import make_token, token_valid
        tok = make_token("s3cret", hours=1, now=1000)
        self.assertTrue(token_valid("s3cret", tok, now=1000 + 3599))
        self.assertFalse(token_valid("s3cret", tok, now=1000 + 3601))      # expired
        self.assertFalse(token_valid("other", tok, now=1000))               # wrong secret
        expiry, sig = tok.split(".")
        self.assertFalse(token_valid("s3cret", f"{int(expiry) + 99999}.{sig}", now=1000))  # extended expiry
        for bad in (None, "", "garbage", "123.", ".abc"):
            self.assertFalse(token_valid("s3cret", bad, now=1000))

    def test_server_requires_a_token_then_uses_the_cookie(self):
        import urllib.error
        import urllib.request
        from live_view import LiveView, make_token
        view = LiveView(port=0, token_secret="s3cret")
        view.start()
        try:
            base = f"http://127.0.0.1:{view._server.server_address[1]}"
            with self.assertRaises(urllib.error.HTTPError) as err:
                urllib.request.urlopen(base + "/metrics")
            self.assertEqual(err.exception.code, 401)

            class NoRedirect(urllib.request.HTTPRedirectHandler):
                def redirect_request(self, *a, **kw):
                    return None
            opener = urllib.request.build_opener(NoRedirect)
            with self.assertRaises(urllib.error.HTTPError) as err:  # 303 to drop the token from the URL
                opener.open(base + "/?token=" + make_token("s3cret"))
            self.assertEqual(err.exception.code, 303)
            cookie = err.exception.headers["Set-Cookie"].split(";")[0]
            self.assertIn("HttpOnly", err.exception.headers["Set-Cookie"])
            req = urllib.request.Request(base + "/metrics", headers={"Cookie": cookie})
            self.assertEqual(urllib.request.urlopen(req).status, 200)
            self.assertGreater(view.last_request, 0)
        finally:
            view.stop()

    def test_no_secret_means_no_check(self):
        import urllib.request
        from live_view import LiveView
        view = LiveView(port=0).start()
        try:
            self.assertEqual(urllib.request.urlopen(f"http://127.0.0.1:{view._server.server_address[1]}/metrics").status, 200)
        finally:
            view.stop()


class IdleStopTest(unittest.TestCase):
    def test_shuts_down_once_after_idle_from_frames_and_viewers(self):
        live = types.SimpleNamespace(last_request=0.0)
        with mock.patch.object(lld_aws.subprocess, "run") as run:
            idle = lld_aws.IdleStop(600, live, shutdown=["true"])
            idle.last = 1000.0
            live.last_request = 1300.0                  # a viewer kept it alive
            self.assertFalse(idle.check(now=1800))      # 500 s since the viewer
            self.assertTrue(idle.check(now=1950))       # 650 s: shut down
            self.assertFalse(idle.check(now=3000))      # only once
            run.assert_called_once_with(["true"], check=False)


class DownloadTest(unittest.TestCase):
    def test_videos_offered_after_finalize_and_redirected_to_presigned_url(self):
        import urllib.error
        import urllib.request
        from live_view import LiveView
        view = LiveView(port=0, presign=lambda key, name: f"https://s3.example/{key}?sig=1&name={name}").start()
        try:
            base = f"http://127.0.0.1:{view._server.server_address[1]}"
            view.update(np.zeros((8, 8, 3), np.uint8), np.zeros((8, 8, 3), np.uint8), {"lane_valid": False}, session="S1")
            self.assertEqual(json.loads(urllib.request.urlopen(base + "/metrics").read())["downloads"], [])

            output = types.SimpleNamespace(session="S1", finalize=lambda: None,
                                           videos=[{"frames": 3, "overlay": "lld/S1/overlay.mp4", "bev": "lld/S1/bev.mp4"}])
            lld_aws.close_session(output, view)
            downloads = json.loads(urllib.request.urlopen(base + "/metrics").read())["downloads"]
            self.assertEqual([d["key"] for d in downloads], ["lld/S1/overlay.mp4", "lld/S1/bev.mp4"])

            class NoRedirect(urllib.request.HTTPRedirectHandler):
                def redirect_request(self, *a, **kw):
                    return None
            opener = urllib.request.build_opener(NoRedirect)
            with self.assertRaises(urllib.error.HTTPError) as err:
                opener.open(base + "/download?key=lld%2FS1%2Foverlay.mp4")
            self.assertEqual(err.exception.code, 302)
            self.assertTrue(err.exception.headers["Location"].startswith("https://s3.example/lld/S1/overlay.mp4"))
            with self.assertRaises(urllib.error.HTTPError) as err:  # only keys it offered
                opener.open(base + "/download?key=live%2FS1%2Fframe.jpg")
            self.assertEqual(err.exception.code, 404)
        finally:
            view.stop()

    def test_multi_part_sessions_label_each_part(self):
        out = types.SimpleNamespace(videos=[{"overlay": "a1", "bev": "b1"}, {"overlay": "a2", "bev": "b2"}])
        self.assertEqual([l for l, _ in lld_aws.video_downloads(out)],
                         ["camera overlay video (part 1)", "bird's-eye video (part 1)",
                          "camera overlay video (part 2)", "bird's-eye video (part 2)"])
