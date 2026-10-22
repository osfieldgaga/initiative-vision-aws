"""Tests for src/scripts/virtual_pi.py: frame pacing and the metadata the Lambda reads."""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "lambda"))
sys.path.insert(0, str(ROOT / "src" / "scripts"))
import on_live_frame  # noqa: E402
import virtual_pi  # noqa: E402


class VirtualPiTest(unittest.TestCase):
    def test_load_frames_relative_times(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "frames.csv").write_text(
                "seq,filename,timestamp_ns\n0,frames/a.jpg,1000000000\n1,frames/b.jpg,1500000000\n")
            frames = virtual_pi.load_frames(d)
        self.assertEqual([(s, p.name, t) for s, p, t in frames], [(0, "a.jpg", 0.0), (1, "b.jpg", 0.5)])

    def test_index_at_picks_latest_frame_not_after_t(self):
        frames = [(k, None, t) for k, t in enumerate([0.0, 0.1, 0.2, 0.5])]
        self.assertEqual(virtual_pi.index_at(frames, 0.0), 0)
        self.assertEqual(virtual_pi.index_at(frames, 0.25), 2)
        self.assertEqual(virtual_pi.index_at(frames, 9.0), 3)
        self.assertEqual(virtual_pi.index_at(frames, 0.3, start=1), 2)

    def test_metadata_round_trips_through_the_lambda(self):
        meta = virtual_pi.upload_metadata(virtual_pi.session_start_name(1792505112.5), 1792505112.512, 42)
        self.assertEqual(on_live_frame.frame_info(meta, None), ("20261020T140512Z", 1792505112.512, 42))

    def test_direct_mode_puts_archive_key_and_queues_lambda_message(self):
        from unittest import mock
        up = virtual_pi.Uploader("20261020T140000Z", "b", enabled=False, queue_url="https://q/x.fifo")
        up.enabled, up.s3, up.sqs = True, mock.Mock(), mock.Mock()
        up._send(b"jpg", 7, 1792505112.512, 1792505112.6)

        put = up.s3.put_object.call_args.kwargs
        self.assertEqual(put["Key"], "live/20261020T140000Z/20261020T140512.512Z.jpg")
        sent = up.sqs.send_message.call_args.kwargs
        self.assertEqual(sent["QueueUrl"], "https://q/x.fifo")
        self.assertEqual(sent["MessageGroupId"], "20261020T140000Z")
        body = __import__("json").loads(sent["MessageBody"])
        self.assertEqual({k: body[k] for k in ("bucket", "key", "session_start", "frame_time", "frame_seq")},
                         {"bucket": "b", "key": put["Key"], "session_start": "20261020T140000Z",
                          "frame_time": 1792505112.512, "frame_seq": 7})
        self.assertLessEqual(body["uploaded_at"], body["queued_at"])


if __name__ == "__main__":
    unittest.main()
