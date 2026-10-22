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


class CanTrackTest(unittest.TestCase):
    CSV = ("timestamp_ns,seq,message,VehicleSpeed,EngineRPM,SteeringWheelAngle,AcceleratorPedal2,BrakePedalPos,PRNDL2,TurnSignals\n"
           "1000000000,1,a,,800,,,,P,0\n"
           "2000000000,2,b,10.5,820,-12.5,127,104,D,2\n")

    def test_lookup_by_drive_time_with_scaling(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "decoded.csv").write_text(self.CSV)
            can = virtual_pi.CanTrack(Path(d) / "decoded.csv", t0_ns=1_000_000_000)
        self.assertTrue(can.available)
        self.assertTrue(all(v is None for v in can.at(-0.5).values()))  # before the first row
        first = can.at(0.5)
        self.assertEqual((first["rpm"], first["gear"], first["speed_mph"]), (800.0, "P", None))
        second = can.at(5.0)
        self.assertEqual((second["speed_mph"], second["steering_deg"], second["gear"], second["blinker"]),
                         (10.5, -12.5, "D", 2.0))
        self.assertAlmostEqual(second["accel_pct"], 50.0)   # 127 / 254
        self.assertAlmostEqual(second["brake_pct"], 50.0)   # 104 / 0xd0

    def test_missing_file_and_empty_values_still_draw(self):
        can = virtual_pi.CanTrack("/nonexistent/decoded.csv", 0)
        self.assertFalse(can.available)
        for values, available in ((can.at(1.0), True), ({}, False)):
            panel = virtual_pi.draw_can_panel(636, values, available)
            self.assertEqual(panel.shape, (636, virtual_pi.PANEL_W, 3))
        self.assertEqual(virtual_pi.draw_timeline(940, 30.0, 120.0).shape, (virtual_pi.TIMELINE_H, 940, 3))

    def test_blinker_flashes_on_its_side_only(self):
        green = (60, 220, 60)
        def lit(blinker, t, x):  # is the arrow around column x drawn green?
            p = virtual_pi.draw_can_panel(636, {"blinker": blinker, "speed_mph": None, "rpm": None,
                                                "steering_deg": None, "accel_pct": None, "brake_pct": None,
                                                "gear": None}, True, t)
            return bool((p[10:35, x - 12:x + 12] == green).all(axis=2).any())
        half_period = 1 / (2 * virtual_pi.BLINK_HZ)
        self.assertTrue(lit(1.0, 0.0, 212))                    # left on
        self.assertFalse(lit(1.0, 0.0, 272))                   # right stays dark
        self.assertFalse(lit(1.0, half_period * 1.01, 212))    # ...and it flashes
        self.assertTrue(lit(2.0, 0.0, 272))
        self.assertFalse(lit(0.0, 0.0, 212) or lit(0.0, 0.0, 272))


if __name__ == "__main__":
    unittest.main()
