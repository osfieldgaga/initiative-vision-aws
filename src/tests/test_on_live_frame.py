"""Tests for src/lambda/on_live_frame.py -- no real AWS needed.

Loaded by file path (it's a Lambda handler file, not part of a package).
"""

import importlib.util
import json
import os
import unittest
from pathlib import Path
from unittest import mock

_SCRIPT = Path(__file__).resolve().parent.parent / "lambda" / "on_live_frame.py"
_spec = importlib.util.spec_from_file_location("on_live_frame", _SCRIPT)
on_live_frame = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(on_live_frame)

# A real S3 -> Lambda event for an overwrite of live/latest_frame.jpg on a versioned bucket
# (trimmed to what the handler reads).
SAMPLE_EVENT = {
    "Records": [
        {
            "eventSource": "aws:s3",
            "eventTime": "2026-10-20T14:05:13.100Z",
            "s3": {
                "bucket": {"name": "initiative-vision-1"},
                "object": {"key": "live/latest_frame.jpg", "size": 123456, "versionId": "v123"},
            },
        }
    ]
}

PI_METADATA = {"session-start": "20261020T140512Z", "frame-time": "1792505112.512", "frame-seq": "7"}


class ParseEventTest(unittest.TestCase):
    def test_extracts_bucket_key_version_and_time(self):
        got = on_live_frame.parse_event(SAMPLE_EVENT)
        self.assertEqual(got, [{"bucket": "initiative-vision-1", "key": "live/latest_frame.jpg",
                                "version_id": "v123", "event_time": "2026-10-20T14:05:13.100Z"}])

    def test_version_id_is_none_without_versioning(self):
        event = {"Records": [{"s3": {"bucket": {"name": "b"}, "object": {"key": "live/latest_frame.jpg"}}}]}
        self.assertIsNone(on_live_frame.parse_event(event)[0]["version_id"])

    def test_url_decodes_the_key(self):
        event = {"Records": [{"s3": {"bucket": {"name": "b"}, "object": {"key": "a+b%2Fc.jpg"}}}]}
        self.assertEqual(on_live_frame.parse_event(event)[0]["key"], "a b/c.jpg")

    def test_malformed_and_empty_events(self):
        self.assertEqual(on_live_frame.parse_event({}), [])
        self.assertEqual(on_live_frame.parse_event({"Records": [{"s3": {"bucket": {}, "object": {}}}]}), [])


class NamingTest(unittest.TestCase):
    def test_frame_name_is_utc_with_millis(self):
        self.assertEqual(on_live_frame.frame_name(1792505112.512), "20261020T140512.512Z.jpg")

    def test_frame_names_sort_chronologically(self):
        times = [1792505112.512, 1792505112.9, 1792505113.05, 1792505199.0]
        names = [on_live_frame.frame_name(t) for t in times]
        self.assertEqual(names, sorted(names))

    def test_archive_key(self):
        self.assertEqual(on_live_frame.archive_key("20261020T140512Z", 1792505112.512),
                         "live/20261020T140512Z/20261020T140512.512Z.jpg")

    def test_event_time_parsing(self):
        self.assertAlmostEqual(on_live_frame.event_time_to_epoch("2026-10-20T14:05:12.512Z"), 1792505112.512, places=3)
        self.assertAlmostEqual(on_live_frame.event_time_to_epoch("2026-10-20T14:05:12Z"), 1792505112.0, places=3)


class FrameInfoTest(unittest.TestCase):
    def test_reads_pi_metadata(self):
        self.assertEqual(on_live_frame.frame_info(PI_METADATA, None), ("20261020T140512Z", 1792505112.512, 7))

    def test_falls_back_without_metadata(self):
        session, t, seq = on_live_frame.frame_info({}, "2026-10-20T14:05:13.100Z")
        self.assertEqual(session, on_live_frame.UNKNOWN_SESSION)
        self.assertAlmostEqual(t, 1792505113.1, places=3)
        self.assertIsNone(seq)


class QueueMessageTest(unittest.TestCase):
    BODY = {"key": "live/s/a.jpg", "session_start": "s"}

    def test_fifo_queue_gets_group_and_dedup_ids(self):
        kw = on_live_frame.queue_message("https://sqs/123/q.fifo", self.BODY)
        self.assertEqual(kw["MessageGroupId"], "s")
        self.assertEqual(kw["MessageDeduplicationId"], "live/s/a.jpg")

    def test_standard_queue_does_not(self):
        kw = on_live_frame.queue_message("https://sqs/123/q", self.BODY)
        self.assertNotIn("MessageGroupId", kw)
        self.assertEqual(json.loads(kw["MessageBody"]), self.BODY)


class HandlerTest(unittest.TestCase):
    def setUp(self):
        self.s3 = mock.Mock()
        self.s3.head_object.return_value = {"Metadata": PI_METADATA}
        self.sqs = mock.Mock()
        patches = [mock.patch.object(on_live_frame, "s3", self.s3),
                   mock.patch.object(on_live_frame, "sqs", self.sqs),
                   mock.patch.dict(os.environ, {"QUEUE_URL": "https://sqs/123/q.fifo"})]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_copies_the_exact_version_that_fired_the_event(self):
        on_live_frame.handler(SAMPLE_EVENT, None)
        self.s3.head_object.assert_called_once_with(Bucket="initiative-vision-1", Key="live/latest_frame.jpg",
                                                    VersionId="v123")
        self.s3.copy_object.assert_called_once_with(
            Bucket="initiative-vision-1",
            Key="live/20261020T140512Z/20261020T140512.512Z.jpg",
            CopySource={"Bucket": "initiative-vision-1", "Key": "live/latest_frame.jpg", "VersionId": "v123"},
        )

    def test_queues_the_archived_key(self):
        on_live_frame.handler(SAMPLE_EVENT, None)
        kw = self.sqs.send_message.call_args.kwargs
        body = json.loads(kw["MessageBody"])
        self.assertEqual(body, {"bucket": "initiative-vision-1", "key": "live/20261020T140512Z/20261020T140512.512Z.jpg",
                                "session_start": "20261020T140512Z", "frame_time": 1792505112.512, "frame_seq": 7})
        self.assertEqual(kw["MessageGroupId"], "20261020T140512Z")

    def test_ignores_its_own_archive_copies(self):
        event = {"Records": [{"s3": {"bucket": {"name": "b"},
                                     "object": {"key": "live/20261020T140512Z/x.jpg", "versionId": "v"}}}]}
        on_live_frame.handler(event, None)
        self.s3.copy_object.assert_not_called()
        self.sqs.send_message.assert_not_called()

    def test_without_version_id_copies_current_object(self):
        event = {"Records": [{"s3": {"bucket": {"name": "b"}, "object": {"key": "live/latest_frame.jpg"}}}]}
        on_live_frame.handler(event, None)
        self.assertNotIn("VersionId", self.s3.copy_object.call_args.kwargs["CopySource"])

    def test_no_queue_url_archives_only(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            on_live_frame.handler(SAMPLE_EVENT, None)
        self.s3.copy_object.assert_called_once()
        self.sqs.send_message.assert_not_called()


if __name__ == "__main__":
    unittest.main()
