"""Tests for phase2/lambda/on_upload.py -- no real AWS needed.

Loaded by file path (it's a Lambda handler file, not part of a package).
"""

import importlib.util
import json
import unittest
from pathlib import Path
from unittest import mock

_SCRIPT = Path(__file__).resolve().parent.parent / "lambda" / "on_upload.py"
_spec = importlib.util.spec_from_file_location("on_upload", _SCRIPT)
on_upload = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(on_upload)

# A real S3 -> Lambda event notification looks like this (trimmed to what
# parse_event actually reads); used as the fixture for every test below.
SAMPLE_EVENT = {
    "Records": [
        {
            "eventSource": "aws:s3",
            "s3": {
                "bucket": {"name": "my-bucket"},
                "object": {"key": "sessions/abcd/frames/00000000_1.jpg", "size": 12345},
            },
        }
    ]
}


class ParseEventTest(unittest.TestCase):
    def test_extracts_bucket_key_size(self):
        got = on_upload.parse_event(SAMPLE_EVENT)
        self.assertEqual(got, [("my-bucket", "sessions/abcd/frames/00000000_1.jpg", 12345)])

    def test_url_decodes_the_key(self):
        # S3 event keys are URL-encoded; a space becomes '+', which must be
        # undone or the key won't match the real object.
        event = {"Records": [{"s3": {"bucket": {"name": "b"}, "object": {"key": "a+b%2Fc.jpg", "size": 1}}}]}
        got = on_upload.parse_event(event)
        self.assertEqual(got[0][1], "a b/c.jpg")

    def test_handles_multiple_records_in_one_event(self):
        event = {"Records": [SAMPLE_EVENT["Records"][0], SAMPLE_EVENT["Records"][0]]}
        self.assertEqual(len(on_upload.parse_event(event)), 2)

    def test_no_records_gives_empty_list(self):
        self.assertEqual(on_upload.parse_event({"Records": []}), [])
        self.assertEqual(on_upload.parse_event({}), [])

    def test_a_malformed_record_is_skipped_not_crashed_on(self):
        event = {"Records": [{"s3": {"bucket": {}, "object": {}}}]}
        self.assertEqual(on_upload.parse_event(event), [])


class ProcessedKeyTest(unittest.TestCase):
    def test_prefixes_with_processed(self):
        self.assertEqual(on_upload.processed_key("sessions/x/frames/a.jpg"), "processed/sessions/x/frames/a.jpg")


class HandlerTest(unittest.TestCase):
    def test_copies_the_uploaded_object_to_the_processed_prefix(self):
        with mock.patch.object(on_upload, "s3") as fake_s3:
            result = on_upload.handler(SAMPLE_EVENT, None)
        fake_s3.copy_object.assert_called_once_with(
            Bucket="my-bucket",
            CopySource={"Bucket": "my-bucket", "Key": "sessions/abcd/frames/00000000_1.jpg"},
            Key="processed/sessions/abcd/frames/00000000_1.jpg",
        )
        self.assertEqual(result["statusCode"], 200)

    def test_skips_objects_already_under_processed_to_avoid_a_trigger_loop(self):
        event = {"Records": [{"s3": {"bucket": {"name": "b"}, "object": {"key": "processed/x.jpg", "size": 1}}}]}
        with mock.patch.object(on_upload, "s3") as fake_s3:
            on_upload.handler(event, None)
        fake_s3.copy_object.assert_not_called()

    def test_multiple_uploads_in_one_event_are_all_processed(self):
        event = {"Records": [SAMPLE_EVENT["Records"][0], SAMPLE_EVENT["Records"][0]]}
        with mock.patch.object(on_upload, "s3") as fake_s3:
            result = on_upload.handler(event, None)
        self.assertEqual(fake_s3.copy_object.call_count, 2)
        body = json.loads(result["body"])
        self.assertEqual(len(body["processed"]), 2)


if __name__ == "__main__":
    unittest.main()
