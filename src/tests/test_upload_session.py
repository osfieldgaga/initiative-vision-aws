"""Tests for phase2/scripts/upload_session.py.

No real AWS needed: the S3 client is a plain mock, so these run anywhere,
including a machine with no AWS credentials at all (this desktop, right now).
Loaded by file path since upload_session.py is a standalone script, not part
of a package -- same technique used for tools/make_video_standalone.py.
"""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "upload_session.py"
_spec = importlib.util.spec_from_file_location("upload_session", _SCRIPT)
upload_session_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(upload_session_mod)


class TmpCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def make_session(self):
        d = self.base / "20260922T000000Z-abcd"
        (d / "frames").mkdir(parents=True)
        (d / "session.json").write_text('{"status": "complete"}')
        (d / "frames.csv").write_text("seq,filename,timestamp_ns\n0,frames/00000000_1.jpg,1\n")
        (d / "frames" / "00000000_1.jpg").write_bytes(b"\xff\xd8fake\xff\xd9")
        (d / "frames" / "00000001_2.jpg").write_bytes(b"\xff\xd8fake2\xff\xd9")
        return d


class IterFilesTest(TmpCase):
    def test_finds_every_file_with_correct_relative_paths(self):
        session = self.make_session()
        found = {str(rel) for _, rel in upload_session_mod.iter_files(session)}
        self.assertEqual(found, {
            "session.json", "frames.csv",
            "frames/00000000_1.jpg", "frames/00000001_2.jpg",
        })

    def test_empty_directory_yields_nothing(self):
        empty = self.base / "empty"
        empty.mkdir()
        self.assertEqual(list(upload_session_mod.iter_files(empty)), [])

    def test_does_not_yield_directories_themselves(self):
        session = self.make_session()
        for local_path, _ in upload_session_mod.iter_files(session):
            self.assertTrue(local_path.is_file())


class UploadSessionTest(TmpCase):
    def test_uploads_every_file_with_prefixed_keys(self):
        session = self.make_session()
        s3 = mock.Mock()
        result = upload_session_mod.upload_session(session, "my-bucket", "sessions/abcd", s3)

        self.assertEqual(result["uploaded"], 4)
        self.assertGreater(result["bytes"], 0)
        self.assertEqual(s3.upload_file.call_count, 4)

        keys = {call.args[2] for call in s3.upload_file.call_args_list}
        self.assertEqual(keys, {
            "sessions/abcd/session.json",
            "sessions/abcd/frames.csv",
            "sessions/abcd/frames/00000000_1.jpg",
            "sessions/abcd/frames/00000001_2.jpg",
        })
        # every call's bucket argument (2nd positional) is the one passed in
        for call in s3.upload_file.call_args_list:
            self.assertEqual(call.args[1], "my-bucket")

    def test_bytes_total_matches_real_file_sizes(self):
        session = self.make_session()
        expected = sum(p.stat().st_size for p, _ in upload_session_mod.iter_files(session))
        result = upload_session_mod.upload_session(session, "b", "p", mock.Mock())
        self.assertEqual(result["bytes"], expected)


class MainTest(TmpCase):
    def test_rejects_a_nonexistent_session_dir(self):
        with mock.patch.object(sys, "argv", ["prog"]):
            with self.assertRaises(SystemExit):
                upload_session_mod.main([str(self.base / "nope"), "bucket"])

    def test_default_prefix_uses_the_session_folder_name(self):
        session = self.make_session()
        with mock.patch("boto3.client") as client_factory:
            fake_s3 = mock.Mock()
            client_factory.return_value = fake_s3
            code = upload_session_mod.main([str(session), "my-bucket"])
        self.assertEqual(code, 0)
        keys = {call.args[2] for call in fake_s3.upload_file.call_args_list}
        self.assertTrue(all(k.startswith(f"sessions/{session.name}/") for k in keys), keys)

    def test_missing_boto3_gives_a_clear_message_not_a_traceback(self):
        session = self.make_session()
        real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

        def blocked_import(name, *a, **kw):
            if name == "boto3":
                raise ImportError("No module named 'boto3'")
            return real_import(name, *a, **kw)

        with mock.patch("builtins.__import__", side_effect=blocked_import):
            code = upload_session_mod.main([str(session), "my-bucket"])
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
