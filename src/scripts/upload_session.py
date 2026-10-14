#!/usr/bin/env python3
"""Upload a Phase 1 session directory into S3, keeping the same relative
paths as keys (frames/00000000_....jpg -> sessions/<id>/frames/00000000_....jpg).

This is deliberately simple: no multipart tuning, no parallelism, no retry
logic. It's the first S3 script of Phase 2 sprint 1, meant to be read and
understood, not a production uploader. It becomes the real "Pi logs -> S3
raw store" step later, once it's proven itself on real data.

Setup (once):
    pip install boto3
    aws configure          # asks for an access key, secret key, region
                            # (see the sprint 1 walkthrough for where to get these)

Usage:
    python3 upload_session.py <local_session_dir> <bucket_name>
    python3 upload_session.py sessions/20260922T212932Z-d24c my-initiative-vision-bucket

What it uploads: every file under the session directory (session.json,
frames.csv, system.csv if present, and every file in frames/), skipping
nothing, so the S3 copy is a faithful mirror of what verify_session already
validated on the Pi/desktop.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def iter_files(session_dir: Path):
    """Every regular file under session_dir, yielded with its path relative
    to session_dir (this relative path becomes the tail of the S3 key)."""
    for path in sorted(session_dir.rglob("*")):
        if path.is_file():
            yield path, path.relative_to(session_dir)


def upload_session(session_dir: Path, bucket: str, prefix: str, s3_client, progress_every: int = 200) -> dict:
    """Upload every file under session_dir to s3://bucket/prefix/<relative path>.

    Returns {"uploaded": n, "bytes": total_bytes} so a caller (or a test)
    can check something real happened, not just that no exception was raised.
    """
    uploaded = 0
    total_bytes = 0
    for local_path, rel_path in iter_files(session_dir):
        key = f"{prefix}/{rel_path.as_posix()}"
        s3_client.upload_file(str(local_path), bucket, key)
        uploaded += 1
        total_bytes += local_path.stat().st_size
        if uploaded % progress_every == 0:
            print(f"  {uploaded} files uploaded so far...", flush=True)
    return {"uploaded": uploaded, "bytes": total_bytes}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session_dir", help="local Phase 1 session directory, e.g. sessions/20260922T.../")
    parser.add_argument("bucket", help="destination S3 bucket name (must already exist)")
    parser.add_argument("--prefix", default=None,
                        help="S3 key prefix; default is 'sessions/<session folder name>'")
    args = parser.parse_args(argv)

    session_dir = Path(args.session_dir)
    if not session_dir.is_dir():
        parser.error(f"{session_dir} is not a directory")
    prefix = args.prefix or f"sessions/{session_dir.name}"

    try:
        import boto3
    except ImportError:
        print("boto3 is not installed. Run: pip install boto3", file=sys.stderr)
        return 2

    s3 = boto3.client("s3")
    print(f"uploading {session_dir} -> s3://{args.bucket}/{prefix}/")
    result = upload_session(session_dir, args.bucket, prefix, s3)
    mb = result["bytes"] / 1e6
    print(f"done: {result['uploaded']} files, {mb:.1f} MB")
    print(f"verify in the console, or: aws s3 ls s3://{args.bucket}/{prefix}/ --recursive | head")
    return 0


if __name__ == "__main__":
    sys.exit(main())
