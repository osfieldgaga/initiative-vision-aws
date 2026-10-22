# Live pipeline setup: Pi → S3 → Lambda → SQS → EC2

The Pi only captures and uploads; all processing happens in the cloud.

```
Pi  upload_live_frame.py ── PUT live/latest_frame.jpg  (metadata: session-start, frame-time, frame-seq)
        │  S3 event, exact key only, carries the versionId of that upload
        ▼
Lambda  on-live-frame (src/lambda/on_live_frame.py)
        ├─ copy that exact version → live/<session-start>/<frame-time>.jpg   (nothing is lost)
        └─ send {bucket, key, session_start, frame_time, frame_seq} → SQS FIFO queue
        ▼
EC2  lld_aws.py live ── receive → lld.process_frame → live view (SSH tunnel) + per-session outputs
                         lld/<session-start>/overlay.mp4, bev.mp4, metrics.json
```

Region `us-east-1` throughout. Bucket `initiative-vision-1` (versioning already on). Replace
`<account-id>` with your 12-digit account id (top-right of the console).

## 1. Lifecycle rule for old versions (S3 console)

With versioning on, every overwrite of `live/latest_frame.jpg` keeps the old one as a
"noncurrent version". The Lambda copies each into `live/<session>/` within a second, so the old
versions are just clutter after that.

S3 → bucket → **Management** → **Create lifecycle rule**
- Name: `expire-live-latest-versions`
- Scope: **Limit the scope using filters** → Prefix: `live/latest_frame.jpg`
- Action: **Permanently delete noncurrent versions of objects** → Days after objects become noncurrent: `1`

## 2. SQS queues (SQS console)

1. **Dead-letter queue first**: Create queue → **FIFO** → name `lld-frames-dlq.fifo` → defaults → Create.
   Frames that fail repeatedly land here instead of blocking the session forever.
2. **Main queue**: Create queue → **FIFO** → name `lld-frames.fifo`
   - Visibility timeout: `120 seconds` (matches `VISIBILITY_S` in lld_aws.py)
   - Message retention: `4 days` (default; frames wait this long if the EC2 instance is off)
   - Content-based deduplication: **off** (the Lambda sends explicit dedup ids)
   - Dead-letter queue: **Enabled** → `lld-frames-dlq.fifo`, Maximum receives `5`
3. Copy the queue **URL** (e.g. `https://sqs.us-east-1.amazonaws.com/<account-id>/lld-frames.fifo`).

FIFO + the Lambda's `MessageGroupId=session` means frames come out in capture order per session,
and `MessageDeduplicationId=<archive key>` means a retried Lambda doesn't queue a frame twice.

## 3. Lambda `on-live-frame` (Lambda console)

1. Create function → Author from scratch → name `on-live-frame`, runtime **Python 3.12**, arm64.
2. Paste `src/lambda/on_live_frame.py` into the editor's `lambda_function.py` → **Deploy**. The console
   expects a function named `lambda_handler`, ours is `handler`: **Runtime settings → Edit → Handler**
   = `lambda_function.handler`.
3. **Configuration → Environment variables**: `QUEUE_URL` = the queue URL from step 2.
4. **Configuration → General**: timeout `10 s` (default 3 s is tight for head + copy + send).
5. **Configuration → Permissions** → role name → Add permissions → Create inline policy → JSON:
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Sid": "ReadLiveUploads",
         "Effect": "Allow",
         "Action": ["s3:GetObject", "s3:GetObjectVersion"],
         "Resource": "arn:aws:s3:::initiative-vision-1/live/latest_frame.jpg"
       },
       {
         "Sid": "WriteArchive",
         "Effect": "Allow",
         "Action": ["s3:PutObject"],
         "Resource": "arn:aws:s3:::initiative-vision-1/live/*"
       },
       {
         "Sid": "QueueFrames",
         "Effect": "Allow",
         "Action": ["sqs:SendMessage"],
         "Resource": "arn:aws:sqs:us-east-1:<account-id>:lld-frames.fifo"
       }
     ]
   }
   ```
   `s3:GetObjectVersion` is the one people miss: reading or copying a *specific version* needs it,
   plain `GetObject` isn't enough.
6. **Add trigger** → S3 → bucket `initiative-vision-1` → Event types: **All object create events**
   → Prefix: `live/latest_frame.jpg` (the full key -- so the archive copies under `live/<session>/`
   don't re-trigger it; the code also guards against this) → acknowledge the recursion warning → Add.

**Test it:** upload anything to `live/latest_frame.jpg` (console or `aws s3 cp x.jpg s3://initiative-vision-1/live/latest_frame.jpg`).
Within seconds: CloudWatch → `/aws/lambda/on-live-frame` shows `archived ... -> live/unknown-session/...`
(no Pi metadata on a manual upload -- expected), and the SQS console's **Send and receive messages →
Poll for messages** shows the message. Delete that test object/message afterwards.

## 4. EC2 instance role

IAM → Roles → the role attached to the instance (or create one: trusted entity EC2) → inline policy:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ListBucket",
      "Effect": "Allow",
      "Action": ["s3:ListBucket"],
      "Resource": "arn:aws:s3:::initiative-vision-1"
    },
    {
      "Sid": "ReadFrames",
      "Effect": "Allow",
      "Action": ["s3:GetObject"],
      "Resource": ["arn:aws:s3:::initiative-vision-1/live/*", "arn:aws:s3:::initiative-vision-1/sessions/*"]
    },
    {
      "Sid": "Results",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject"],
      "Resource": "arn:aws:s3:::initiative-vision-1/lld/*"
    },
    {
      "Sid": "ConsumeQueue",
      "Effect": "Allow",
      "Action": ["sqs:ReceiveMessage", "sqs:DeleteMessage", "sqs:GetQueueAttributes"],
      "Resource": "arn:aws:sqs:us-east-1:<account-id>:lld-frames.fifo"
    }
  ]
}
```
`ListBucket` matters even though lld_aws only lists in replay mode: without it, reading a key that
doesn't exist returns AccessDenied instead of NoSuchKey, and the live mode's "is there already a
metrics.json for this session?" check can't tell the difference.

## 5. Run it on the instance

The model runs on OpenCV's DNN module (`cv.dnn`) from an ONNX file, so the instance doesn't need
torch. Make the ONNX once on a machine that has torch:
```
python src/scripts/export_onnx.py                                   # -> data/weights/yolopv2.onnx
python src/scripts/compare_onnx.py <some frames dir>                # should report IoU 1.0000
```
Sign reading (`signs.py`) needs OpenCV's CRNN text recognizer (33 MB):
```
mkdir -p data/weights/ocr
curl -L -o data/weights/ocr/text_recognition_CRNN_EN_2021sep.onnx \
  https://huggingface.co/opencv/text_recognition_crnn/resolve/main/text_recognition_CRNN_EN_2021sep.onnx
```
Copy over `lld.py`, `signs.py`, `lld_aws.py`, `live_view.py`, `data/weights/yolopv2.onnx` and
`data/weights/ocr/`. `LLD_SIGNS=0` turns sign reading off. On the instance,
check that COOL's OpenCV is 5.x and has the DNN module:
`python -c "import cv2; print(cv2.__version__); cv2.dnn.readNetFromONNX('data/weights/yolopv2.onnx')"`.
Then in COOL's venv (don't `pip install opencv-python` there -- it would shadow COOL's OpenCV):
```
pip install boto3
LLD_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/<account-id>/lld-frames.fifo python lld_aws.py live
```
To keep it running after you log out and across reboots, a systemd unit
(`/etc/systemd/system/lld.service`, fix the paths):
```ini
[Unit]
Description=lane pipeline, live mode
After=network-online.target

[Service]
WorkingDirectory=/home/ec2-user/lld
Environment=LLD_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/<account-id>/lld-frames.fifo
ExecStart=/path/to/cool/venv/bin/python lld_aws.py live
Restart=on-failure

[Install]
WantedBy=multi-user.target
```
`sudo systemctl enable --now lld`, logs with `journalctl -u lld -f`. A restart mid-session is safe:
lld_aws resumes the session's metrics and starts a new video part.

**Watch:** `ssh -L 8080:localhost:8080 ec2-user@<instance>` → http://localhost:8080. The `lag` row
is capture-on-Pi → processed-on-EC2, and the per-frame breakdown (download / inference / post-proc /
outputs) shows where the time goes. Inference (TorchScript) was ~4 s/frame on 1 vCPU, ~1 s on 4 -- use at least a
`c8g.xlarge`; a `.medium` can't keep up with the Pi. `LLD_WEIGHTS=data/weights/yolopv2.pt` switches back
to TorchScript (needs `pip install torch`) for comparisons.

If processing does fall behind, frames older than `LLD_MAX_LAG_S` (default 10 s) are skipped while
newer ones are waiting, so the live view stays current. They're still archived in `live/<session>/`;
`python lld_aws.py replay live/<session>/` processes every frame. `LLD_MAX_LAG_S=0` never skips.

## 6. Pi

`upload_live_frame.py` now tags every upload with `session-start` (the moment the uploader started),
`frame-time` and `frame-seq`, and survives network drops. Start it alongside `start.py`; each start
is a new session.

## Rough running cost at ~2 frames/s

| Piece | Per hour of driving |
|---|---|
| Lambda (7,200 invocations) | free tier (1M/month) |
| S3 PUT + COPY (~14,400 requests) | ~$0.07 |
| SQS FIFO (send + receive + delete) | ~$0.01 |
| Archived frames (~1 GB) | ~$0.023 per month kept |
| EC2 instance | while it's running -- the dominant cost; stop it when not driving |
