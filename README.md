# Initiative Vision: live lane, vehicle and sign perception from a car, in the cloud

A Raspberry Pi camera in a real car uploads about one frame per second over cellular. On AWS,
OpenCV 5 with COOL on Graviton finds the lane lines, the car's position in the lane, vehicles,
stop signs and speed limits, and shows them on a live page, about 1.2 to 1.4 s after the camera
captured the frame (measured from the Pi, 2026-10-22).

Entry for the [OpenCV AI Competition 2026](https://opencv26.devpost.com).
Report: *link to come*. Video: *link to come*. Live demo: [www.initiativelabs.cc/ivision-demo](https://www.initiativelabs.cc/ivision-demo) (password in the submission).

> Not a safety system: it never controls the vehicle, and the driver must not rely on it.

## For judges: run the car yourself

You don't need the car or an AWS account. The virtual Pi replays a real drive on your computer and
uploads it through the real AWS pipeline, exactly like the car does; you watch the results live.

1. **Get the code and the drive** (503 MB: 7.7 minutes of driving, privacy-processed, see `PRIVACY.md` inside):
   ```bash
   git clone https://github.com/osfieldgaga/initiative-vision-aws.git && cd initiative-vision-aws
   python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt     # Python 3.12 or 3.13
   curl -L -O https://github.com/osfieldgaga/initiative-vision-aws/releases/download/judge-data/ivision-drive-20261022.zip
   python3 -m zipfile -e ivision-drive-20261022.zip .
   ```
2. **Open the results page**, [www.initiativelabs.cc/ivision-demo](https://www.initiativelabs.cc/ivision-demo),
   with the password from the submission's judging instructions, and press **Start the demo**. The cloud
   instance takes 1-2 minutes to boot (it stops itself after 20 idle minutes).
3. **Start the virtual Pi** (it asks for the same password and gets one-hour upload credentials from the site):
   ```bash
   .venv/bin/python src/scripts/virtual_pi.py 20261022T164127Z-084b --demo
   ```
   A window shows the camera next to the car's CAN data (speed, RPM, gear, steering, pedals, blinkers).
   Keys: space pause, a / d or arrows seek 10 s, click the timeline to jump, q quit. Put it next to the
   results page: the gap between them is the end-to-end lag.
4. **When you close the virtual Pi**, the session's videos become downloadable on the results page about
   30 s later.

`--no-upload` plays the drive locally without AWS. On Windows, use `.venv\Scripts\python` instead of `.venv/bin/python`.

## How it works

![Architecture: the Pi uploads and queues each frame; OpenCV 5 with COOL on Graviton does the vision](docs/architecture.png)

```
Pi (in the car)  upload_live_frame.py: 640x640 JPEG, 1 frame/s
   ├─ PUT  s3://<bucket>/live/<session>/<frame time>.jpg
   └─ send SQS message ─────────────▶ SQS FIFO ──▶ EC2 c8g (Graviton4), COOL OpenCV 5.1
                                                    lld_aws.py live
                                                      undistort → YOLOPv2 on cv.dnn (lanes, drivable area,
                                                      vehicles) → bird's-eye view → lane offsets, departure
                                                      → sign reader (colour, shape, CRNN OCR)
                                                      ├─ live page (live_view.py)
                                                      └─ s3://<bucket>/lld/<session>/ overlay.mp4, bev.mp4, metrics.json
```

Everything after the upload runs on OpenCV 5; PyTorch isn't needed at inference time. A Lambda
path (S3 event → `src/lambda/on_live_frame.py` → SQS) remains as a fallback. Full setup with IAM,
queues and costs: [`phase2/live-pipeline-setup.md`](phase2/live-pipeline-setup.md).

## Repository layout

| Path | What |
|---|---|
| `lld.py` | The perception pipeline: `undistort_frame`, `process_frame`, YOLOPv2 decoding, bird's-eye view, metrics |
| `signs.py` | Stop-sign and speed-limit reader: HSV/edges → shape → CRNN OCR scored with lexicon-constrained CTC |
| `lld_aws.py` | EC2 service: `live` (consume SQS) and `replay` (process an archived session) |
| `live_view.py` | Live web page (stream, metrics, objects, signs, lag breakdown) |
| `src/lambda/` | `on_live_frame.py` (fallback live path), `on_upload.py` (whole-session uploads) |
| `src/scripts/` | `export_onnx.py`, `compare_onnx.py`, `dnn_bench.py`, `dnn_check.py`, `virtual_pi.py`, `upload_session.py` |
| `src/tests/` | Unit tests |
| `phase2/` | Setup guide, compute benchmark, competition checklist |
| `archive/` | Early experiments, kept for history |

The Pi-side recorder and uploader (camera + CAN logging on a Raspberry Pi 3B) live in a separate
repository, `initiative-vision`.

## Quickstart

### 1. Model weights (not in git)

```bash
mkdir -p data/weights/ocr
# YOLOPv2 (lanes, drivable area, vehicles), from the authors' release
curl -L -o data/weights/yolopv2.pt https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt
# OpenCV model zoo CRNN text recognizer, for signs.py
curl -L -o data/weights/ocr/text_recognition_CRNN_EN_2021sep.onnx \
  https://huggingface.co/opencv/text_recognition_crnn/resolve/main/text_recognition_CRNN_EN_2021sep.onnx
```

Then convert YOLOPv2 to ONNX for `cv.dnn`, once, on a machine with torch:

```bash
pip install torch==2.14.0 onnx==1.23.2
python src/scripts/export_onnx.py          # -> data/weights/yolopv2.onnx
python src/scripts/compare_onnx.py <frames_dir>   # cv.dnn vs TorchScript, expect IoU 1.0000
```

### 2. Laptop: tests and the virtual Pi

```bash
python3.13 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
for t in src/tests/test_*.py; do .venv/bin/python -m unittest "$t"; done   # 83 tests
```

`virtual_pi.py` replays a recorded drive in real time and uploads frames exactly as the Pi does,
through the real AWS pipeline:

```bash
LLD_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/<account-id>/lld-frames.fifo \
  .venv/bin/python src/scripts/virtual_pi.py <session_dir> --direct --size 640
```

`--no-upload` only plays it locally.

To share a recorded session, make a privacy-safe copy first. Faces and plate areas are blurred, the GPS
columns are removed from `decoded.csv`, and raw `can.csv` (which holds the VIN) is left out:

```bash
curl -L -o data/weights/privacy/face_detection_yunet_2023mar.onnx --create-dirs \
  https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx
.venv/bin/python src/scripts/privacy_package.py <session_dir> <out_dir> [--start 60 --end 600]
```

Check the before/after samples it writes to `<out_dir>/<session>-review/` before publishing.

### 3. EC2: the live service

On a COOL AMI instance (we use c8g.4xlarge for drives), inside COOL's venv. Don't install
`opencv-python` there: it would shadow COOL's build.

```bash
sudo apt install -y ffmpeg        # session videos are converted to H.264 so they play in a browser
/opt/cool/venvs/python_3.12/bin/pip install -r requirements.txt
LLD_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/<account-id>/lld-frames.fifo \
  /opt/cool/venvs/python_3.12/bin/python lld_aws.py live
# from the laptop: ssh -L 8080:localhost:8080 ubuntu@<instance>, then http://localhost:8080
```

Per-frame lag, inference and processing time go to CloudWatch (namespace `InitiativeVision`). The
dashboard definition is `infra/cloudwatch-dashboard.json`; the IAM and setup steps are in the setup guide.

Useful settings: `LLD_METRICS=0` turns CloudWatch off, `LLD_SIGNS=0` turns sign reading off, `LLD_MAX_LAG_S` sets how old a frame may
get before live mode skips it, and `LLD_WEIGHTS=data/weights/yolopv2.pt` switches to TorchScript.

## Results so far

| | |
|---|---|
| Capture to result, live from the Pi (direct mode, 640 uploads) | ~1.2 to 1.4 s |
| YOLOPv2 inference, COOL cv.dnn classic engine + FP16 target | 0.742 / 0.418 / 0.222 s on c8g.xlarge / 2xlarge / 4xlarge |
| Same model, PyTorch, c8g.xlarge | 0.89 s |
| cv.dnn vs TorchScript output, 40 real frames | mask IoU 1.0000, identical metrics |
| Stop signs, false positives | 0 in 6,154 frames (1280) and 3,060 frames (640) |

Method and full tables: [`phase2/ec2-compute-benchmark.md`](phase2/ec2-compute-benchmark.md).

## Limitations

The detector only finds vehicles (YOLOPv2 merges all traffic objects into one class). Speed-limit
reading has been validated only on synthetic signs so far. The bird's-eye view is valid only for
the calibrated camera pose. Frames don't reach AWS in cellular dead zones.
