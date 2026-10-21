# OpenCV AI Competition 2026: submission checklist

**Deadline: Sun Oct 26, 2026, 11:59 pm PT.** (Devpost's banner says 11:45 pm, so treat 11:45 as the real cutoff.)
Judging runs Oct 27 to Nov 9, and judges may open the endpoint at any point in that window.
Rules: https://opencv26.devpost.com/rules

**Mode: live only (decided 2026-10-21).** Everything runs through Pi → S3 → Lambda → SQS → EC2 as it is captured. There is no offline post-processing of full 30 fps recordings. What a drive leaves behind is the ~1 fps archive in `live/<session>/` plus the results in `lld/<session>/`.

Every entry must use **OpenCV 5 for substantive image/video analysis** and **run a meaningful component on AWS**.

## How we're scored

| Criterion | Weight | Where we stand (sweep, 2026-10-21) |
|---|---|---|
| Technical execution (OpenCV 5 depth, architecture, reliability, evaluation) | 30% | Pipeline works end to end. OpenCV does undistort, warp, contours and drawing, but **inference is PyTorch**, and there is **no evaluation yet** |
| Innovation | 20% | Live Pi→cloud lane pipeline. The CAN differentiator from the brief isn't used on the cloud side yet |
| Real-world impact | 20% | Needs a clear story in the report and video (lane departure / AV data) |
| User experience | 10% | Live view only reachable through an SSH tunnel; mp4v videos won't play in a browser |
| Documentation and presentation | 10% | Setup doc exists. No report, diagram, README or video |
| Cloud delivery, reproducibility, responsible operation | 10% | Least-privilege IAM and a DLQ are in place. Infra is all console clicks, deps aren't pinned, no CloudWatch metrics |

Special awards ($1k each, judged separately and optional): **Best Use of COOL** (needs verified COOL on Graviton plus a measured baseline) and **Agentic Vision** (vision output must change a later agent action). Nothing exists for the agentic one yet.

---

## 1. Required deliverables (missing any one makes the entry incomplete)

- [ ] **Technical report** covering problem, users, architecture, OpenCV 5 implementation, AWS deployment, evaluation, limitations, and responsible use
- [ ] **Judge-accessible repo** (`github.com/osfieldgaga/initiative-vision-aws`). Make it public or add judges.
  - [ ] README with what it is, a quickstart, and links to the report, video and endpoint
  - [ ] `yolopv2.pt` is gitignored, so document where to download it (or host it in S3)
  - [ ] Cleanup: add `__pycache__/` to `.gitignore` and untrack the `.pyc` files; decide on the 200 MB of `data/` and `outputs/`; move scratch scripts (`draw.py`, `read.py`, `rescale.py`, `yolop_test.py`, ...) somewhere out of the way
  - [ ] Remove the hardcoded account id from the `QUEUE_URL` default in `lld_aws.py`
- [ ] **Pinned dependencies**: `requirements.txt` for EC2 (boto3, torch, numpy, plus the COOL AMI version) and for local/tests. Add `boto3` to `.venv`, which fixes the one failing test.
- [ ] **Build / deploy / test instructions**: `live-pipeline-setup.md` is a good base. Add a how-to-run-the-tests section (5 suites, 62 tests, all pass except `test_upload_session` because boto3 is missing).
- [ ] **Architecture diagram** showing OpenCV 5 and AWS components, plus COOL if we claim it. A real image, not ASCII.
- [ ] **Working web endpoint** (or an arranged live screen-share). See section 2.
- [ ] **Video of 5 min or less**, public or unlisted. Must show the team, the app working, the architecture, and the main results.
- [ ] **Evaluation evidence, including failure cases**. See section 4.

## 2. Virtual Pi + judge endpoint

Judges mostly care about the AWS side. They won't have the car, so they get a **virtual Pi**: a replayer that takes a recorded session and feeds it into the *real* cloud pipeline exactly as the Pi would.

- [ ] **Virtual Pi replayer** (not started, planned for later)
  - Reads a live-archived session (`live/<session>/<frame time>.jpg`, at ~1 fps); the frame time is in the key
  - PUTs each frame to `live/latest_frame.jpg` with the same metadata as `upload_live_frame.py` (`session-start`, `frame-time`, `frame-seq`), at the original pace or sped up, respecting the 1 upload/s cap
  - Rebase the archived frame times onto "now" so the lag numbers on the live page stay meaningful
  - Goes through S3 → Lambda → SQS → EC2 unchanged, so judges see the actual deployed system rather than a mock
  - Open decision: how judges trigger it. Either (a) a "replay highway drive" button on the endpoint that starts it server-side (no AWS creds for judges, so this is preferred), or (b) a CLI plus a narrowly scoped IAM user
- [ ] **Public endpoint**: right now `live_view.py` binds 127.0.0.1 and is reached through `ssh -L`. Judges need a URL.
  - HTTPS with basic auth, or a shared-secret link (so we don't expose an open EC2 port)
  - Show the live stream while a replay runs, plus **finished sessions** (`lld/<session>/` videos + metrics) for when nobody is replaying
  - Re-encode outputs to **H.264** so they play in a browser (mp4v doesn't)
- [ ] **Availability for Oct 27 to Nov 9**: either keep EC2 up (check cost: c8g.xlarge is roughly $0.16/h on-demand, ~$50 for two weeks, plus the COOL fee if it isn't covered by credits) or serve finished results from S3/static hosting and start EC2 only for live demos. Pick one and write it in the report.

## 3. The highway recording (one of the core submission recordings)

Before the drive:
- [ ] **Camera mount unchanged since the cone calibration.** The BEV homography (`src` points in `lld.py`) is only valid for that exact pose. If it moved, redo the 6x6 m cone square.
- [ ] Pi clock synced (`timedatectl`); note the clock source in `session.json`
- [ ] Pi free disk is enough for hours of 1280x1280 JPEG at 30 fps (check MB/min from an existing session)
- [ ] CAN recording on, with the standard decode blacklist ready
- [ ] Live uploader running over cellular for the whole drive, and EC2 `lld_aws.py live` running before you leave. Results are only produced live.
- [ ] Coverage gaps: cellular dead zones mean frames never reach S3. Note where they happen; they're an honest limitation for the report.
- [ ] Frames skipped for lag (`LLD_MAX_LAG_S`) are still archived in `live/<session>/`. Decide whether to run with `LLD_MAX_LAG_S=0` for the recording drive so every uploaded frame gets processed.

After the drive:
- [ ] Pull clips for the video: good lanes, a lane change, a departure, and failure cases (faded lines, glare, tunnel/overpass shadow, trucks occluding lines)
- [ ] **Privacy pass** before anything is made public: plates and faces in the footage (see section 6)

## 4. Evaluation (30% "technical execution" includes evaluation)

- [ ] **Latency / throughput**: we already log download / inference / post-proc / outputs per frame. Report p50/p95 end-to-end lag (Pi capture → EC2 processed) and frames/s per instance type (m8g.medium ~4 s/frame vs c8g.xlarge ~1 s/frame is already a finding).
- [ ] **Lane detection quality**: % of frames with `lane_valid`, plus a small hand-labeled set (e.g. 100–200 frames from the highway drive: lanes visible y/n, departure y/n) for precision/recall of `lane_valid` and `departure`
- [ ] **CAN cross-check (cheap differentiator)**: only possible if CAN reaches the cloud live, e.g. speed/steering/turn signal added to the upload metadata. Otherwise cut it. If we do it: compare vision-detected lane changes/departures against steering angle and turn signal from `decoded.csv`. "Departure with no turn signal" is a real ADAS feature, and it uses the CAN bus nobody else has.
- [ ] **Cost**: $/drive-hour (the table in `live-pipeline-setup.md` is a start; add the EC2 and COOL fee)
- [ ] **Failure cases with images** (required by the rules)

## 5. Correctness fixes (known open items)

- [x] Missing lane reported as -4.00 m offset (fixed 2026-10-21). Per-side offsets are `None` when that side isn't detected; `offset_m`/`road_width_m` are `None` unless both are. Overlay, `format_metrics` and the live page show n/a. Note: sessions processed before the fix still have the bad values in their `metrics.json`.
- [ ] Lambda memory 128 → 256 MB
- [ ] Stop EC2 when idle (or a schedule / CloudWatch alarm on queue-empty)

## 6. Cloud delivery, reproducibility, responsible operation (10%)

- [ ] **Reproducible infra**: a CloudFormation/SAM template or a single `aws` CLI script that creates the bucket config, queues, Lambda, roles and trigger. Currently it's all console steps in `live-pipeline-setup.md`. This is one of the cheapest points available.
- [ ] **Observability**: push per-frame lag/inference as CloudWatch custom metrics (or embedded-metric-format logs) + one dashboard screenshot; DLQ depth alarm
- [ ] **Security**: endpoint auth; no creds in repo (check git history); IAM policies already least-privilege, so say so in the report
- [ ] **Responsible use**: footage contains other drivers' plates and maybe pedestrians' faces. Either blur them (OpenCV can do it: a plate/face detector plus `GaussianBlur`) or state that only lane overlays/BEV are published. Also include a "this is not a safety system, no vehicle actuation" statement, which the brief already holds.

## 7. OpenCV 5 and COOL (requirement + optional award)

- [ ] **Verify OpenCV 5 on the EC2 instance**: run `python -c "import cv2; print(cv2.__version__)"` in the COOL venv. Local `.venv` is 5.0.0. If COOL ships 4.x we have a requirement problem, so check this first.
- [x] **YOLOPv2 on `cv.dnn`** (2026-10-21): `src/scripts/export_onnx.py` exports the two seg heads to ONNX (99 MB, detection head pruned) and `lld.load_model` picks cv.dnn for `.onnx`, which is now the default. Torch is no longer needed at inference. `src/scripts/compare_onnx.py` on 40 real frames: lane/drivable mask IoU 1.0000, lane_valid/departure 40/40, offset diff 0.0000 m; median inference 0.35 s cv.dnn vs 0.50 s torch (x86 laptop CPU).
  - [ ] On EC2: confirm COOL's cv2 is 5.x with the DNN module and loads the ONNX, then measure per-frame time on the c8g vs the old ~0.89 s torch number
  - [ ] Optional: try `net.setPreferableBackend/Target` options and an FP16 export for more speed
- [ ] **Best Use of COOL** (optional): record COOL version + instance config, prove COOL executes the workload (`cv2.getBuildInformation()`), and benchmark the *pipeline's* OpenCV stages (undistort, resize, warp, cvtColor) on Graviton+COOL vs stock OpenCV (same box, and ideally an x86 instance). `time_cool_vs_stock.py` covers resize/adaptiveThreshold only, and no numbers have been recorded yet.

## 8. Scope decisions to make now (5 days left)

The Phase 2 brief (scenario mining, MCP agent, pseudo-label panels, MATLAB) is far beyond what fits in 5 days. Suggested cut, to be confirmed:

- **Keep / finish:** live lane pipeline, virtual Pi, public endpoint, highway drive, evaluation incl. a CAN cross-check, report, video
- **Now (in progress):** cv.dnn port (helps the 30% criterion and COOL), COOL benchmark
- **Cut:** MCP agent / Agentic Vision award, depth/semantic panels, MATLAB. Say so in the report's "limitations / future work".

## Suggested order

1. Day 1 (Oct 21–22): fix missing-lane metrics, verify OpenCV 5 on EC2, prep and do the highway drive
2. Day 2: replay the drive, pin deps, README, repo cleanup, H.264 outputs
3. Day 3: virtual Pi + public endpoint, IaC script, CloudWatch metrics
4. Day 4: evaluation numbers + failure cases, architecture diagram, report draft
5. Day 5 (Oct 25–26): record and edit the video, final read-through, submit with margin (not at 11:40 pm)
