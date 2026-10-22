# OpenCV AI Competition 2026: submission checklist

**Deadline: Sun Oct 26, 2026, 11:59 pm PT.** (Devpost's banner says 11:45 pm, so treat 11:45 as the real cutoff.)
Judging runs Oct 27 to Nov 9, and judges may open the endpoint at any point in that window.
Rules: https://opencv26.devpost.com/rules

**Mode: live only (decided 2026-10-21).** Everything runs through Pi → S3 → Lambda → SQS → EC2 as it is captured. There is no offline post-processing of full 30 fps recordings. What a drive leaves behind is the ~1 fps archive in `live/<session>/` plus the results in `lld/<session>/`.

Every entry must use **OpenCV 5 for substantive image/video analysis** and **run a meaningful component on AWS**.

## Checklist by judging criterion (sweep, 2026-10-22)

✅ done · 🟡 partial · ⬜ not started. **Bold** = essential (required by the rules, or a large share of that criterion's points).

### Technical execution, 30%: correctness and depth of the OpenCV 5 work
- ✅ The whole perception stack runs on OpenCV 5: undistort, BEV homography warp, YOLOPv2 on `cv.dnn` (parity IoU 1.000 vs torch), `NMSBoxes` vehicle decoding, HSV/contour/shape sign candidates, CRNN OCR on `cv.dnn` with lexicon-constrained CTC
- ✅ Tests: unit tests for decoding, CTC, the sign reader and metrics formatting
- ⬜ **Verify on EC2 that COOL's `cv2` is 5.x with DNN and loads `yolopv2.onnx`.** If it isn't, the "OpenCV 5" requirement and the COOL story both break, so do this first
- ⬜ **Lane evaluation**: hand-label 100–200 highway frames (lanes visible y/n, departure y/n), then report precision and recall for `lane_valid` and `departure`
- 🟡 **Latency/throughput**: per-stage timings are logged; p50/p95 end-to-end lag on the c8g hasn't been reported yet
- 🟡 Sign evaluation: real stop-sign results and synthetic speed-limit results exist only as scratch scripts. Make them reproducible (`eval_signs.py`) and test speed limits on real highway signs
- ⬜ **Failure cases with images** (required): faded lines, glare, intersections, trucks, the "lane_valid with L off -3.99 m" anomaly
- 🟡 Reliability: DLQ and lag-skip logic are in place; the behaviour during cellular gaps hasn't been documented yet

### Innovation, 20%: originality and thoughtful use of CV/AI
- ✅ A real car feeds a live cloud ADAS pipeline over cellular, from a Pi, rather than an offline demo
- ✅ Sign OCR scored against the legal lexicon (CTC), which turns a weak generic OCR into a reader with 0 false positives. This is a good "thoughtful" point to explain
- ⬜ Explain it clearly in the report and video. Innovation that isn't explained doesn't score
- ⬜ (Optional) Forward collision distance via `perspectiveTransform` on box bottoms. This is cheap and adds a visible ADAS feature

### Real-world impact, 20%: importance, usefulness, evidence of benefit
- ⬜ **Problem/user story**: who uses it (fleet/driver monitoring, AV data collection, retrofit ADAS for older cars) and why cloud processing makes sense
- ⬜ **Evidence**: highway drive results, cost per drive-hour, latency budget versus what a warning needs
- ⬜ Honest framing: "not a safety system, no actuation"

### User experience, 10%: usability, accessibility, interaction
- 🟡 The live page shows the stream, metrics, objects, signs and timings, but it's only reachable through an SSH tunnel
- ⬜ **Public judge URL** with auth, and a "replay a drive" button (virtual Pi) so judges can see it work without the car
- ⬜ Finished sessions view with **H.264** videos (mp4v doesn't play in browsers)
- ⬜ Quick accessibility pass: readable contrast, departure warning shown as text as well as colour

### Documentation and presentation, 10%
- 🟡 Setup doc (`live-pipeline-setup.md`) exists
- ⬜ **README** (what it is, quickstart, links to the report, video and endpoint, plus where to download the weights)
- ⬜ **Technical report** (problem, users, architecture, OpenCV 5, AWS, evaluation, limitations, responsible use)
- ⬜ **Architecture diagram** (Pi → S3 → Lambda → SQS → EC2/COOL → endpoint; OpenCV stages marked)
- ⬜ **Video of 5 minutes or less** (team, app working, architecture, results)
- ⬜ Repo cleanup: untrack `.pyc`, move scratch scripts, decide what to do with `data/` and `outputs/`, remove the hardcoded account ID

### Cloud delivery, reproducibility, responsible operation, 10%
- ✅ Least-privilege IAM, DLQ, S3 versioning, a working event-driven pipeline on Graviton
- ⬜ **Pinned dependencies** (`requirements.txt` for EC2 and for local/tests)
- ⬜ IaC (one `aws` CLI script or a SAM template). Right now it's all console clicks
- ⬜ CloudWatch: lag/inference custom metrics, one dashboard screenshot, a DLQ alarm
- ⬜ Security: endpoint auth, and check the git history for credentials
- ⬜ **Privacy**: blur plates and faces, or publish only overlays. State it in the report
- ⬜ Plan for keeping the endpoint up from Oct 27 to Nov 9 (keep EC2 running, or serve static results from S3 and start EC2 only for demos)

## Prizes: which to run for (decided 2026-10-22)

- **Main prizes (1st–3rd):** automatically entered. Everything above feeds this.
- **Best Use of COOL ($1k): no longer an easy win (re-assessed 2026-10-22).** The workload does run on COOL on Graviton, with scaling measured at 4, 8 and 16 vCPUs. But stock `opencv-python-headless` 5.0.0 on the same c8g.4xlarge ran YOLOPv2 just as fast (0.223 vs 0.222 s, classic FP16), because the stock aarch64 wheel already bundles KleidiCV and the same DNN MLAS kernels. The image operations around inference total only about 12 ms per 640 frame (`src/scripts/ops_bench.py`, laptop), so even a large COOL speed-up there barely changes the frame time. Still worth submitting as an honest, reproducible comparison, but it's unlikely to win. Finish `ops_bench.py` in both environments, then decide.
- **Agentic Vision ($1k): skip.** Nothing exists yet. It needs an agent loop where vision output changes a later action, plus a workflow diagram, a trace, and evaluation of success, failure handling and human control. That's 1.5–2 days we don't have, and a rushed version risks being judged as "a chatbot explaining a fixed result". Revisit only if everything else is finished by Oct 25. The cheapest qualifying idea: a controller that reacts to low `lane_valid` or a sign sighting by requesting full-rate uploads for the next N seconds, and flags segments for human review.

---

## 1. Required deliverables (missing any one makes the entry incomplete)

- [ ] **Technical report** covering problem, users, architecture, OpenCV 5 implementation, AWS deployment, evaluation, limitations, and responsible use
- [ ] **Judge-accessible repo** (`github.com/osfieldgaga/initiative-vision-aws`). Make it public or add judges.
  - [x] README with what it is, a quickstart, and links to the report, video and endpoint (2026-10-22; the links are placeholders until those exist)
  - [x] `yolopv2.pt` and `yolopv2.onnx` are gitignored: the README has the download URLs and the export step
  - [x] Cleanup (2026-10-22): `__pycache__/` gitignored and untracked; scratch scripts moved to `archive/early-experiments/`. Still open: whether to keep the ~430 MB of `data/` and `outputs/` in git
  - [x] Remove the hardcoded account id from the `QUEUE_URL` default in `lld_aws.py` (now required via `LLD_QUEUE_URL`)
- [x] **Pinned dependencies** (2026-10-22): `requirements.txt` (EC2, installed into COOL's venv; OpenCV comes from the COOL AMI) and `requirements-dev.txt` (laptop). `boto3` is in `.venv`, so all 8 test files (83 tests) pass
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
  - [x] Outputs re-encoded to **H.264** with ffmpeg at session finalize (2026-10-22); install ffmpeg on EC2
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

- [x] **Latency (2026-10-22): from ~3.2 s to ~0.87 s capture-to-result.** The S3 event delivery to Lambda (~1 s) was removed from the live path: the Pi now uploads to the archive key and sends the SQS message itself. Uploads are 640x640 q70 (~25 KB instead of ~190 KB), and `lld.py` scales the intrinsics to match. EC2 inference uses the classic cv.dnn engine with the FP16 target; c8g.4xlarge for drives. Details are in `live-pipeline-setup.md` and `ec2-compute-benchmark.md`.
  - [ ] First real Pi run in direct mode, after adding the IAM policy (`s3:PutObject` live/*, `sqs:SendMessage`) to the Pi's user
  - [ ] Validate 640 uploads on real speed-limit signs: the 48 px minimum means signs must be about twice as close as at 1280

- [x] Missing lane reported as -4.00 m offset (fixed 2026-10-21). Per-side offsets are `None` when that side isn't detected; `offset_m`/`road_width_m` are `None` unless both are. Overlay, `format_metrics` and the live page show n/a. Note: sessions processed before the fix still have the bad values in their `metrics.json`.
- [ ] Lambda memory 128 → 256 MB
- [ ] Stop EC2 when idle (or a schedule / CloudWatch alarm on queue-empty)

## 6. Cloud delivery, reproducibility, responsible operation (10%)

- [ ] **Reproducible infra**: a CloudFormation/SAM template or a single `aws` CLI script that creates the bucket config, queues, Lambda, roles and trigger. Currently it's all console steps in `live-pipeline-setup.md`. This is one of the cheapest points available.
- [x] **Observability** (2026-10-22): per-frame lag, its hops, inference and processing time go to CloudWatch (`InitiativeVision`, p50/p95 via value lists); dashboard in `infra/cloudwatch-dashboard.json`. Still to do: dashboard screenshot for the report, DLQ depth alarm
- [ ] **Security**: endpoint auth; no creds in repo (check git history); IAM policies already least-privilege, so say so in the report
- [ ] **Responsible use**: footage contains other drivers' plates and maybe pedestrians' faces. Either blur them (OpenCV can do it: a plate/face detector plus `GaussianBlur`) or state that only lane overlays/BEV are published. Also include a "this is not a safety system, no vehicle actuation" statement, which the brief already holds.

## 7. OpenCV 5 and COOL (requirement + optional award)

- [ ] **Verify OpenCV 5 on the EC2 instance**: run `python -c "import cv2; print(cv2.__version__)"` in the COOL venv. Local `.venv` is 5.0.0. If COOL ships 4.x we have a requirement problem, so check this first.
- [x] **YOLOPv2 on `cv.dnn`** (2026-10-21): `src/scripts/export_onnx.py` exports `data/weights/yolopv2.onnx` (seg heads + raw detection maps; the anchor grids are hardcoded in `lld.DET_ANCHORS`), and `lld.load_model` picks cv.dnn for `.onnx`, which is now the default. Torch is no longer needed at inference. `src/scripts/compare_onnx.py` on 40 real frames: lane/drivable mask IoU 1.0000, detections 40/40 same count with box IoU 1.0000, lane_valid/departure 40/40, offset diff 0.0000 m; median inference 0.42 s cv.dnn vs 0.50 s torch (x86 laptop CPU).
  - [ ] On EC2: confirm COOL's cv2 is 5.x with the DNN module and loads the ONNX, then measure per-frame time on the c8g vs the old ~0.89 s torch number
  - [ ] Optional: try `net.setPreferableBackend/Target` options and an FP16 export for more speed
- [ ] **Best Use of COOL** (optional): record COOL version + instance config, prove COOL executes the workload (`cv2.getBuildInformation()`), and benchmark the *pipeline's* OpenCV stages (undistort, resize, warp, cvtColor) on Graviton+COOL vs stock OpenCV (same box, and ideally an x86 instance). `time_cool_vs_stock.py` covers resize/adaptiveThreshold only, and no numbers have been recorded yet.

## 7b. Object detection and ADAS stretch goals

- [x] **Vehicle boxes from YOLOPv2** (2026-10-21): `lld.decode_detections` decodes the YOLOv7-style head with `cv.dnn.NMSBoxes`. Boxes are drawn on the overlay, `metrics["objects"]` holds `[{label, score, box}]` nearest first, and the live page lists them.
  - **Limitation:** YOLOPv2's detector only finds **vehicles**. It was trained on BDD100K with every traffic object merged into one class, so every box comes out as class 3. On 81 frames it found no people, traffic lights or signs, even where they were clearly visible. Put this in the report.
- [ ] **Second detector for people / traffic lights** (optional now that signs are handled by `signs.py`): a small COCO model (e.g. YOLOv8n/YOLO11n ONNX on cv.dnn) covers `person`, `bicycle` and `traffic light`. Cost: a second forward pass per frame, so measure it on the c8g first.
- [x] **Stop-sign + speed-limit reader** (`signs.py`, 2026-10-22), all OpenCV:
  1. color: HSV red mask for STOP; Canny edges for the white speed sign (a white sign on a bright sky has no color edge)
  2. shape: octagon-ish red blob / upright ~4:5 quadrilateral with a bright face and dark legend
  3. OCR: OpenCV's CRNN (`cv.dnn.TextRecognitionModel` weights, run through `cv.dnn`) on the text band, scored with **lexicon-constrained CTC**: log P("stop") or log P(each of 15..85 step 5) versus the free-form best reading. That's why a 60 px blurry "stoe" still counts as STOP, and a random blob reading "t" doesn't.
  - On the real stop-sign approach (`20260927T184255Z-fddd` frames 4000-4120): STOP in 20/83 frames, and basically every frame once the sign is >= ~60 px wide.
  - **False positives: 0** across 6,154 frames (every 15th frame of all local sessions), including 107 red look-alikes (lens flares, billboards, brake lights, trees tinted pink by the camera).
  - Speed limits: **no speed-limit sign exists in the local footage**, so this was validated synthetically. The public-domain MUTCD R2-1 sign (Wikimedia) with each value re-drawn was pasted into real frames with blur, purple cast and JPEG q70; 2,000 signs over 5 seeds. Read correctly: 100% at >= 85 px wide, ~70% at 50-70 px (misses are all at the candidate stage), **0 misreads**. Signs under 48 px are skipped on purpose: at 40 px "55" was read as "65".
  - Cost: `detect()` median 16 ms, p95 ~60 ms per frame on this laptop
  - [ ] **Validate speed limits on the highway drive**, the first real data with R2-1 signs. At ~1 upload/s a sign may only be in 1–2 frames at readable size.
  - [ ] Not handled: the yellow advisory speed signs, "MINIMUM", night (retroreflective glare), signs seen at an angle. The CRNN font isn't Highway Gothic.
  - [ ] Optional: promote the scratch evaluation (real sweep + synthetic) to `src/scripts/eval_signs.py` so the numbers are reproducible for the report
  - [ ] Stretch: persist the last speed limit across frames (it stays in force until the next sign) and show it on the live page

- [ ] **Forward collision warning** (stretch): the bottom-center of each vehicle box is on the road, so `cv.perspectiveTransform` with the BEV homography `lld.M` gives its distance ahead (y=700 is 4.874 m from the car) and its lateral position (in-lane or not). Then warn on distance or on its change between frames (time-to-collision). Caveats: the ground-plane assumption breaks on hills, and the homography was calibrated only out to ~7 m, so far estimates are rough. Car speed isn't uploaded live, so TTC would come from vision only.

## 8. Scope decisions to make now (5 days left)

The Phase 2 brief (scenario mining, MCP agent, pseudo-label panels, MATLAB) is far beyond what fits in 5 days. Suggested cut, to be confirmed:

- **Keep / finish:** live lane pipeline, virtual Pi, public endpoint, highway drive, evaluation incl. a CAN cross-check, report, video
- **Done:** cv.dnn port, vehicle boxes, sign reader. **Next:** COOL verification + benchmark (Best Use of COOL)
- **Cut:** MCP agent / Agentic Vision award, depth/semantic panels, MATLAB. Say so in the report's "limitations / future work".

## Suggested order (revised 2026-10-22; the highway drive hasn't happened yet)

1. **Thu Oct 22:** on EC2, verify COOL `cv2` 5.x + DNN, deploy the ONNX, OCR weights and `signs.py`, and measure per-frame time. Then prep for and **do the highway drive**, since everything downstream needs that footage.
2. **Fri Oct 23:** COOL vs stock benchmark on the c8g (optionally x86); pinned requirements, README, repo cleanup, H.264 outputs.
3. **Sat Oct 24:** virtual Pi + public endpoint with auth, IaC script, CloudWatch metrics.
4. **Sun Oct 25:** evaluation (hand labels, latency, signs, failure cases), architecture diagram, report draft.
5. **Mon Oct 26:** video, privacy pass, final read-through, submit by early evening PT. Don't leave it until 11:40 pm.
