# EC2 compute benchmark: YOLOPv2 on OpenCV 5 + COOL

**Goal:** pick the instance size for live mode, and document how COOL's OpenCV performs on Graviton, as evidence for the Best Use of COOL award.

**So far:** with the classic `cv.dnn` engine and FP16 target, YOLOPv2 takes **0.742 s** on c8g.xlarge, **0.418 s** on c8g.2xlarge and **0.222 s** on c8g.4xlarge, scaling at 84–89% efficiency. On the xlarge that already beats PyTorch's 0.89 s on the same instance.

## Setup

| | |
|---|---|
| Region | us-east-1 |
| AMI | COOL (AWS Marketplace). *To record: AMI name and version.* |
| OpenCV | 5.1.0-dev, from COOL's venv `/opt/cool/venvs/python_3.12` (`cv2` at `/opt/cool/python_3.12/site-packages`) |
| COOL build (from `getBuildInformation()`, saved as `cool_buildinfo.txt` on the instance) | Custom HAL: carotene 0.0.1 + KleidiCV 0.7.0 · DNN MLAS with ARM64 NEON SGEMM/SGEMV kernels · CPU baseline SVE, NEON, FP16, NEON_DOTPROD, NEON_FP16, NEON_BF16 · pthreads |
| Model | YOLOPv2 exported to ONNX (`src/scripts/export_onnx.py`), 1x3x640x640 input; outputs `da`, `ll`, `det8`, `det16`, `det32` |
| Threads | OpenCV default: one per vCPU |

## Method

1. Stop `lld_aws.py`, so nothing else competes for the CPU.
2. From the repo root, in COOL's venv:
   ```bash
   /opt/cool/venvs/python_3.12/bin/python src/scripts/dnn_bench.py
   ```
3. The script loads the model once per engine/target combination and runs one warm-up pass. It then times 5 forward passes and prints the median and the minimum. The input is random, because a conv net's cost doesn't depend on the pixel values.
4. To check that accuracy holds on real frames, run `src/scripts/dnn_check.py <frames_dir> 30`. It times the configurations interleaved, frame by frame, and gives lane/drivable-area IoU and detection agreement against the new engine in FP32.

`[ WARN ... ]` lines are expected:
- "Targets are not supported by the new graph engine": the new engine ignores the FP16 target and runs FP32.
- "fall back to DNN_TARGET_CPU": printed for classic FP16. *Open question:* on the c8g.xlarge that run was still 24% faster than classic FP32, so the warning may not mean what it says. `dnn_check.py` should settle it.

## Results: YOLOPv2 inference, median seconds per frame

| Instance | vCPU | Arch | new / cpu | new / cpu_fp16 | classic / cpu | **classic / cpu_fp16** | On-demand $/h (approx.) |
|---|---|---|---|---|---|---|---|
| c8g.xlarge | 4 | Graviton4 | 1.203 | 1.199 | 0.969 | **0.742** | ~0.16 |
| c8g.2xlarge | 8 | Graviton4 | 0.750 | 0.622 | 0.565 | **0.418** | ~0.32 |
| c8g.4xlarge | 16 | Graviton4 | 0.346 | 0.358 | 0.320 | **0.222** | ~0.64 |
| Dev laptop (i7-10750H, x86, not EC2) | 12 threads | x86-64 | 0.414 | 0.413 | 0.743 | 0.748 (falls back to FP32) | n/a |

All three c8g rows are from 2026-10-22. For classic / cpu_fp16 the min was within 0.003 s of the median on the xlarge and 2xlarge, and within 0.011 s on the 4xlarge. On the 2xlarge the new engine was noisy: "new / cpu" and "new / cpu_fp16" are the same computation, because the new engine ignores the target, yet they gave 0.750 and 0.622 s (min 0.678 and 0.616).

**Scaling, classic / cpu_fp16:**

| Instance | Inference | Speed-up vs 4 vCPU | Parallel efficiency | Compute cost per 1,000 frames (approx.) |
|---|---|---|---|---|
| c8g.xlarge | 0.742 s | 1.00x | 100% | $0.033 |
| c8g.2xlarge | 0.418 s | 1.78x | 89% | $0.037 |
| c8g.4xlarge | 0.222 s | 3.34x | 84% | $0.039 |

Parallel efficiency = speed-up / (vCPUs / 4). OpenCV's DNN keeps using the extra cores up to 16 vCPUs, so the cost per frame of compute rises only about 18% from the smallest instance to the largest. In live mode, though, the instance runs for the whole drive at 1 frame/s, so what you actually pay is the hourly rate: x1, x2 or x4.

Across all three sizes, the FP16 target on the classic engine was 23–31% faster than classic FP32 (0.742 vs 0.969, 0.418 vs 0.565, 0.222 vs 0.320), whatever the warning says. The laptop ran stock OpenCV 5.0.0, not COOL, so it isn't a COOL baseline. It shows that the best engine depends on the CPU: on x86 the new engine is faster, and on Graviton the classic engine is.

Prices are approximate on-demand EC2 rates, excluding any COOL Marketplace software fee. *To check: the COOL fee per instance size.*

## Earlier baselines (same YOLOPv2 model)

| Instance | Backend | Inference s/frame | Notes |
|---|---|---|---|
| m8g.medium (1 vCPU) | PyTorch (TorchScript) | ~4 | couldn't keep up; queue lag reached 160 s |
| c8g.xlarge (4 vCPU) | PyTorch (TorchScript) | 0.89 | before the move to cv.dnn |
| c8g.xlarge (4 vCPU) | cv.dnn new engine, live run | 1.21 to 1.23 | 20 frames from the Pi feed, session `20261022T140513Z`; total per frame 1.29 to 1.51 s |

## Live per-frame timing by instance (from the `lld_aws.py` log)

Fill in from a virtual-Pi run in `--direct` mode (`src/scripts/virtual_pi.py`), ignoring the first 2 frames. Use the median over at least 30 frames.

| Instance | Engine / target | infer | signs | post | total | lag p50 | lag p95 |
|---|---|---|---|---|---|---|---|
| c8g.xlarge | classic / cpu_fp16 | | | | | | |
| c8g.2xlarge | classic / cpu_fp16 | | | | | | |
| c8g.4xlarge | classic / cpu_fp16 | | | | | | |

## Still to measure

- [x] c8g.2xlarge and c8g.4xlarge results (`dnn_bench.py`, 2026-10-22)
- [ ] `dnn_check.py` on real frames: is FP16 really faster, and do the masks still agree?
- [ ] **COOL vs stock OpenCV** on the same instance: `pip install opencv-python-headless` in a *separate* venv (never COOL's) and rerun `dnn_bench.py`. This is the baseline the COOL award asks for.
- [ ] Optional: the same model on an x86 instance of the same size (e.g. c7i.xlarge), for an Arm vs x86 cost comparison
- [ ] Cost per drive-hour for each instance (EC2 + COOL fee) next to the live lag
