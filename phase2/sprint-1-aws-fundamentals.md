# Phase 2, Sprint 1 — AWS fundamentals: S3, triggers, EC2, COOL

**Status:** Planned, awaiting your approval
**Depends on:** at least one real Phase 1 session to use as test data (you have several)
**Cost:** the S3 + Lambda parts are effectively free at this scale. The EC2 + COOL parts cost real money per hour — see §4, and nothing there launches without your go-ahead on the specific instance and cost first.

## Goal

Not to build the real triage pipeline yet — to get you hands-on comfortable with the four building blocks the whole rest of Phase 2 sits on: an S3 bucket you can put things in and get them out of, an event that fires the instant something lands there, a Graviton EC2 instance you understand how to launch and reach, and COOL specifically (which turns out to work differently than "install a library" — see below). Each part produces something small and real, not just reading.

## What COOL actually is (verified, not assumed)

Checked against opencv.org and the AWS Marketplace listing directly, because the brief itself flags "does COOL accelerate DNN inference?" as unresolved and I didn't want to plan around a guess:

- **COOL (Cloud-Optimized OpenCV Library)** is an AWS Marketplace product: a pre-built **Amazon Machine Image (AMI)**, Ubuntu 24.04 on Arm, built with Arm's KleidiCV optimizations.
- It is **not** a package you `pip install` into an environment you already have. You launch an EC2 instance *from the COOL AMI specifically*, on a Graviton4 instance (`c8g`, `m8g`, or `r8g` family — the vendor recommends `m8g.4xlarge` or larger "for optimal performance"), SSH in, and `source` one of the ready-made Python virtualenvs at `/opt/cool/venvs/` (3.10, 3.11, or 3.12). The accelerated OpenCV build is already there.
- What it documents accelerating: classic/core OpenCV operations — resize, adaptive Gaussian thresholding, contour detection, and similar. **Whether it accelerates DNN inference (the depth/segmentation models in later sprints) is still not confirmed anywhere I found** — that's a real open question, worth testing directly once we're there, not assuming either way.
- Pricing: metered hourly, on top of the normal EC2 instance cost — the listing shows examples from $0.01/hr (`c8g.large`) to $0.80/hr (`m8g.48xlarge`). A 7-day free trial exists and **auto-converts to a paid subscription** if you don't cancel it.

This shapes the architecture: COOL can only ever run *on an EC2 instance launched from that AMI*. It can't run inside a Lambda function, a container task, or any box you didn't launch from it. That's why this sprint teaches the S3-trigger part separately, with something free and instant (Lambda), before bringing in the EC2/COOL part at all.

## Part A — S3: put something in, get it out

Learn by doing, in this order:

1. **Console first.** Create a bucket (region matters — see the roadmap's open question), upload one file by hand through the web console, download it back. See what a bucket, an object, and a "key" (the full path-like name, e.g. `sessions/2026.../frames/00000000_....jpg`) actually are. S3 has no real folders — the `/`-separated prefix in a key just *looks* like one in the console.
2. **CLI/boto3 next**, once the concept is concrete. I'll write a small script (`upload_session.py`) that mirrors a Phase 1 session's local layout into S3 under the same relative paths — this becomes the actual "Pi logs → S3 raw store" step from the pipeline diagram later, so it's not throwaway.
3. **Permissions, briefly.** Just enough to understand: a bucket is private by default; an IAM user/role needs an explicit policy to read or write it; this is why "AWS keeps refusing my upload" is almost always a permissions problem, not a bug.

## Part B — Triggers: something reacts the instant a file lands

1. **S3 Event Notifications → Lambda**, the most direct mechanism, learned with something trivial and free: a Lambda function that just logs the bucket/key it was triggered with (visible in CloudWatch Logs), or copies the object to a `processed/` prefix as proof it ran. Console-first again, so you see the pieces: the S3 bucket's notification configuration, the Lambda function itself (Python), and the IAM permission that lets S3 invoke it.
2. **Why not stop there for the real pipeline.** Lambda can't run COOL (previous section) and has execution-time/memory limits unsuited to real image processing anyway. So the trigger's real job, later, is narrower than "do the processing" — more likely S3 → an event → a queue (SQS) that a COOL-running EC2 instance drains, so the expensive instance only runs when there's actual work queued, not sitting idle. That's a later sprint's design question, not this one's — here the goal is just understanding the trigger mechanism itself.
3. Worth knowing it exists: **EventBridge** is AWS's newer, more general event bus, and is what AWS now generally recommends over the older direct S3-Event-Notification mechanism for new projects. I'll show the simple direct version first since it's easier to see cause-and-effect, and mention where EventBridge would replace it later.

## Part C — EC2, cheaply, before EC2 with COOL, which costs more

1. **Free-tier EC2 first.** Launch a `t4g.micro` (also Graviton/ARM — free-tier eligible for a new AWS account for 12 months) with plain Ubuntu, SSH in, run `uname -m` (confirms `aarch64`, i.e. you're really on ARM, same architecture as the Pi), and terminate it. This is EC2 mechanics — AMI, instance type, key pair, security group — with no cost pressure while you're still learning where the buttons are.
2. **Then COOL, deliberately, with the cost stated up front.** Subscribe to the COOL AMI in Marketplace (starts the 7-day trial clock), launch the *cheapest* viable size first (`c8g.large`, ~$0.01/hr software fee, likely enough to prove the workflow even though the vendor recommends bigger for real throughput), SSH in, source the venv, and run one accelerated operation — a resize or threshold on a single real frame from your Phase 1 data — timed against plain OpenCV on the same box. That single number is your first real data point toward the "Best Use of COOL" evidence the brief asks for later.
3. **Stop or terminate the instance the moment you're done**, and put a reminder in about day 6 to cancel the Marketplace subscription if you're not continuing straight into Sprint 3 (COOL triage at scale) — otherwise the trial silently becomes a paid subscription.

## What I'll prepare vs. what you do

Per the roadmap's default: I write the code and the exact command sequences; you run them in the AWS console/CLI yourself, since it's your account and your money. Tell me if you'd rather I execute some of this directly (I'd need AWS credentials with appropriately scoped permissions from you to do that, and would still confirm before anything that costs money).

What I'll have ready before you start:
- `upload_session.py` — mirrors a local Phase 1 session into an S3 bucket/prefix.
- The Lambda function code and its S3 event notification JSON config.
- The exact `aws ec2 run-instances` / console steps for both the free-tier test and the COOL instance, with the specific AMI ID and instance types named.
- A tiny `time_cool_vs_stock.py` that runs the same OpenCV operation via the COOL venv and the system's stock OpenCV, and prints both timings — the actual measurement, not just "COOL should be faster."

## Acceptance criteria

1. A real Phase 1 session's frames exist in an S3 bucket, uploaded by your own script run.
2. Uploading a new file to that bucket visibly triggers the Lambda (seen in CloudWatch Logs), without you doing anything else.
3. A `t4g.micro` was launched and reached over SSH, then terminated.
4. A COOL-AMI instance was launched, one real frame was processed with a COOL-accelerated operation, and its timing was compared against stock OpenCV on the same machine — with the actual numbers written down.
5. Nothing costing money is still running when the sprint ends.

## Open questions for you

1. AWS account and region (roadmap §1).
2. Cost comfort, specifically: is a few dollars for this sprint's EC2/COOL time fine, or do you want to stay on the absolute cheapest instance sizes throughout? (roadmap §2)
3. Rough AWS experience level, so I calibrate explanation depth (roadmap §3).
4. Do you want me executing any of this directly, or is it entirely you-driven for this sprint? (roadmap §4)
5. Which single OpenCV frame should we use for the COOL timing test — any real frame from one of your sessions is fine; I'll pick one from the most recent drive unless you'd rather choose.
