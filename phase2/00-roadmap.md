# Initiative Vision 1 — Phase 2 Sprint Roadmap

Source: `claude/Initiative Vision 1 — Phase 2 (Processing) Project Description.md`. That document is the scope/intent brief; this maps it into a sprint sequence. Nothing below starts until you say go, same as Phase 1.

## How this track is different from Phase 1

You delegated Phase 1 (the Pi-side acquisition code) to me almost entirely — you reviewed and tested, I wrote and iterated. You've said explicitly you want the opposite division of labor here: Phase 2 is where you want to spend your own time, and you want to *get familiar with the tools* (S3, triggers, EC2, COOL), not just receive finished code you didn't build. So my role on this track is different: plan it with you, write starter code and reference material you review and extend yourself, explain the AWS mechanics as we go, and hold off on doing the hands-on AWS work myself unless you ask me to. I won't provision real AWS resources (buckets, instances, IAM roles — anything that costs money or needs your account) without you explicitly asking for that specific thing.

**Current state, checked just now:** no AWS CLI installed, no credentials configured, nothing exists in AWS yet. Genuinely starting from zero.

## Sprint list

| # | Sprint | Maps to brief's timeline | Depends on |
|---|--------|---------------------------|------------|
| 1 | [AWS fundamentals: S3, triggers, EC2, COOL](sprint-1-aws-fundamentals.md) | Week of Sep 29 (the hands-on half) | Phase 1 producing at least one real session |
| 2 | Alignment job: joining frames/CAN/IMU by timestamp | Week of Sep 29 | Sprint 1; Phase 1 CAN sprint done |
| 3 | COOL triage at scale + x86 benchmark | Week of Sep 29 | Sprints 1–2 |
| 4 | CAN-side event detection (brake, steer/yaw) | Week of Sep 29 | Sprint 2 |
| 5 | Candidate queue + MCP tool server + agent decision loop | Week of Oct 6 | Sprints 3–4 |
| 6 | Coverage-driven curation, keep/discard/escalate, trace logging | Week of Oct 6 | Sprint 5 |
| 7 | Pseudo-labeling panels (depth, semantics, motion mask) | Week of Oct 13 | Sprint 6 |
| 8 | Web UI + judge endpoint | Week of Oct 13 | Sprints 6–7 |
| 9 | Evaluation: hand-labeling, precision/recall, COOL numbers | Week of Oct 13 | Sprints 3, 8 |
| 10 | MATLAB tie-in (optional, explicitly not required for submission) | Post-Oct 20 if time allows | Sprint 9 |
| 11 | Report, architecture diagrams, 5-minute video, deploy | Week of Oct 20 | Everything above |

Only **Sprint 1** is written up in detail right now, matching what you asked to start with. The rest are sketched here at the level the brief already specifies; each gets its own full sprint doc, with your input, when we get there — the same "don't plan five moves ahead of what's actually known" approach that worked for Phase 1. Several later sprints depend on open questions the brief itself hasn't answered yet (which LLM/MCP setup, which depth/segmentation models) — those get resolved in the sprint that needs them, not guessed now.

## Sprint 1 exists to answer these questions hands-on, not just read about them

- What is an S3 bucket, actually, and how do you put something in one and get it back out.
- What is an event trigger — specifically, what happens the instant a file lands in S3, and how do you wire something to react to it.
- What Graviton and EC2 actually are as concrete things you launch and pay for, not just names.
- What COOL actually is: **verified today, not assumed** — it's an AWS Marketplace product, delivered as a pre-built Amazon Machine Image (AMI) for Graviton4 instances (the `c8g`/`m8g`/`r8g` families), not a pip package or a container. You launch an EC2 instance *from that AMI*, SSH in, and the accelerated OpenCV build is already there in a Python virtualenv. This is a real, material fact that shapes how every later COOL-using sprint has to work — you can't just `pip install cool` into an existing box.

## Decisions (answered 2026-09-22)

1. **AWS account:** you have one already, no region preference. **Decision: `us-east-1`** — the safest default for a new Marketplace subscription (COOL) and new-service availability generally; it's where AWS usually rolls things out first and least likely to have a regional gap. All S3/EC2/Lambda work for this project uses this region unless a specific later need forces a change.
2. **Cost comfort:** $108 in first-time-user free-tier credits, all of which can go toward this project. **Important, flagged clearly because it changes real money math:** promotional/free-tier credits from AWS commonly **do not cover AWS Marketplace software charges** (COOL's hourly fee is billed as a Marketplace product, separate from the underlying EC2 compute charge) — only standard AWS service usage. This needs checking in your Billing console (Credits page, which shows exactly what a credit applies to) before relying on the $108 to cover COOL specifically. The underlying EC2 compute cost is a normal AWS charge and should be covered; the COOL software fee itself may not be. I'll flag actual expected cost, split into "compute" vs "COOL software fee," before every real launch.
3. **Involvement:** you drive, I prepare. I do not have or ask for your AWS credentials for this track; every command below is written for you to run yourself.
4. **Experience:** new to AWS. Everything below explains what each piece is and why it exists, not just what to type.

## Ground rules for this whole track

- Nothing gets provisioned in real AWS without you asking for that specific resource.
- Every real-money step gets its expected cost stated before you approve it.
- Claims about what a tool (COOL, a specific AWS service) does get verified, not assumed — same standard Phase 1 held for hardware claims.
- Code and docs are written so you can read and extend them yourself, not as a black box.
