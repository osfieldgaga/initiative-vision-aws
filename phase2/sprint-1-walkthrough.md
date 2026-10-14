# Phase 2, Sprint 1 — walkthrough

Companion to `sprint-1-aws-fundamentals.md` (the plan and reasoning). This is the "do this, in this order" version. Written for someone new to AWS: every step says what it's for, not just what to click.

**Region: `us-east-1`** throughout — pick it in the console's top-right region dropdown before you start, and don't switch mid-sprint. S3 buckets are region-scoped, EC2 instances are region-scoped, and mixing regions is the single most common source of "it can't find my thing" confusion for someone new to AWS.

**Cost note before you start anything:** your $108 in credits almost certainly covers normal AWS usage (EC2 compute, S3, Lambda) but may **not** cover the COOL Marketplace software fee specifically. Check this first: AWS Console → Billing → **Credits**, and look at what each credit's terms say it applies to. If it excludes Marketplace charges, the COOL portion of this sprint (§4 below) costs real out-of-pocket money — small (the cheapest instance is about $0.01/hr software fee, likely under $1 total for this sprint), but real. Everything before §4 (S3, Lambda, the free-tier EC2 test) should be covered regardless.

I'll ask you to confirm you've checked this before we get to §4.

---

## 0. One-time setup: the AWS CLI and boto3

You'll run everything from a terminal on this machine (or wherever you have this project checked out).

1. Install the AWS CLI:
   ```
   sudo apt install -y awscli     # or: pip install --user awscli
   aws --version                   # confirms it worked
   ```
2. Get an access key: AWS Console (top-right, your account name) → **Security credentials** → **Access keys** → **Create access key**. This gives you an Access Key ID and a Secret Access Key — copy both immediately, the secret is shown only once.
3. Configure the CLI with them:
   ```
   aws configure
   ```
   It asks for the access key, secret key, default region (`us-east-1`), and output format (`json` is fine).
4. Confirm it works:
   ```
   aws sts get-caller-identity
   ```
   This should print your AWS account ID and user ARN — if it does, the CLI can talk to your account.
5. Install boto3 (the Python library the scripts use) in whatever Python environment you'll run them from:
   ```
   pip install boto3        # or pip install --user boto3, or into a venv
   ```

**What an access key actually is**, since this matters for security: it's a long-lived credential that can do anything your IAM user is allowed to do, from any machine that has it. Don't commit it to git, don't paste it into a chat, don't leave it in a script. `aws configure` stores it in `~/.aws/credentials`, outside the project.

---

## 1. S3: create a bucket, upload a real session

1. **Console:** S3 → **Create bucket**. Name it something globally unique (bucket names are unique across *all* of AWS, not just your account) — e.g. `initiative-vision-<your-name>-2026`. Region `us-east-1`. Leave everything else default (block all public access — correct for this project, nothing here should be public).
2. **Look at it empty.** Click into the bucket. Nothing there yet. This is the blank slate.
3. **Upload one file by hand**, through the console's Upload button, just to see the mechanic — pick any single frame from a session folder. Watch it appear. Click it, see its details (size, a way to download it). Delete it once you've seen this — the real upload is next.
4. **Upload a whole real session** with the script I wrote and already tested (`phase2/scripts/upload_session.py`, 8/8 unit tests passing against a mocked S3 client — this run is the first time it talks to real AWS):
   ```
   cd /path/to/initiative-vision
   python3 phase2/scripts/upload_session.py sessions/<a_session_folder> <your-bucket-name>
   ```
   Pick a small-ish session for this first run (not your 17,000-frame 10-minute drive — a shorter test session, if you have one, or accept this will take a few minutes and upload a few thousand small files, which is realistic but not fast on a home connection).
5. **Verify:**
   ```
   aws s3 ls s3://<your-bucket-name>/sessions/ --recursive | head
   aws s3 ls s3://<your-bucket-name>/sessions/ --recursive | wc -l
   ```
   The count should match the number of files in the local session folder.

**What you just learned:** a bucket, an object, a key (the full `sessions/.../frames/....jpg` path — S3 has no real folders, the `/`s are just part of the name), and that "uploading" is really just a PUT request per object — `upload_session.py`'s whole logic is a loop doing exactly that, nothing more.

---

## 2. Trigger: wire up the Lambda

1. **Create the function.** Lambda console → **Create function** → Author from scratch. Name: `on-upload`. Runtime: **Python 3.12**. Architecture: either x86_64 or arm64 is fine for something this small (arm64 is very slightly cheaper per invocation — pick it if you want one more small taste of Graviton, it makes no practical difference here).
2. **Paste the code.** Open `phase2/lambda/on_upload.py` (17 tests passing on its `parse_event`/`processed_key`/`handler` logic already), copy its contents into the Lambda console's code editor, replacing the placeholder. Click **Deploy**.
3. **Give it permission to touch your bucket.** Lambda console → your function → **Configuration** → **Permissions** → click the execution role's name (opens IAM in a new tab) → **Add permissions** → **Create inline policy** → JSON tab, paste:
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Action": ["s3:GetObject", "s3:PutObject"],
         "Resource": "arn:aws:s3:::<your-bucket-name>/*"
       }
     ]
   }
   ```
   Name it, save. (This is broader than a real production policy would be — fine for a learning sprint; tightening this is a "later" concern, not a "right now" one.)
4. **Wire the trigger.** Back in the Lambda console, your function → **Add trigger** → S3 → your bucket → Event type: **All object create events** → **Prefix**: `sessions/` (so uploads to `processed/` don't re-trigger it — belt-and-suspenders with the guard already in the code) → Add.
5. **Test it.** Upload one more file to the bucket under `sessions/` (console upload, or run `upload_session.py` again on a different small session):
   ```
   aws s3 cp some_file.jpg s3://<your-bucket-name>/sessions/test/some_file.jpg
   ```
6. **Verify it actually fired**, two ways:
   - CloudWatch → Log groups → `/aws/lambda/on-upload` → the newest log stream should show `triggered by s3://...` within a few seconds of the upload.
   - `aws s3 ls s3://<your-bucket-name>/processed/sessions/test/` should show the copied file.

**What you just learned:** an event notification (S3's side: "tell someone when this happens"), an event source mapping (the trigger config connecting bucket → function), and an execution role (the permissions a Lambda runs with, separate from your own IAM user's permissions) — the three pieces every AWS "X happens automatically when Y occurs" system is built from, not just this one.

---

## 3. EC2, free tier, no COOL yet

1. EC2 console → **Launch instance**. Name: `learning-t4g`. AMI: search "Ubuntu", pick **Ubuntu Server 24.04 LTS (HVM), arm64**. Instance type: `t4g.micro` (free-tier eligible for a new account's first 12 months, up to 750 hours/month — check the console's own "Free tier eligible" tag to be sure it still applies to your account).
2. **Key pair:** create a new one, download the `.pem` file, `chmod 400 it.pem`. This is how you'll SSH in — anyone with this file can log in as this instance, so it's a secret same as the access key.
3. **Security group:** allow SSH (port 22) from "My IP" (not 0.0.0.0/0 — no reason to expose SSH to the whole internet for a learning box).
4. Launch. Wait ~30s for it to reach "running".
5. SSH in: `ssh -i it.pem ubuntu@<the instance's public IP, shown in the console>`
6. Once in: `uname -m` should print `aarch64` — you're on the same CPU architecture as the Pi, on a completely different machine. That's the whole point of Graviton: it's just ARM, at cloud scale.
7. **Terminate it** (not just stop — terminate, so it's gone and definitely not billing): EC2 console → select it → Instance state → Terminate.

---

## 4. COOL — confirm the credits situation first (see the note at the top), then:

1. **Subscribe.** AWS Marketplace → search "Cloud Optimized OpenCV" → the Graviton4 listing → **Subscribe**. This starts the 7-day trial clock (mark day 6 on a calendar).
2. **Launch from the Marketplace listing**, not the plain EC2 console — this matters, since it's *that specific AMI* that has COOL preinstalled. Instance type: `c8g.large` (cheapest tier the listing shows, ~$0.01/hr software fee + its own small EC2 compute cost — plenty for this test, not the vendor's recommended size, which is fine for proving the workflow rather than measuring peak throughput). Same key pair and "SSH from my IP" security group approach as §3.
3. SSH in once it's running: `ssh -i it.pem ubuntu@<public IP>`
4. **Find the actual venv path** — the marketplace listing says `/opt/cool/venvs/`, with 3.10/3.11/3.12 versions, but I haven't personally confirmed the exact directory names, so check first rather than guess:
   ```
   ls /opt/cool/venvs/
   ```
5. Copy `phase2/scripts/time_cool_vs_stock.py` and a real frame onto the instance (from your laptop, if you're following the earlier "scp to laptop" pattern, or straight from wherever this project lives):
   ```
   scp -i it.pem phase2/scripts/time_cool_vs_stock.py ubuntu@<IP>:~/
   scp -i it.pem sessions/<a_session>/frames/<one_frame>.jpg ubuntu@<IP>:~/frame.jpg
   ```
6. **Run it with COOL active:**
   ```
   source /opt/cool/venvs/<whatever ls showed>/bin/activate
   python3 time_cool_vs_stock.py frame.jpg
   deactivate
   ```
7. **Run it with stock OpenCV, same machine, for the comparison:**
   ```
   pip install --user opencv-python-headless
   python3 time_cool_vs_stock.py frame.jpg
   ```
8. Write down both sets of numbers (`resize_half` and `adaptive_threshold`, mean ms). That comparison — same operation, same frame, same physical machine, only the OpenCV build different — is your first real "Best Use of COOL" data point, however small.
9. **Stop the instance** the moment you're done (Instance state → Stop, or Terminate if you're confident you won't need it again soon — stopping keeps the EBS disk and costs a little for storage; terminating is fully done and cheapest).
10. **Before day 6:** Marketplace → Manage subscriptions → cancel, unless you're continuing straight into Sprint 3.

---

## When you're done

Tell me the numbers from §4 and how each part went — especially anything that didn't match this walkthrough exactly (the COOL venv path is the one detail here I flagged as unconfirmed; if it turns out to be something other than expected, that's useful to know and easy to fix in the docs). Sprint 2 (aligning frames/CAN/IMU by timestamp) is next, and doesn't need anything from this sprint except the S3 upload habit.
