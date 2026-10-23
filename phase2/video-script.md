# Video script: Initiative Vision (target 4:40, hard limit 5:00)

The rules ask the video to show **the team, the application working, its architecture, and its principal
results**. Each of those has its own scene below. Narration is written to be read at a calm pace
(~150 words per minute); the times are cumulative.

**Recording setup:** OBS (or any screen recorder) at 1920x1080. For the demo scenes, put the virtual Pi
window on the left half of the screen and `www.initiativelabs.cc/ivision-demo` on the right half, with
the instance already started (press Start ~2 minutes before recording). Use the judges' drive:
`python src/scripts/virtual_pi.py test_sessions/private-640/20261022T164127Z-084b --demo --start <s>`.
Useful moments in that drive:

| Drive time | `--start` | What happens |
|---|---|---|
| 2:44-3:01 | 160 | right blinker, then a stop sign at 2:51 (detected on the results page) |
| 4:50-5:09 | 285 | left blinker, then a stop sign at 4:56 |
| 5:27-6:12 | 327 | longest stretch above 20 mph: clean lane tracking, offset plot |

---

## 1. Hook and team (0:00-0:25)

**On screen:** you, in or next to the 2014 Malibu, Pi and camera visible on the dash. Lower-third title:
"Osfield Gaga · Initiative Labs".

**Narration:**
> I'm Osfield, and this is Initiative Labs. My car is a 2014 Chevy Malibu: no lane assist, no sign
> recognition, nothing. So I put a Raspberry Pi and a camera on the dashboard, and let the cloud do the
> seeing. This is Initiative Vision, for the OpenCV AI Competition 2026.

## 2. The device (0:25-0:50)

**On screen:** close-ups: the Pi 3B, the camera, the CAN connection under the dash. Optional: the launcher
terminal starting the recorder and uploader.

**Narration:**
> The Vision 1 is deliberately small: a Raspberry Pi 3 with one gigabyte of RAM. It records the road at
> 1280 by 1280, logs the car's CAN bus, and about once a second sends a small 640-pixel frame to AWS.
> That's all the Pi is allowed to do. Everything else happens in the cloud.

## 3. It working (0:50-1:55)

**On screen:** the virtual Pi (left) and the results page (right), `--start 160`. Let it run through the
right turn and the stop sign at 2:51 (about 25 s of drive), then cut to `--start 327` for ~15 s of
cruising. Point at things as you name them (cursor or a zoom in the edit).

**Narration:**
> For this demo, the virtual Pi replays a real drive from my car, on my laptop, at real speed, and uploads
> it exactly like the car does. On the left, the camera with the car's own data from the CAN bus: speed,
> gear, the steering wheel, the pedals, the blinker. On the right, what comes back from AWS: the drivable
> road in green, the lane lines in red, the vehicles boxed, and a bird's-eye view with my position in the
> lane. Here I'm turning right, and there's the stop sign: the cloud reads it and flags STOP. On a straight
> road, the offset plot tracks how centred I am, and it would warn me if I drifted out of my lane. Judges
> can do exactly this: download the drive, open the demo page, and run the virtual Pi themselves.

## 4. Architecture (1:55-2:45)

**On screen:** `docs/architecture.png`, building up one box at a time (or a slow pan), then a few seconds
of the CloudWatch dashboard (`docs/cloudwatch.png`).

**Narration:**
> Here's the path of every frame. The Pi uploads it to S3 and puts a message on an SQS queue. A Graviton
> EC2 instance running COOL, the Cloud Optimized OpenCV image, picks it up, runs the whole vision pipeline
> in OpenCV 5, and streams the result to the live page, while the videos and per-frame metrics go back to
> S3. Every frame's latency, inference time and lane quality goes to CloudWatch. Each piece has its own
> minimal permissions; the website starts the instance through Vercel's OIDC federation, with no stored
> keys, and the instance switches itself off after twenty idle minutes.

## 5. OpenCV 5 inside (2:45-3:35)

**On screen:** quick cuts, ~8 s each: the cone square calibration photo (`data/images/w6l6.jpg` and its
warped view); an overlay frame with lanes and boxes; a sign crop with the OCR reading ("stoe" -> STOP);
a few lines of `lld.py` / `signs.py` if you like.

**Narration:**
> Inside, it's OpenCV 5 end to end. The frame is undistorted with my camera's calibration. YOLOPv2, a
> lane and vehicle network, runs on OpenCV's DNN module: I exported it to ONNX and checked it gives exactly
> the same output as the original PyTorch model. A homography I calibrated with traffic cones turns the
> lane mask into a bird's-eye view in real metres. And the sign reader is built from scratch: colour and
> shape find the candidates, OpenCV's text recognizer reads them, and instead of trusting the raw OCR, I
> score how likely it is to say "stop" or a legal speed limit. A blurry "stoe" still counts as STOP; random
> shapes never do.

## 6. Results (3:35-4:25)

**On screen:** simple title cards, one per number (or the tables from the report).

**Narration:**
> The results. End to end, from the camera to the result on screen: about one point three seconds, down
> from three point two, once I found that over a second was just S3 notifying Lambda, and had the Pi
> queue its frames directly. Inference: point seven four seconds on four Graviton cores, point two two on
> sixteen, faster than the original PyTorch model. COOL against standard OpenCV on the same machine? An
> honest tie for this model. Lanes, against frames I labelled by hand: when it says it sees both lines,
> it's right ninety-eight percent of the time. Stop signs: zero false alarms in eight thousand frames.
> And it has limits: it misses about a quarter of visible lanes, the detector only knows vehicles, and
> speed limits were only tested on synthetic signs so far.

## 7. Responsible use and what's next (4:25-4:50)

**On screen:** a before/after privacy frame (from the review folder), then the demo page URL and the repo URL.

**Narration:**
> It never controls the car, and the data I share is anonymised: faces and plates blurred, GPS and the VIN
> removed. Next: alerts on a small screen back in the car, getting under a second, and using all this
> data to tune my vehicle simulation. Everything is open source. Thanks for watching.

---

**If it runs long, cut in this order:** scene 2's optional terminal shot; the second demo segment
(`--start 327`); the code close-ups in scene 5. Keep scenes 1, 3, 4 and 6: they are the four things the
rules ask for.
