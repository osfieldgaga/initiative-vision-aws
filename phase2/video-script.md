# Video script: Initiative Vision (target 1:50, hard limit 2:00)

The rules ask the video to show **the team, the application working, its architecture, and its principal
results**. One scene each, plus a hook and a close. Narration is ~250 words, about 1:45 at a brisk but
clear ~150 words per minute; the times are cumulative.

**Recording setup:** OBS (or any screen recorder) at 1920x1080. For the demo scene, put the virtual Pi
window on the left half of the screen and `www.initiativelabs.cc/ivision-demo` on the right, with the
instance already started (press Start ~2 minutes before recording). Run:
`python src/scripts/virtual_pi.py test_sessions/private-640/20261022T164127Z-084b --demo --start 165`
(right blinker, then the stop sign at 2:51 of the drive). Record ~40 s of it and cut to the best 30 s.

**Editing tip:** let the screen do the talking. Show each number as a big title card while you say it;
no scene should hold one static shot for more than ~8 s.

---

## 1. Hook and team (0:00-0:12)

**On screen:** you next to the 2014 Malibu, the Pi and camera on the dash. Lower third:
"Osfield Gaga · Initiative Labs".

> My 2014 Malibu has no driver assistance at all. So I gave it a Raspberry Pi, a camera, and the cloud.
> I'm Osfield from Initiative Labs, and this is Initiative Vision.

## 2. It working (0:12-0:52)

**On screen:** split screen, virtual Pi left, results page right. Zoom in on the stop sign when it's flagged.

> The Pi only records and sends a small frame to AWS every second. Here, the virtual Pi replays a real
> drive, with the car's own CAN data: speed, gear, steering, pedals, blinker. On the right, what comes
> back from AWS, just over a second later: the road, the lane lines, the vehicles, a bird's-eye view with my
> position in the lane, and here, a stop sign, read and flagged. Judges can run this exact demo
> themselves, no car and no AWS account needed.

## 3. Architecture (0:52-1:17)

**On screen:** `docs/architecture.png`, slow zoom from the Pi to the EC2 box; 2 s of the CloudWatch
dashboard (`docs/cloudwatch.png`) at the end.

> Each frame goes to S3 and an SQS queue. A Graviton instance running COOL, the Cloud Optimized OpenCV
> image, runs the whole pipeline in OpenCV 5: undistortion, a lane and vehicle network on OpenCV's DNN
> module, a bird's-eye view I calibrated with traffic cones, and a sign reader built from scratch.
> Everything is monitored in CloudWatch.

## 4. Results (1:17-1:47)

**On screen:** title cards, one per line.

> Camera to result: about 1.3 seconds, down from 3.2.
> Inference: 0.74 seconds on 4 Graviton cores, faster than the original PyTorch model, and 0.22 on 16.
> Lanes: when it sees both lines, it's right 98 percent of the time.
> Stop signs: zero false alarms in 8,000 frames.

## 5. Close (1:47-1:57)

**On screen:** a before/after privacy frame for 2 s, then the two URLs: the demo page and the GitHub repo.

> It never controls the car, and shared data is anonymised. Everything is open source. Try it yourself.

---

**If it runs long:** drop the CAN list in scene 2 ("speed, gear, steering, pedals, blinker") and the
CloudWatch sentence in scene 3. Never cut scenes 1-4: they are the four things the rules ask for.
