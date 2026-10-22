#!/usr/bin/env python3
"""Hand-label lane visibility and car position on sampled frames, then score the pipeline.

    python src/scripts/label_lanes.py [label] <session_dir> [--n 150]   # one key per frame (default step)
    python src/scripts/label_lanes.py score <session_dir> [--size 640]

label: shows evenly spaced frames (undistorted and upright, no pipeline output, so the answer
can't steer you). When the session has decoded.csv, only frames where the car moves faster than
--min-mph are sampled (a parked car says little about lanes). One key per frame:

    c  both lines of our lane clearly visible, car centred
    l  both visible, car drifting left (well off centre, towards or over the left line)
    r  both visible, car drifting right
    o  only one line, or lines too faded / hidden to place both
    n  no lane lines (intersection, parking lot, unmarked road)
    s  skip (can't tell)          u / backspace  undo the last label          q  save and quit

Labels go to <session_dir>/lane_labels.csv after every key, so quitting and re-running resumes.

score: runs lld.process_frame on each labelled frame (resized to --size first, 640 = what the Pi
uploads) and compares:
  - lane_valid (both lines found, consistent) against c / l / r = both lines visible
  - departure (more than lld.DEPARTURE_THRESHOLD off centre) against l / r, on frames where both
    agree the lane is visible
Writes <session_dir>/lane_eval/: results.json and failure-case images (overlay + bird's-eye view)
for every disagreement, named by kind, for the report.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lld  # noqa: E402
from virtual_pi import CanTrack, first_timestamp_ns, load_frames  # noqa: E402

KEYS = {"c": "centred", "l": "left", "r": "right", "o": "one_line", "n": "no_lines", "s": "skip"}
HELP = ["c centred   l drifting left   r drifting right",
        "o only one line / faded   n no lane lines   s skip",
        "u undo   q save and quit"]
VISIBLE = {"centred", "left", "right"}


def sample(session_dir, n, min_mph):
    """n evenly spaced (seq, path, t) frames, only while moving if CAN speed is available."""
    frames = load_frames(session_dir)
    decoded = Path(session_dir) / "decoded.csv"
    if decoded.exists() and min_mph > 0:
        can = CanTrack(decoded, first_timestamp_ns(session_dir))
        moving = [f for f in frames if (can.at(f[2])["speed_mph"] or 0) >= min_mph]
        print(f"{len(moving)} of {len(frames)} frames at >= {min_mph} mph")
        frames = moving or frames
    idx = np.linspace(0, len(frames) - 1, min(n, len(frames))).round().astype(int)
    return [frames[i] for i in dict.fromkeys(idx)]


def read_labels(path):
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {r["frame"]: r["label"] for r in csv.DictReader(f)}


def write_labels(path, labels):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "label"])
        w.writerows(labels.items())


def label(args):
    session = Path(args.session_dir)
    out = session / "lane_labels.csv"
    labels = read_labels(out)
    todo = sample(session, args.n, args.min_mph)
    history = []
    k = 0
    while k < len(todo):
        seq, path, t = todo[k]
        name = Path(path).name
        if name in labels:
            k += 1
            continue
        view = lld.undistort_frame(cv.imread(str(path)))
        view = cv.resize(view, (720, round(view.shape[0] * 720 / view.shape[1])), interpolation=cv.INTER_AREA)
        bar = np.full((96, view.shape[1], 3), 24, np.uint8)
        done = sum(1 for f in todo if Path(f[1]).name in labels)
        cv.putText(bar, f"{done}/{len(todo)} labelled   frame #{seq}  t={t:.0f}s", (12, 20),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv.LINE_AA)
        for i, line in enumerate(HELP):
            cv.putText(bar, line, (12, 44 + 22 * i), cv.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv.LINE_AA)
        cv.imshow("label lanes", np.vstack([view, bar]))
        key = cv.waitKey(0) & 0xFF
        ch = chr(key) if key < 128 else ""
        if ch == "q" or key == 27:
            break
        if ch == "u" or key == 8:
            if history:
                last = history.pop()
                labels.pop(last, None)
                write_labels(out, labels)
                k = next(i for i, f in enumerate(todo) if Path(f[1]).name == last)
            continue
        if ch in KEYS:
            labels[name] = KEYS[ch]
            history.append(name)
            write_labels(out, labels)
            k += 1
    cv.destroyAllWindows()
    print(f"{len(labels)} labels in {out}")


def ratio(a, b):
    return round(a / b, 3) if b else None


def score(args):
    session = Path(args.session_dir)
    labels = {f: l for f, l in read_labels(session / "lane_labels.csv").items() if l != "skip"}
    if not labels:
        raise SystemExit("no labels yet: run the label step first")
    out = session / "lane_eval"
    (out / "failures").mkdir(parents=True, exist_ok=True)
    model, device = lld.load_model()
    by_name = {Path(p).name: p for _, p, _ in load_frames(session)}

    tp = fp = fn = tn = 0
    dep = {"agree": 0, "total": 0}
    confusion = {}  # (truth position, predicted departure) -> count
    failures = []
    for name, truth in sorted(labels.items()):
        frame = cv.imread(str(by_name[name]))
        if args.size:
            frame = cv.resize(frame, (args.size, args.size), interpolation=cv.INTER_AREA)
        metrics, overlay, warped = lld.process_frame(frame, model, device)
        visible, valid = truth in VISIBLE, metrics["lane_valid"]
        kind = None
        if visible and valid:
            tp += 1
            want = {"centred": None, "left": "LEFT", "right": "RIGHT"}[truth]
            got = metrics["departure"]
            dep["total"] += 1
            dep["agree"] += want == got
            confusion[f"{truth} -> {got or 'in lane'}"] = confusion.get(f"{truth} -> {got or 'in lane'}", 0) + 1
            if want != got:
                kind = "wrong_departure"
        elif visible:
            fn += 1
            kind = "missed_lane"
        elif valid:
            fp += 1
            kind = "false_lane"
        else:
            tn += 1
        if kind:
            failures.append({"frame": name, "kind": kind, "label": truth, "lane_valid": valid,
                             "departure": metrics["departure"], "offset_m": metrics["offset_m"]})
            side = cv.resize(warped, (overlay.shape[0], overlay.shape[0]))
            cv.imwrite(str(out / "failures" / f"{kind}_{Path(name).stem}.jpg"), np.hstack([overlay, side]))

    results = {
        "session": session.name, "frames_labelled": len(labels), "input_size": args.size or "as recorded",
        "label_counts": {k: list(labels.values()).count(k) for k in sorted(set(labels.values()))},
        "lane_valid": {"true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn,
                       "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
                       "accuracy": ratio(tp + tn, tp + fp + fn + tn)},
        "departure_on_valid_frames": {"accuracy": ratio(dep["agree"], dep["total"]), "frames": dep["total"],
                                      "confusion": confusion},
        "failures": failures,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2))
    lv = results["lane_valid"]
    print(f"{len(labels)} labelled frames ({results['label_counts']})")
    print(f"lane_valid: precision {lv['precision']}  recall {lv['recall']}  accuracy {lv['accuracy']}  "
          f"(TP {tp} FP {fp} FN {fn} TN {tn})")
    print(f"departure, where both see the lane: {dep['agree']}/{dep['total']} agree; {confusion}")
    print(f"{len(failures)} failure images in {out / 'failures'}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("label")
    p.add_argument("session_dir")
    p.add_argument("--n", type=int, default=150)
    p.add_argument("--min-mph", type=float, default=5.0)
    p.set_defaults(fn=label)
    p = sub.add_parser("score")
    p.add_argument("session_dir")
    p.add_argument("--size", type=int, default=640, help="resize frames like the Pi's uploads (0 = as recorded)")
    p.set_defaults(fn=score)
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in ("label", "score", "-h", "--help"):
        argv.insert(0, "label")  # a session dir on its own means: label it
    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
