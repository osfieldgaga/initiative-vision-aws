"""Stop-sign and US speed-limit-sign reader: classic OpenCV + a CRNN text recognizer on cv.dnn.

YOLOPv2's detector only finds vehicles, so signs are found from scratch, in three stages:

  1. color   -- STOP: HSV red mask. SPEED LIMIT: Canny edges instead, since a white sign
                against a bright sky has no color edge to segment on; its color is checked in
                stage 2 (bright, unsaturated -- tolerant of this camera's purple cast)
  2. shape   -- STOP: an octagon-ish red blob (6-10 hull vertices, ~square, solid).
                SPEED LIMIT: a closed upright ~4:5 quadrilateral whose inside is mostly bright
                and unsaturated with some dark text
  3. OCR     -- the text band of each surviving candidate goes through OpenCV's CRNN text
                recognizer, and the result is scored against what that sign can say: the word
                "stop", or one of the legal speed values. Scoring uses the CTC probability of
                the expected text (lexicon-constrained decoding) rather than the free-form
                reading, which at 60 px of blurry sign is often "stoe" or "st0p".

Weights: OpenCV model zoo text_recognition_CRNN_EN_2021sep.onnx (32x100 gray in, 24 steps of
blank + 0-9a-z out), https://huggingface.co/opencv/text_recognition_crnn
"""
import cv2 as cv
import numpy as np


OCR_WEIGHTS = "data/weights/ocr/text_recognition_CRNN_EN_2021sep.onnx"
OCR_VOCAB = "0123456789abcdefghijklmnopqrstuvwxyz"  # index 0 of the output is the CTC blank
OCR_SIZE = (100, 32)

SEARCH_TOP = 2 / 3        # signs are above the hood and road: only the top of the frame is searched
MAX_CANDIDATES = 4        # per sign type and frame, largest first -- bounds the OCR cost

# stage 1: color
RED_RANGES = (((0, 110, 60), (10, 255, 255)), ((170, 110, 60), (180, 255, 255)))
WHITE_MAX_SATURATION = 90   # the sign's white reads lavender on this camera

# stage 2: shape
STOP_MIN_AREA = 250       # px at full res; smaller ones are too small to read anyway
STOP_VERTICES = (6, 10)   # approxPolyDP of the hull; a blurred octagon rarely gives exactly 8
STOP_ASPECT = (0.75, 1.35)
SPEED_MIN_AREA = 600
SPEED_MIN_WIDTH = 48      # px; at 40 px "55" was read as "65" even with a confident score -- wait to get closer
SPEED_ASPECT = (1.1, 1.5) # h / w; the R2-1 sign is 24 x 30 in
SPEED_DARK_FRACTION = (0.05, 0.5)   # share of dark (text) pixels inside the sign
SPEED_BRIGHT_FRACTION = 0.4         # share of bright, unsaturated pixels inside the sign
MIN_SOLIDITY = 0.85

# stage 3: OCR, scores are log P(expected text) - log P(best free-form reading), <= ~0
STOP_MIN_SCORE = -5.0     # real signs >= 60 px wide score -4..0; no false candidate scored above -6.9
SPEED_VALUES = [str(v) for v in range(15, 90, 5)]  # two digits only: a lone "5" matched random blobs
SPEED_MIN_SCORE = -5.0
SPEED_MIN_MARGIN = 2.0    # best value must beat the runner-up by this much (55 vs 35, ...),
SPEED_EXACT_MIN_SCORE = -1.0  # ...unless the free-form reading is exactly that value with a score this good
SPEED_EXACT_MIN_WIDTH = 50    # and the sign is this wide


def _ctc_logp(lp, word):
    """log P(word) under CTC, summed over every alignment (forward algorithm).
    lp: T x (1 + len(OCR_VOCAB)) log-probabilities, blank first."""
    lab = np.zeros(2 * len(word) + 1, dtype=int)  # blank, c1, blank, c2, ..., blank
    lab[1::2] = [OCR_VOCAB.index(c) + 1 for c in word]
    can_skip = np.zeros(len(lab), dtype=bool)     # may jump over a blank between different chars
    can_skip[2:] = (lab[2:] != 0) & (lab[2:] != lab[:-2])

    alpha = np.full(len(lab), -np.inf)
    alpha[:2] = lp[0, lab[:2]]
    for t in range(1, lp.shape[0]):
        stay = alpha
        step = np.r_[-np.inf, alpha[:-1]]
        skip = np.where(can_skip, np.r_[-np.inf, -np.inf, alpha[:-2]], -np.inf)
        alpha = np.logaddexp(np.logaddexp(stay, step), skip) + lp[t, lab]
    return float(np.logaddexp(alpha[-1], alpha[-2]))


def _greedy(lp):
    """Best free-form reading and its log-probability (best path)."""
    ids = lp.argmax(axis=1)
    text = "".join(OCR_VOCAB[i - 1] for k, i in enumerate(ids) if i != 0 and (k == 0 or i != ids[k - 1]))
    return text, float(lp.max(axis=1).sum())


def _shape(contour):
    """(area, bounding box, hull vertex count, solidity) of a contour."""
    area = cv.contourArea(contour)
    hull = cv.convexHull(contour)
    vertices = len(cv.approxPolyDP(hull, 0.02 * cv.arcLength(hull, True), True))
    return area, cv.boundingRect(contour), vertices, area / max(cv.contourArea(hull), 1.0)


def _inside(a, b):
    """True if box a (x, y, w, h) lies mostly inside box b."""
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return ix * iy > 0.6 * a[2] * a[3]


class SignReader:
    def __init__(self, weights=OCR_WEIGHTS):
        self.net = cv.dnn.readNetFromONNX(weights)

    ## stage 3 helpers

    def _logprobs(self, gray):
        blob = cv.dnn.blobFromImage(gray, scalefactor=1 / 127.5, size=OCR_SIZE, mean=127.5)
        self.net.setInput(blob)
        out = self.net.forward()[:, 0, :]  # T x classes, raw scores
        out = out - out.max(axis=1, keepdims=True)
        return out - np.log(np.exp(out).sum(axis=1, keepdims=True))

    def _bands(self, crop, channels, bands, invert, side):
        """Contrast-stretched text bands of a sign crop, one per (channel, band) combination,
        with `side` of the width cut off left and right."""
        h, w = crop.shape[:2]
        for ch in channels:
            for top, bottom in bands:
                band = ch[int(h * top):int(h * bottom), int(w * side):int(w * (1 - side))]
                if band.size == 0:
                    continue
                band = cv.normalize(band, None, 0, 255, cv.NORM_MINMAX)
                yield 255 - band if invert else band  # the recognizer reads dark text on light

    def _read_stop(self, crop):
        """White STOP on red: the green channel (or min over channels) makes the letters bright
        and the red dark; inverted that's dark-on-light."""
        best = (-np.inf, "")
        for band in self._bands(crop, (crop[:, :, 1], crop.min(axis=2)), ((0.25, 0.75), (0.3, 0.7), (0.2, 0.8)), True, 0.05):
            lp = self._logprobs(band)
            text, best_path = _greedy(lp)
            score = _ctc_logp(lp, "stop") - best_path
            if score > best[0]:
                best = (score, text)
        return best

    def _read_speed(self, crop):
        """Black digits on white in the lower part of the sign. Returns (score, value, margin, text)."""
        gray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
        best = (-np.inf, None, 0.0, "")
        # side 0.12: the sign's black border stroke must stay out, or it reads as a "1" ("35" -> 15)
        for band in self._bands(crop, (gray,), ((0.52, 0.93), (0.55, 0.9), (0.5, 0.95)), False, 0.12):
            lp = self._logprobs(band)
            text, best_path = _greedy(lp)
            scores = sorted(((_ctc_logp(lp, v) - best_path, v) for v in SPEED_VALUES), reverse=True)
            (top, value), (second, _) = scores[0], scores[1]
            if top > best[0]:
                best = (top, int(value), top - second, text)
        return best

    ## stages 1 + 2

    def _stop_candidates(self, hsv):
        mask = sum(cv.inRange(hsv, lo, hi) for lo, hi in RED_RANGES).astype(np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8))  # fill the letters
        found = []
        for contour in cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]:
            area, (x, y, w, h), vertices, solidity = _shape(contour)
            if (area >= STOP_MIN_AREA and STOP_VERTICES[0] <= vertices <= STOP_VERTICES[1]
                    and STOP_ASPECT[0] < h / w < STOP_ASPECT[1] and solidity > MIN_SOLIDITY):
                found.append((area, (x, y, w, h)))
        return [box for _, box in sorted(found, reverse=True)[:MAX_CANDIDATES]]

    def _speed_candidates(self, hsv, gray):
        edges = cv.Canny(gray, 50, 150)
        edges = cv.dilate(edges, np.ones((3, 3), np.uint8))  # close small gaps in a blurry border
        found = []
        for contour in cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[0]:
            area, (x, y, w, h), vertices, solidity = _shape(contour)
            if not (area >= SPEED_MIN_AREA and w >= SPEED_MIN_WIDTH and vertices == 4
                    and SPEED_ASPECT[0] < h / w < SPEED_ASPECT[1]
                    and solidity > MIN_SOLIDITY):
                continue
            # inside the border: mostly bright unsaturated sign face, plus some dark legend
            inner_g = gray[y + h // 8:y + 7 * h // 8, x + w // 8:x + 7 * w // 8]
            inner_s = hsv[y + h // 8:y + 7 * h // 8, x + w // 8:x + 7 * w // 8, 1]
            if inner_g.size == 0:
                continue
            threshold, _ = cv.threshold(inner_g, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            bright = (inner_g > threshold) & (inner_s < WHITE_MAX_SATURATION)
            dark = (inner_g < threshold).mean()
            if bright.mean() > SPEED_BRIGHT_FRACTION and SPEED_DARK_FRACTION[0] < dark < SPEED_DARK_FRACTION[1]:
                found.append((area, (x, y, w, h)))
        # the inner and outer edge of the same border both pass: keep the outer one
        boxes = [b for _, b in sorted(found, reverse=True)]
        kept = []
        for b in boxes:
            if all(not _inside(b, k) for k in kept):
                kept.append(b)
        return kept[:MAX_CANDIDATES]

    def detect(self, img):
        """Find and read signs in an undistorted, upright BGR frame.

        Returns [{"type": "stop" | "speed_limit", "value": int | None, "score", "text", "box"}],
        box = [x1, y1, x2, y2] in img pixels; text is the free-form OCR reading (for debugging).
        """
        top = img[:int(img.shape[0] * SEARCH_TOP)]
        hsv = cv.cvtColor(top, cv.COLOR_BGR2HSV)
        signs = []

        for x, y, w, h in self._stop_candidates(hsv):
            score, text = self._read_stop(top[y:y + h, x:x + w])
            if score >= STOP_MIN_SCORE:
                signs.append({"type": "stop", "value": None, "score": round(score, 2), "text": text,
                              "box": [x, y, x + w, y + h]})

        gray = cv.cvtColor(top, cv.COLOR_BGR2GRAY)
        for x, y, w, h in self._speed_candidates(hsv, gray):
            score, value, margin, text = self._read_speed(top[y:y + h, x:x + w])
            exact = text == str(value) and score >= SPEED_EXACT_MIN_SCORE and w >= SPEED_EXACT_MIN_WIDTH
            if score >= SPEED_MIN_SCORE and (margin >= SPEED_MIN_MARGIN or exact):
                signs.append({"type": "speed_limit", "value": value, "score": round(score, 2), "text": text,
                              "box": [x, y, x + w, y + h]})
        return signs
