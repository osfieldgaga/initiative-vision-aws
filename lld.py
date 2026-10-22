import time

import cv2 as cv
import numpy as np
from pathlib import Path


## Camera Intrinsics

imx219_matrix = np.array([[908.3685, 0, 647.8779],
[0, 912.6167, 576.7830],
[0, 0, 1.0]], dtype=float)

imx219_dist = np.array([-0.3404, 0.1337, 0, 0 ,0])

## Perpective warping params
src = np.float32([
    [134.8,  582.5],   # near-left cone's base, read off the photo
    [602, 599],  # near-right cone's base
    [493.7,  551],   # far-right cone's base
    [265.4,   544],    # far-left cone's base
])

scale = 100  # px per unit, same convention you used for the rectangle itself

dst = np.float32([
    [1 * scale, 7 * scale],   # lower-left
    [7 * scale, 7 * scale],   # lower-right
    [7 * scale, 1 * scale],   # upper-right
    [1 * scale, 1 * scale],   # upper-left
])

M = cv.getPerspectiveTransform(src, dst, solveMethod=0)

canvas_w = 8 * scale   # 1 unit padding on each side of a 6-unit-wide rectangle
canvas_h = 8 * scale   # 1 unit padding on each side of a 6-unit-deep rectangle

car_center = np.float32([4 * scale, 700])

## Lane pipeline params
MODEL_SIZE = 640          # YOLOPv2 input, frames are resized to MODEL_SIZE x MODEL_SIZE
LANE_THRESHOLD = 0.65     # lane-line logit threshold
MIN_CONTOUR_AREA = 60
TOLERANCE = 70            # px; contours within this of the nearest line's anchor are the same (dashed) line
DEGREE = 2                # polyfit degree, x = f(y)
LOOKAHEAD_Y = 400         # BEV row where lane positions / offset are measured
DEPARTURE_THRESHOLD = 0.4 # m; ~ (3.5 m lane - 1.8 m car) / 2 * 0.5

## Object detection params (YOLOPv2's detection head, YOLOv7-style)
# Trained on BDD100K with all traffic objects merged into one class: it finds vehicles only (every
# box comes out as class 3 whatever the object), no people, signs or traffic lights.
DET_CONF = 0.3            # objectness * class score, same defaults as the YOLOPv2 demo
DET_IOU = 0.45            # NMS overlap threshold
DET_LABEL = "vehicle"
DET_STRIDES = (8, 16, 32)
DET_ANCHORS = np.array([  # px at MODEL_SIZE, 3 anchors per stride (the model's anchor grids)
    [[12, 16], [19, 36], [40, 28]],
    [[36, 75], [76, 55], [72, 146]],
    [[142, 110], [192, 243], [459, 401]],
], dtype=np.float32)

DEFAULT_WEIGHTS = "data/weights/yolopv2.onnx"  # cv.dnn; "data/weights/yolopv2.pt" = TorchScript
DNN_OUTPUTS = ["da", "ll", "det8", "det16", "det32"]  # names set by src/scripts/export_onnx.py
# Classic engine + FP16 target: 0.74 s/frame on a c8g.xlarge (COOL, Graviton4) vs 1.20 s for the
# new engine and 0.97 s classic FP32. FP16 only takes effect on ARMv8; elsewhere OpenCV warns and
# falls back to FP32 (on x86 the new engine is faster: 0.41 vs 0.74 s on the dev laptop).
DNN_ENGINE = cv.dnn.ENGINE_CLASSIC
DNN_TARGET = cv.dnn.DNN_TARGET_CPU_FP16


def load_model(weights_path=DEFAULT_WEIGHTS, device=None):
    """Load YOLOPv2. Returns (model, device).

    .onnx -> OpenCV DNN (cv.dnn.Net, device "opencv-cpu"), made by src/scripts/export_onnx.py.
    Anything else -> TorchScript on CUDA if available, else CPU. torch is only imported here,
    so the ONNX path runs without it.
    """
    if str(weights_path).endswith(".onnx"):
        net = cv.dnn.readNetFromONNX(str(weights_path), engine=DNN_ENGINE)
        net.setPreferableTarget(DNN_TARGET)
        return net, "opencv-cpu"

    import torch
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.jit.load(weights_path, map_location=device)
    model.eval()
    return model, device


# getOptimalNewCameraMatrix only depends on frame size, so compute it once per size
_undistort_cache = {}

def _undistort_params(w, h):
    if (w, h) not in _undistort_cache:
        _undistort_cache[(w, h)] = cv.getOptimalNewCameraMatrix(imx219_matrix, imx219_dist, (w, h), 1, (w, h))
    return _undistort_cache[(w, h)]


def _infer_dnn(net, bgr):
    """OpenCV DNN forward pass. Returns (drivable mask, lane mask, detection maps, seconds):
    masks 0/255 uint8, detection maps the three raw 1x255xHxW arrays for decode_detections."""
    # BGR -> RGB, HWC -> NCHW, 0..255 -> 0..1 in one call
    blob = cv.dnn.blobFromImage(bgr, scalefactor=1 / 255.0, swapRB=True)
    t_infer = time.perf_counter()
    net.setInput(blob)
    # OpenCV 5's new engine wants every output requested at once (the classic one doesn't mind)
    da, ll, *det_maps = net.forward(DNN_OUTPUTS)
    infer_s = time.perf_counter() - t_infer

    da_mask = da[0].argmax(axis=0).astype(np.uint8) * 255
    ll_mask = (ll[0, 0] > LANE_THRESHOLD).astype(np.uint8) * 255
    return da_mask, ll_mask, det_maps, infer_s


def _infer_torch(model, device, bgr):
    """TorchScript forward pass, same contract as _infer_dnn."""
    import torch

    rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)

    # change frame to tensor for YOLO
    img_tensor = rgb.transpose(2, 0, 1)
    img_tensor = np.ascontiguousarray(img_tensor)
    img_tensor = torch.from_numpy(img_tensor).to(device).float()
    img_tensor /= 255.0

    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        t_infer = time.perf_counter()
        detections, driveable_area, lld = model(img_tensor)
        if img_tensor.is_cuda:
            torch.cuda.synchronize()  # CUDA runs async; wait so the timing is real
        infer_s = time.perf_counter() - t_infer

    # Extract binary masks for drivable area and lanes
    da_mask = driveable_area.argmax(dim=1)[0].cpu().numpy().astype(np.uint8) * 255
    ll_mask = (lld[0, 0] > LANE_THRESHOLD).cpu().numpy().astype(np.uint8) * 255
    det_maps = [m.cpu().numpy() for m in detections[0]]
    return da_mask, ll_mask, det_maps, infer_s


def decode_detections(det_maps, conf=DET_CONF, iou=DET_IOU):
    """Raw YOLOPv2 detection maps -> list of {"label", "score", "box"}, after NMS.

    box is [x1, y1, x2, y2] in MODEL_SIZE pixels (the overlay image), sorted nearest first
    (largest bottom y). Each map is 1 x (3 anchors * 85) x H x W: per anchor x, y, w, h,
    objectness, then 80 class scores, all before the sigmoid.
    """
    rows = []
    for raw, anchors, stride in zip(det_maps, DET_ANCHORS, DET_STRIDES):
        _, _, ny, nx = raw.shape
        p = raw.reshape(3, 85, ny, nx).transpose(0, 2, 3, 1)  # anchor, y, x, channel
        p = 1 / (1 + np.exp(-p))
        gy, gx = np.mgrid[0:ny, 0:nx]
        xy = (p[..., 0:2] * 2 - 0.5 + np.stack((gx, gy), axis=-1)) * stride  # box center
        wh = (p[..., 2:4] * 2) ** 2 * anchors[:, None, None, :]
        rows.append(np.concatenate([xy, wh, p[..., 4:]], axis=-1).reshape(-1, 85))
    p = np.concatenate(rows)
    p = p[p[:, 4] > conf]

    scores = p[:, 4] * p[:, 5:].max(axis=1)
    p, scores = p[scores > conf], scores[scores > conf]
    boxes = np.column_stack([p[:, 0:2] - p[:, 2:4] / 2, p[:, 2:4]])  # x, y, w, h for NMSBoxes
    keep = cv.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf, iou)

    objects = []
    for i in np.array(keep, dtype=int).flatten():
        x, y, w, h = boxes[i]
        objects.append({
            "label": DET_LABEL,
            "score": round(float(scores[i]), 3),
            "box": [round(float(v), 1) for v in (x, y, x + w, y + h)],
        })
    objects.sort(key=lambda o: -o["box"][3])
    return objects


def undistort_frame(frame):
    """Raw camera frame -> undistorted, cropped to the valid area, rotated upright (full res)."""
    h, w = frame.shape[:2]

    new_calib_mtx, roi = _undistort_params(w, h)

    # undistort
    undistorted_frame = cv.undistort(frame, imx219_matrix, imx219_dist, None, new_calib_mtx)

    # then crop
    x, y, w, h = roi
    undistorted_frame = undistorted_frame[y:y+h, x:x+w]

    # only rotate after undistorting to preserve calibration
    undistorted_r90 = cv.rotate(undistorted_frame, cv.ROTATE_90_CLOCKWISE)

    return undistorted_r90


def process_frame(frame, model, device, sign_reader=None):
    """Run the full lane pipeline on one raw BGR camera frame.

    sign_reader: a signs.SignReader to also look for stop / speed-limit signs (None = skip).

    Returns (metrics, overlay, warped):
      metrics -- dict of plain Python types (JSON-serializable)
      overlay -- 640x640 annotated camera view
      warped  -- 800x800 annotated bird's-eye view
    """
    undistorted_r90 = undistort_frame(frame)

    undistorted_resized = cv.resize(undistorted_r90, (MODEL_SIZE, MODEL_SIZE), interpolation=cv.INTER_AREA)
    if isinstance(model, cv.dnn.Net):
        da_mask, ll_mask, det_maps, infer_s = _infer_dnn(model, undistorted_resized)
    else:
        da_mask, ll_mask, det_maps, infer_s = _infer_torch(model, device, undistorted_resized)

    objects = decode_detections(det_maps)

    # signs are searched at full resolution (they're small), then boxes scaled to the overlay
    found_signs, signs_s = [], 0.0
    if sign_reader is not None:
        t_signs = time.perf_counter()
        found_signs = sign_reader.detect(undistorted_r90)
        signs_s = time.perf_counter() - t_signs
        sx = MODEL_SIZE / undistorted_r90.shape[1]
        sy = MODEL_SIZE / undistorted_r90.shape[0]
        for sign in found_signs:
            x1, y1, x2, y2 = sign["box"]
            sign["box"] = [round(x1 * sx, 1), round(y1 * sy, 1), round(x2 * sx, 1), round(y2 * sy, 1)]

    # Resize masks back to the frame size for overlay
    da_mask_resized = cv.resize(da_mask, (MODEL_SIZE, MODEL_SIZE), interpolation=cv.INTER_AREA)
    ll_mask_resized = cv.resize(ll_mask, (MODEL_SIZE, MODEL_SIZE), interpolation=cv.INTER_AREA)

    # Colorize the masks (BGR) and blend onto the frame
    drivable_color = np.zeros_like(undistorted_resized)
    lane_color = np.zeros_like(undistorted_resized)

    drivable_color[da_mask_resized == 255] = [0, 255, 0]
    lane_color[ll_mask_resized == 255] = [0, 0, 255]

    overlay = cv.addWeighted(undistorted_resized, 1.0, drivable_color, 0.3, 0)
    overlay = cv.addWeighted(overlay, 1.0, lane_color, 0.6, 0)


    # turn lane zones into curves
    frame_h = ll_mask_resized.shape[0]

    roi_top = int(frame_h * 0.75)
    roi_bottom = int(frame_h * 0.9)

    ll_mask_resized[roi_bottom:, :] = 0
    ll_mask_resized[:roi_top, :] = 0

    ll_mask_warped = cv.warpPerspective(ll_mask_resized, M, (canvas_w, canvas_h))

    contours, _ = cv.findContours(ll_mask_warped, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    center_x = canvas_w / 2

    candidates = []
    for contour in contours:
        if cv.contourArea(contour) < MIN_CONTOUR_AREA:
            continue
        xs = contour[:, :, 0].flatten()
        ys = contour[:, :, 1].flatten()
        bottom_y = ys.max()
        anchor_x = xs[ys > bottom_y - 15].mean()  # this contour's x near its bottom (nearest the car)
        candidates.append((anchor_x, xs, ys))

    left_candidates = [c for c in candidates if c[0] < center_x]
    right_candidates = [c for c in candidates if c[0] >= center_x]

    left_x, left_y = [], []
    if left_candidates:
        # largest c[0] (anchor) is the left lane
        nearest_left_anchor = max(c[0] for c in left_candidates)  # closest to center, from the left
        for anchor_x, xs, ys in left_candidates:
            if (abs(anchor_x - nearest_left_anchor) <= TOLERANCE
                and len(xs) > 50 # not too few points
                and (ys.max() - ys.min()) > 100 # extends far enough
            ):
                left_x.extend(xs)
                left_y.extend(ys)

    right_x, right_y = [], []
    if right_candidates:
        # smallest c[0] (anchor) is the right lane
        nearest_right_anchor = min(c[0] for c in right_candidates)  # closest to center, from the right
        for anchor_x, xs, ys in right_candidates:
            if (abs(anchor_x - nearest_right_anchor) <= TOLERANCE
                and len(xs) > 50 # not too few points
                and (ys.max() - ys.min()) > 100 # extends far enough
            ):
                right_x.extend(xs)
                right_y.extend(ys)

    left_x_ahead = None   # lane x at LOOKAHEAD_Y, None if that side wasn't found
    right_x_ahead = None

    warped = cv.warpPerspective(overlay, M, (canvas_w, canvas_h))

    a_left = 0
    a_right = 0

    left_coeff = None
    right_coeff = None

    for lane, xs, ys, color in ((0, left_x, left_y, (255, 0, 128)), (1, right_x, right_y, (128, 0, 255))):
        if len(xs) < DEGREE + 1:
            continue

        xs = np.array(xs)
        ys = np.array(ys)

        coefficients = np.polyfit(ys, xs, DEGREE)          # x = f(y)

        y_range = np.linspace(ys.min(), ys.max(), num=50)
        x_fit = np.polyval(coefficients, y_range)

        if(lane == 0): # left
            left_x_ahead = np.polyval(coefficients, LOOKAHEAD_Y)
            left_coeff = coefficients
            a_left = coefficients[0]
        else:
            right_x_ahead = np.polyval(coefficients, LOOKAHEAD_Y)
            right_coeff = coefficients
            a_right = coefficients[0]

        curve_points = np.vstack((x_fit, y_range)).T.reshape((-1, 1, 2)).astype(np.int32)
        cv.polylines(warped, [curve_points], isClosed=False, color=color, thickness=2, lineType=cv.LINE_AA)

    both_lanes_present = left_coeff is not None and right_coeff is not None

    def to_m(x):
        return None if x is None else float((x - canvas_w / 2) / scale)

    def fmt_m(v):
        return "n/a" if v is None else f"{v:.2f} m"

    # each side's offset needs that side; the center offset and width need both. A missing side
    # used to count as x=0 (the canvas edge), which showed up as a ~-4 m offset.
    left_lane_offset_from_center = to_m(left_x_ahead)
    right_lane_offset_from_center = to_m(right_x_ahead)
    lane_center_x = lat_off_warped = lane_width = None
    if both_lanes_present:
        lane_center_x = (left_x_ahead + right_x_ahead) / 2
        lat_off_warped = float((car_center[0] - lane_center_x) / scale)
        lane_width = abs(left_lane_offset_from_center) + abs(right_lane_offset_from_center)

    cv.putText(overlay, f"Lane center offset: {fmt_m(lat_off_warped)}", (20, 20), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))

    # bool(): `and` can hand back a numpy.bool_, which json can't serialize
    lane_valid = bool(both_lanes_present and abs(a_left - a_right) < 0.1 and lane_width < 6)
    cv.putText(overlay, f"Valid lanes: {lane_valid}", (20, 110), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255) if not lane_valid else (0,255,0))

    cv.putText(overlay, f"L off: {fmt_m(left_lane_offset_from_center)} | R off: {fmt_m(right_lane_offset_from_center)}", (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))
    cv.putText(overlay, f"Road width: {fmt_m(lane_width)}", (20, 80), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))

    # lane departure: |offset| beyond threshold = departing, sign gives the side
    departure = None
    if lane_valid and abs(lat_off_warped) > DEPARTURE_THRESHOLD:
        departure = "RIGHT" if lat_off_warped > 0 else "LEFT"  # positive offset = car right of lane center

    if not lane_valid:
        departure_text, departure_color = "Departure: n/a", (128, 128, 128)
    elif departure:
        departure_text, departure_color = f"DEPARTING {departure}", (0, 0, 255)
    else:
        departure_text, departure_color = "In lane", (0, 255, 0)
    cv.putText(overlay, departure_text, (20, 140), cv.FONT_HERSHEY_SIMPLEX, 1, departure_color)

    # detected objects: boxes on the camera view, count in the text block
    for obj in objects:
        x1, y1, x2, y2 = (int(v) for v in obj["box"])
        cv.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
        if x2 - x1 >= 40:  # distant boxes are too small for a readable label; the live page lists them all
            cv.putText(overlay, f"{obj['label']} {obj['score']:.2f}", (x1, max(y1 - 4, 10)),
                       cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv.LINE_AA)
    cv.putText(overlay, f"Objects: {len(objects)}", (20, 170), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))

    # signs: red box for STOP, white for speed limits, and what was read
    sign_texts = []
    for sign in found_signs:
        text = "STOP" if sign["type"] == "stop" else f"LIMIT {sign['value']}"
        color = (0, 0, 255) if sign["type"] == "stop" else (255, 255, 255)
        x1, y1, x2, y2 = (int(v) for v in sign["box"])
        cv.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv.putText(overlay, text, (x1, max(y1 - 4, 10)), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv.LINE_AA)
        sign_texts.append(text)
    if sign_texts:
        cv.putText(overlay, "Signs: " + ", ".join(sign_texts), (20, 200), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

    # closest point the camera can see right after the hood
    cv.line(warped, (0, 700), (800, 700), (255, 255, 0), 2)

    #center of the car
    cv.drawMarker(warped, (int(car_center[0]), int(car_center[1])), color=(0, 0, 255), markerType=cv.MARKER_CROSS, markerSize=20, thickness=2)

    # center of the car projected ahead, at the lookahead row
    cv.drawMarker(warped, (int(canvas_w / 2), LOOKAHEAD_Y), color=(128, 128, 128), markerType=cv.MARKER_CROSS, markerSize=20, thickness=2)

    # lane center both at the lookahead row and at the hood
    if lane_center_x is not None:
        cv.drawMarker(warped, (int(lane_center_x), LOOKAHEAD_Y), color=(128, 255, 128), markerType=cv.MARKER_CROSS, markerSize=20, thickness=2)
        cv.drawMarker(warped, (int(lane_center_x), 700), color=(128, 255, 128), markerType=cv.MARKER_CROSS, markerSize=20, thickness=2)

    metrics = {
        # None when the lane line(s) they need weren't detected
        "offset_m": lat_off_warped,
        "left_off_m": left_lane_offset_from_center,
        "right_off_m": right_lane_offset_from_center,
        "road_width_m": lane_width,
        "left_present": left_coeff is not None,
        "right_present": right_coeff is not None,
        "left_coeff": [float(c) for c in left_coeff] if left_coeff is not None else None,
        "right_coeff": [float(c) for c in right_coeff] if right_coeff is not None else None,
        "lane_valid": lane_valid,
        "departure": departure,  # None, "LEFT" or "RIGHT"
        "departure_text": departure_text,
        "objects": objects,  # [{"label", "score", "box": [x1, y1, x2, y2] in overlay px}], nearest first
        "signs": found_signs,  # [{"type": "stop" | "speed_limit", "value", "score", "text", "box"}], see signs.py
        "infer_s": round(infer_s, 3),  # model forward pass only
        "signs_s": round(signs_s, 3),  # sign search + OCR
    }

    return metrics, overlay, warped


def format_metrics(metrics):
    def m(key):
        v = metrics[key]
        return "n/a" if v is None else f"{v:.2f} m"

    return (
        f"Offset: {m('offset_m')} | "
        f"L off: {m('left_off_m')} | R off: {m('right_off_m')} | "
        f"Road width: {m('road_width_m')} | "
        f"Valid: {metrics['lane_valid']} | "
        f"{metrics['departure_text']}"
    )


def main():
    model, device = load_model()

    # fourcc = cv.VideoWriter_fourcc(*'mp4v')
    # out = cv.VideoWriter('yolopv2_output.mp4', fourcc, 2.0, (640, 640))  # size must match every frame you write, exactly
    # bev = cv.VideoWriter('bev.mp4', fourcc, 2.0, (800, 800))  # size must match every frame you write, exactly

    # remote_path = Path("/home/osfield/osfield@creampie/Desktop/initiative-vision/latest_frame.jpg")
    # session_path = Path("/home/osfield/osfield@creampie/Desktop/initiative-vision/sessions/20261001T170034Z-d015/frames")
    session_path = Path("./test_sessions/20261001T170034Z-d015/frames")

    frame_names = sorted(item.name for item in session_path.iterdir() if item.is_file())

    START_INDEX = 10000  # skip ahead in the session

    while True:
        for index, frame_itr in enumerate(frame_names):
            if index < START_INDEX:
                continue

            frame = cv.imread(str(session_path / frame_itr))
            if frame is None:
                print(f"Frame {index}: could not read {frame_itr}, skipping")
                continue

            metrics, overlay, warped = process_frame(frame, model, device)

            cv.imshow('YOLOPv2 Panoptic Perception Overlay', overlay)
            # out.write(overlay)
            cv.imshow("Bird Eye's View - Lane Mask", warped)
            # bev.write(warped)

            print(f"Frame {index}/{len(frame_names)} | {format_metrics(metrics)}")

            if cv.waitKey(20) & 0xFF == ord('q'):
                cv.destroyAllWindows()
                return


if __name__ == "__main__":
    main()
