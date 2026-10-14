import torch
import cv2 as cv
import numpy as np
from pathlib import Path
import sys
import os


YOLOP_REPO_PATH = "/media/osfield/Data/Dev/YOLOPv2" 
if YOLOP_REPO_PATH not in sys.path:
    sys.path.insert(0, YOLOP_REPO_PATH)

from utils.utils import non_max_suppression

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


## Load Yolo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.jit.load("data/weights/yolopv2.pt", map_location=device)
model.eval()

fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter('yolopv2_output.mp4', fourcc, 2.0, (640, 640))  # size must match every frame you write, exactly
bev = cv.VideoWriter('bev.mp4', fourcc, 2.0, (800, 800))  # size must match every frame you write, exactly

remote_path = Path("/home/osfield/osfield@creampie/Desktop/initiative-vision/latest_frame.jpg")

session_path = Path("/home/osfield/osfield@creampie/Desktop/initiative-vision/sessions/20261001T170034Z-d015/frames")
latest_session_frame = None

frame_names = []

# Loop through all items in the directory
for item in session_path.iterdir():
    if item.is_file():
        frame_names.append(item.name)

frame_names.sort()

while True:
    ## pre-process image
    # remote_path = Path("/home/osfield/osfield@creampie/Desktop/initiative-vision/latest_frame.jpg")

    # session_path = Path("/home/osfield/osfield@creampie/Desktop/initiative-vision/sessions/20260929T191344Z-4993/frames")
    # latest_session_frame = None



    # Loop through all items in the directory
    for index, frame_itr in enumerate(frame_names):
        latest_session_frame = Path(f"/home/osfield/osfield@creampie/Desktop/initiative-vision/sessions/20261001T170034Z-d015/frames/{frame_itr}")

        # latest_session_frame = Path("/home/osfield/osfield@creampie/Desktop/initiative-vision/latest_frame.jpg")

        # print(remote_path)

        # USE_SESSION_REPLAY = 1

        # if (USE_SESSION_REPLAY):
        #     frame = cv.imread(latest_session_frame) 
        # else:
        #     frame = cv.imread(remote_path) 

        if index < 10000:
            continue

        frame = cv.imread(latest_session_frame) 
        h, w = frame.shape[:2]

        new_calib_mtx, roi = cv.getOptimalNewCameraMatrix(imx219_matrix, imx219_dist, (w, h), 1, (w, h))

        # undistort
        undistorted_frame = cv.undistort(frame, imx219_matrix, imx219_dist, None, new_calib_mtx)

        # then crop
        x, y, w, h = roi
        undistorted_frame = undistorted_frame[y:y+h, x:x+w]


        # only rotate after undistorting to preserve calibration
        undistorted_r90 = cv.rotate(undistorted_frame, cv.ROTATE_90_CLOCKWISE)


        undistorted_resized = cv.resize(undistorted_r90, (640, 640), interpolation=cv.INTER_AREA)
        undistorted_rgb = cv.cvtColor(undistorted_resized, cv.COLOR_BGR2RGB)

        # change frame to tensor for YOLO
        img_tensor = undistorted_rgb.transpose(2, 0, 1)
        img_tensor = np.ascontiguousarray(img_tensor)
        img_tensor = torch.from_numpy(img_tensor).to(device).float()
        img_tensor /= 255.0

        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            detections, driveable_area, lld = model(img_tensor)



        # Example: Extract binary masks for drivable area and lanes
        # Take the argmax of the segmentation outputs across the channel dimension
        da_mask = driveable_area.argmax(dim=1)[0].cpu().numpy().astype(np.uint8) * 255
        ll_mask = (lld[0, 0] > 0.65).cpu().numpy().astype(np.uint8) * 255

        # Resize masks back to your original image size for display/overlay
        da_mask_resized = cv.resize(da_mask, (640, 640), interpolation=cv.INTER_AREA)
        ll_mask_resized = cv.resize(ll_mask, (640, 640), interpolation=cv.INTER_AREA)

        # 5. Display the result
        # cv.imshow('Drivable Area Mask', da_mask_resized)
        # cv.imshow('Lane Line Mask', ll_mask_resized)


        # 1. Create empty colored layers (same size as your image)
        drivable_color = np.zeros_like(undistorted_resized)
        lane_color = np.zeros_like(undistorted_resized)

        # 2. Colorize the masks (OpenCV uses BGR format)
        # Set Drivable Area to Green [BGR: 0, 255, 0] where the mask is white
        drivable_color[da_mask_resized == 255] = [0, 255, 0]

        # Set Lane Lines to Red [BGR: 0, 0, 255] where the mask is white
        lane_color[ll_mask_resized == 255] = [0, 0, 255]

        # 3. Blend the colored layers onto your original frame
        # Change 0.3 to adjust transparency (lower = more see-through)
        overlay = cv.addWeighted(undistorted_resized, 1.0, drivable_color, 0.3, 0)
        overlay = cv.addWeighted(overlay, 1.0, lane_color, 0.6, 0)


        # turn lane zones into curves
        
       
        frame_w = ll_mask_resized.shape[1]
        frame_h = ll_mask_resized.shape[0]

        roi_top = int(frame_h * 0.75) #0.75
        roi_bottom = int(frame_h * 0.9)

        ll_mask_resized[roi_bottom:, :] = 0
        ll_mask_resized[:roi_top, :] = 0 # Erase everything in the top half

        ll_mask_warped = cv.warpPerspective(ll_mask_resized, M, (canvas_w, canvas_h))

        contours, _ = cv.findContours(ll_mask_warped, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        

        frame_w = ll_mask_resized.shape[1]

        center_x = canvas_w / 2

        candidates = []
        for contour in contours:
            if cv.contourArea(contour) < 60:
                continue
            xs = contour[:, :, 0].flatten()
            ys = contour[:, :, 1].flatten()
            bottom_y = ys.max()
            anchor_x = xs[ys > bottom_y - 15].mean()  # this contour's x near its bottom (nearest the car)
            candidates.append((anchor_x, xs, ys))

        left_candidates = [c for c in candidates if c[0] < center_x]
        right_candidates = [c for c in candidates if c[0] >= center_x]

        TOLERANCE = 70  # px; contours within this of the nearest line's anchor are the same (dashed) line

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

        degree = 2

        left_x_ahead = 0 
        right_x_ahead  = 0
        lane_center_x = 0

        warped = cv.warpPerspective(overlay, M, (canvas_w, canvas_h))

        a_left = 0
        a_right = 0

        both_lanes_present = False
        left_coeff = None
        right_coeff = None

        for lane, xs, ys, color in ((0, left_x, left_y, (255, 0, 128)), (1, right_x, right_y, (128, 0, 255))):
            if len(xs) < degree + 1:
                continue

            xs = np.array(xs)
            ys = np.array(ys)

            coefficients = np.polyfit(ys, xs, degree)          # x = f(y)

            # if lane == 0:
            #     a_left = coefficients[0]
            # else:
            #     a_right = coefficients[0]

            # if (abs(a_left - a_right) > 0.1): continue

            y_range = np.linspace(ys.min(), ys.max(), num=50)
            x_fit = np.polyval(coefficients, y_range)


            if(lane == 0): # left
                left_x_ahead = np.polyval(coefficients, 400)
                left_coeff = coefficients
                a_left = coefficients[0]
            else:
                right_x_ahead = np.polyval(coefficients, 400)
                right_coeff = coefficients
                a_right = coefficients[0]


            curve_points = np.vstack((x_fit, y_range)).T.reshape((-1, 1, 2)).astype(np.int32)
            cv.polylines(warped, [curve_points], isClosed=False, color=color, thickness=2, lineType=cv.LINE_AA)
        # cv.drawContours(overlay, contours, -1, (255,0,0), 3)

        both_lanes_present = left_coeff is not None and right_coeff is not None

        # 4. Display the beautiful blended result
        lane_center_x = (left_x_ahead + right_x_ahead) / 2
        lat_off_warped = (car_center[0] - lane_center_x) / scale
        
        cv.putText(overlay, f"Lane center offset: {lat_off_warped:.2f} m", (20, 20), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))

        left_lane_offset_from_center = (left_x_ahead - 400) / scale
        right_lane_offset_from_center = (right_x_ahead - 400) / scale

        lane_width = abs(left_lane_offset_from_center) + abs(right_lane_offset_from_center)

        lane_valid = (abs(a_left - a_right) < 0.1) and lane_width < 6 and both_lanes_present
        cv.putText(overlay, f"Valid lanes: {lane_valid}", (20, 110), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255) if not lane_valid else (0,255,0))

        cv.putText(overlay, f"L off: {left_lane_offset_from_center:.2f} m | R off: {right_lane_offset_from_center:.2f} m", (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))
        cv.putText(overlay, f"Road width: {lane_width:.2f} m", (20, 80), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))

        # lane departure: |offset| beyond threshold = departing, sign gives the side
        DEPARTURE_THRESHOLD = 0.4  # m; ~ (3.5 m lane - 1.8 m car) / 2 * 0.5

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

        cv.imshow('YOLOPv2 Panoptic Perception Overlay', overlay)
        # out.write(overlay) 

        # warped = cv.warpPerspective(ll_mask_resized, M, (canvas_w, canvas_h))

        # closest point the camera can see right after the hood
        cv.line(warped, (0, 700), (800, 700), (255, 255, 0), 2)

        #center of the car
        cv.drawMarker(warped, (int(car_center[0]), int(car_center[1])), color=(0, 0, 255), markerType=cv.MARKER_CROSS, markerSize=20, thickness=2)

        # center of the car projected 7.874m ahead
        cv.drawMarker(warped, (400, 400), color=(128, 128, 128), markerType=cv.MARKER_CROSS, markerSize=20, thickness=2)

        # lane center both 7.874m and at the hood center
        cv.drawMarker(warped, (int(lane_center_x), 400), color=(128, 255, 128), markerType=cv.MARKER_CROSS, markerSize=20, thickness=2)
        cv.drawMarker(warped, (int(lane_center_x), 700), color=(128, 255, 128), markerType=cv.MARKER_CROSS, markerSize=20, thickness=2)

        
        # cv.imshow("Bird Eye's View", warped)
        cv.imshow("Bird Eye's View - Lane Mask", warped)
        # bev.write(warped) 
        

        

        print(f"Frame {index}/{len(frame_names)}")

        if cv.waitKey(20) & 0xFF == ord('q'):
                break

    cv.destroyAllWindows()