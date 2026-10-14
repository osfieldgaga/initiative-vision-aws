import torch
import cv2 as cv
import numpy as np
from pathlib import Path
import sys
import os

## Camera Intrinsics

imx219_matrix = np.array([[908.3685, 0, 647.8779],
[0, 912.6167, 576.7830],
[0, 0, 1.0]], dtype=float) 

imx219_dist = np.array([-0.3404, 0.1337, 0, 0 ,0])


src = np.float32([
    [134.8,  582.5],   # near-left cone's base, read off the photo
    [602, 599],  # near-right cone's base
    [493.7,  557.7],   # far-right cone's base
    [265.4,   544],    # far-left cone's base
])

scale = 100  # px per unit, same convention you used for the rectangle itself

dst = np.float32([
    [1 * scale, 4 * scale],   # lower-left
    [7 * scale, 4 * scale],   # lower-right
    [7 * scale, 1 * scale],   # upper-right
    [1 * scale, 1 * scale],   # upper-left
])

M = cv.getPerspectiveTransform(src, dst, solveMethod=0)

canvas_w = 8 * scale   # 1 unit padding on each side of a 6-unit-wide rectangle
canvas_h = 5 * scale   # 1 unit padding on each side of a 3-unit-deep rectangle



## Load Yolo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.jit.load("data/weights/yolopv2.pt", map_location=device)
model.eval()


# latest_session_frame = Path("/home/osfield/osfield@creampie/Desktop/initiative-vision/latest_frame.jpg")
latest_session_frame = Path("/home/osfield/osfield@creampie/Desktop/initiative-vision/latest_frame.jpg")

print(latest_session_frame)


# frame = cv.imread(latest_session_frame) 
frame = cv.imread("data/images/notebook_demo.jpg") 
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

    print("\n\n")
    print("==== Detections ====")
    print(detections[0][0].size())
    """
    ==== Driveale areas ====
    torch.Size([1, 255, 80, 80])
    """


    print("\n\n")
    print("==== Driveale areas ====")
    print(driveable_area[0].size())
    """
    ==== Driveale areas ====
    torch.Size([2, 640, 640])
    """

    print("\n\n")
    print("==== Lanes ====")
    print(lld)
    """
    ==== Lanes ====
    torch.Size([1, 1, 640, 640])
    """
    print("\n\n")



# Example: Extract binary masks for drivable area and lanes
# Take the argmax of the segmentation outputs across the channel dimension
da_mask = driveable_area.argmax(dim=1)[0].cpu().numpy().astype(np.uint8) * 255
ll_mask = (lld[0, 0] > 0.5).cpu().numpy().astype(np.uint8) * 255

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

cv.imshow('YOLOPv2 Panoptic Perception Overlay', overlay)
        # out.write(overlay) 

warped = cv.warpPerspective(overlay, M, (canvas_w, canvas_h))

cv.imshow("Bird Eye's View", warped)


cv.waitKey(0)
cv.destroyAllWindows()
