import torch
import cv2 as cv
import numpy as np
from pathlib import Path
import sys

frame = cv.imread("data/images/car_center_2+with markers.jpg") 




src = np.float32([
    [134.8,  582.5],   # near-left cone's base, read off the photo
    [602, 599],  # near-right cone's base
    [493.7,  551],   # far-right cone's base
    [265.4,   544],    # far-left cone's base
])

scale = 100  # px per unit, same convention you used for the rectangle itself


dst = np.float32([
    [2 * scale, 7 * scale],   # lower-left
    [7 * scale, 7 * scale],   # lower-right
    [7 * scale, 1 * scale],   # upper-right
    [2 * scale, 1 * scale],   # upper-left
])


M = cv.getPerspectiveTransform(src, dst, solveMethod=0)



canvas_w = 8 * scale   # 1 unit padding on each side of a 6-unit-wide rectangle
canvas_h = 8 * scale   # 1 unit padding on each side of a 3-unit-deep rectangle

warped = cv.warpPerspective(frame, M, (canvas_w, canvas_h))

cv.imshow("Unwarped", frame)
cv.imshow("Warped", warped)

# cv.imwrite("data/images/car_center_2+with markers_warped.jpg", warped)

cv.waitKey(0)
cv.destroyAllWindows()