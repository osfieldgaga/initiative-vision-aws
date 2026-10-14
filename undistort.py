import cv2 as cv
import numpy as np

frame = cv.imread("data/images/car_center_2.jpg") 

h, w = frame.shape[:2]

imx219_matrix = np.array([[908.3685, 0, 647.8779],
[0, 912.6167, 576.7830],
[0, 0, 1.0]], dtype=float) 

imx219_dist = np.array([-0.3404, 0.1337, 0, 0 ,0])

print(h, w, imx219_matrix)

new_calib_mtx, roi = cv.getOptimalNewCameraMatrix(imx219_matrix, imx219_dist, (w, h), 1, (w, h))

# undistort
dst = cv.undistort(frame, imx219_matrix, imx219_dist, None, new_calib_mtx)

# then crop
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]



# only rotate after undistorting to preserve calibration

frame_r90 = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
undistort_r90 = cv.rotate(dst, cv.ROTATE_90_CLOCKWISE)

# resize to display
frame_resized = cv.resize(frame_r90, (640, 640), interpolation=cv.INTER_AREA)
undistort_resized = cv.resize(undistort_r90, (640, 640), interpolation=cv.INTER_AREA)


cv.imshow("Distorted", frame_resized)
cv.imshow("Undistorted", undistort_resized)

# cv.imwrite("data/images/notebook_demo_frame_r90.jpg", frame_r90)
cv.imwrite("data/images/car_center_2.jpg", undistort_resized)

cv.waitKey(0)
