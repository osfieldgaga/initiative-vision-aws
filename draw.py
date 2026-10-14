import cv2 as cv
import numpy as np

blank = np.zeros((500,500, 3), dtype="uint8")


# img = cv.imread("Photos/cat.jpg")
# cv.imshow("El Gato", img)

blank[200:300, 300:400] = 0,0,255

# cv.imshow("Blank", blank)

cv.rectangle(blank, (0,0), (250, 250), (0,255,0), thickness=2)

cv.circle(blank, (300,  67), 40, (0,0,255), thickness=3)

cv.line(blank, (100, 100), (150,150), (255,255,255), thickness=4)

cv.putText(blank, "Sauvage", (225, 225), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0, 255, 0), 2)

cv.imshow("Rectangle", blank)



cv.waitKey(0)