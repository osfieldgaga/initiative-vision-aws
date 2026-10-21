import cv2 as cv

img = cv.imread("Photos/cat_large.jpg")

cv.imshow("Huge cat", img)

def rescale_frame(frame, scale = 0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


capture = cv.VideoCapture("Videos/dog.mp4")

while True:
    isTrue, frame = capture.read()

    frame_resized = rescale_frame(frame, 0.2)


    cv.imshow("Dog", frame)
    cv.imshow("Dog resized", frame_resized)


    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()
cv.destroyAllWindows()