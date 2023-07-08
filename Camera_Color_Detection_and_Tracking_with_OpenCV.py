import cv2
import numpy as np

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

def emptyFunction(a):
    threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
    area = cv2.getTrackbarPos("Area", "Parameters")
    print("Threshold1:", threshold1)
    print("Threshold2:", threshold2)
    print("Area:", area)

cv2.namedWindow("HSV")
cv2.resizeWindow("HSV", 640, 240)
cv2.createTrackbar("HUE Min", "HSV", 0, 179, emptyFunction)
cv2.createTrackbar("HUE Max", "HSV", 179, 179, emptyFunction)
cv2.createTrackbar("SAT Min", "HSV", 0, 255, emptyFunction)
cv2.createTrackbar("SAT Max", "HSV", 255, 255, emptyFunction)
cv2.createTrackbar("VALUE Min", "HSV", 0, 255, emptyFunction)
cv2.createTrackbar("VALUE Max", "HSV", 255, 255, emptyFunction)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame from the camera.")
        break

    imgHsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    hMin = cv2.getTrackbarPos("HUE Min", "HSV")
    hMax = cv2.getTrackbarPos("HUE Max", "HSV")
    sMin = cv2.getTrackbarPos("SAT Min", "HSV")
    sMax = cv2.getTrackbarPos("SAT Max", "HSV")
    vMin = cv2.getTrackbarPos("VALUE Min", "HSV")
    vMax = cv2.getTrackbarPos("VALUE Max", "HSV")

    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])
    mask = cv2.inRange(imgHsv, lower, upper)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    hStack = np.hstack([frame, mask, result])
    cv2.imshow('Horizontal Stacking', hStack)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
