import cv2
import numpy as np

# Set the dimensions of the captured image
width_img = 720
height_img = 640

# Open the default camera device
cap = cv2.VideoCapture(0)
cap.set(10, 150)


def preprocess_image(img):
    # Convert the image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the grayscale image
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)

    # Apply Canny edge detection to the blurred image
    img_canny = cv2.Canny(img_blur, 200, 200)

    # Dilate the edges to close gaps
    kernel = np.ones((5, 5))
    img_dial = cv2.dilate(img_canny, kernel, iterations=2)

    # Erode the dilated image to reduce thickness
    img_thres = cv2.erode(img_dial, kernel, iterations=1)

    return img_thres


def get_contours(img):
    biggest = np.array([])
    max_area = 0

    # Find contours in the image
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area

    return biggest


def reorder_points(my_points):
    my_points = my_points.reshape((4, 2))
    my_points_new = np.zeros((4, 1, 2), np.int32)
    add = my_points.sum(1)
    my_points_new[0] = my_points[np.argmin(add)]
    my_points_new[3] = my_points[np.argmax(add)]
    diff = np.diff(my_points, axis=1)
    my_points_new[1] = my_points[np.argmin(diff)]
    my_points_new[2] = my_points[np.argmax(diff)]
    return my_points_new


def get_warp(img, biggest):
    biggest = reorder_points(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [width_img, 0], [0, height_img], [width_img, height_img]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_output = cv2.warpPerspective(img, matrix, (width_img, height_img))

    img_cropped = img_output[20:img_output.shape[0] - 20, 20:img_output.shape[1] - 20]
    img_cropped = cv2.resize(img_cropped, (width_img, height_img))

    return img_cropped


def stack_images(scale, img_array):
    rows = len(img_array)
    cols = len(img_array[0])
    rows_available = isinstance(img_array[0], list)
    width = img_array[0][0].shape[1]
    height = img_array[0][0].shape[0]

    if rows_available:
        # Stack images horizontally
        hor = [np.hstack(img_row) for img_row in img_array]
        # Stack rows vertically```python
ver = np.vstack(hor)
    else:
        # Stack images horizontally
        hor = np.hstack(img_array)
        ver = hor

    return cv2.resize(ver, None, fx=scale, fy=scale)


while True:
    success, img = cap.read()
    img = cv2.resize(img, (width_img, height_img))
    img_contour = img.copy()

    img_thres = preprocess_image(img)
    biggest = get_contours(img_thres)

    if biggest.size != 0:
        img_warped = get_warp(img, biggest)
        image_array = ([img_contour, img_warped])
        cv2.imshow("Image Warped", img_warped)
    else:
        image_array = ([img_contour, img])

    stacked_images = stack_images(0.6, image_array)
    cv2.imshow("Capture Recording", stacked_images)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
