import cv2

cap = cv2.VideoCapture(0)  # Adjust the camera index if needed

while True:
    success, frame = cap.read()

    if not success:
        print("Failed to read frame")
        break

    print(frame.shape)  # Print the shape of the frame (height, width, channels)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
