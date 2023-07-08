import cv2
import time

tracker = cv2.TrackerKCF_create()

cap = cv2.VideoCapture(0)

success, frame = cap.read()
if not success:
    print("Failed to read video")
    exit()

bbox = cv2.selectROI("Tracking", frame, False)
tracker.init(frame, bbox)

fps_start_time = time.time()
fps_counter = 0
fps = 0  # Initialize fps variable

def drawBox(img, bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x, y), ((x + w), (y + h)), (0, 255, 255), 3, 3)
    cv2.putText(img, "Tracking", (x, y - 10), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    
while True:
    success, img = cap.read()
    if not success:
        print("Failed to read video")
        break

    success, bbox = tracker.update(img)

    if success:
        drawBox(img, bbox)
    else:
        cv2.putText(img, "Lost", (20, 40), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.rectangle(img, (15, 15), (200, 90), (255, 0, 0), 2)
    cv2.putText(img, "Status:", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(img, "Tracking", (103, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Calculate and display FPS
    fps_counter += 1
    fps_end_time = time.time() - fps_start_time
    if fps_end_time > 1:
        fps = fps_counter / fps_end_time
        fps_counter = 0
        fps_start_time = time.time()

    cv2.putText(img, "FPS: {:.2f}".format(fps), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Tracking", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
