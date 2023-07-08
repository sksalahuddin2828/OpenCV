# First draw a shape over the object with the mouse and press enter to start the trucking, press C to cancel. 

import cv2
import time

# Create a tracker object
tracker = cv2.TrackerKCF_create()

# Open the video capture
cap = cv2.VideoCapture(0)

# Read the first frame from the video
success, frame = cap.read()
if not success:
    print("Failed to read video")
    exit()

# Select the region of interest (ROI) for tracking
bbox = cv2.selectROI("Tracking", frame, False)

# Initialize the tracker with the selected ROI
tracker.init(frame, bbox)

# Variables for calculating FPS
fps_start_time = time.time()
fps_counter = 0
fps = 0

# Function to draw bounding box and status on the image
def drawBox(img, bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x, y), ((x + w), (y + h)), (0, 255, 255), 3, 3)
    cv2.putText(img, "Tracking", (x, y - 10), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

# Main loop for video processing
while True:
    # Read a frame from the video
    success, img = cap.read()
    if not success:
        print("Failed to read video")
        break

    # Update the tracker with the current frame
    success, bbox = tracker.update(img)

    # If tracking is successful, draw the bounding box
    if success:
        drawBox(img, bbox)
    else:
        # If tracking is lost, display "Lost" message
        cv2.putText(img, "Lost", (20, 40), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

    # Display status and FPS on the image
    cv2.rectangle(img, (15, 15), (200, 90), (255, 0, 0), 2)
    cv2.putText(img, "Status:", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(img, "Tracking", (102, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Calculate and display FPS
    fps_counter += 1
    fps_end_time = time.time() - fps_start_time
    if fps_end_time > 1:
        fps = fps_counter / fps_end_time
        fps_counter = 0
        fps_start_time = time.time()

    cv2.putText(img, "FPS: {:.2f}".format(fps), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Display the image with tracking results
    cv2.imshow("Tracking", img)
    
    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows() 
