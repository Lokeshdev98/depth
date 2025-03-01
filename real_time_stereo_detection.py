#real_time_stereo_detection.py
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open stereo cameras (modify indexes if needed)
cap_left = cv2.VideoCapture(0)
cap_right = cv2.VideoCapture(1)

if not cap_left.isOpened() or not cap_right.isOpened():
    print("❌ Error: Could not open one or both cameras!")
    exit()

# Stereo Block Matching for depth
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

# Camera parameters (adjust these based on your setup)
focal_length = 700  # Example focal length in pixels
baseline = 0.1  # Distance between two cameras in meters

while True:
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()

    if not ret_left or not ret_right:
        print("❌ Error: Could not capture frames!")
        break

    # Convert to grayscale
    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

    # Compute disparity map
    disparity = stereo.compute(gray_left, gray_right)
    disparity = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disparity = np.uint8(disparity)

    # Run YOLO object detection
    results = model(frame_left)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0].item())
            label = model.names[cls]

            # Get object center for depth calculation
            obj_x = (x1 + x2) // 2
            obj_y = (y1 + y2) // 2

            # Extract disparity value
            disp_value = disparity[obj_y, obj_x]

            # Avoid division by zero
            if disp_value > 0:
                depth = (focal_length * baseline) / disp_value
            else:
                depth = float("inf")

            # Draw bounding box and depth label
            cv2.rectangle(frame_left, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_left, f"{label}: {depth:.2f}m", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Show results
    cv2.imshow("Left Camera - Object Detection", frame_left)
    cv2.imshow("Disparity Map", disparity)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
