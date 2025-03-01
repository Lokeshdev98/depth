#stereo_depth_object_distance.py
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Load YOLOv8 model (pretrained)
model = YOLO("yolov8n.pt")

# Camera parameters (adjust based on your setup)
focal_length = 700  # pixels
baseline = 6  # cm

# Open stereo cameras
cap_left = cv2.VideoCapture(0)  # Left camera
cap_right = cv2.VideoCapture(1)  # Right camera

if not cap_left.isOpened() or not cap_right.isOpened():
    print("❌ Error: Could not open one or both cameras!")
    exit()

# StereoBM for disparity calculation
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)

while True:
    retL, left_frame = cap_left.read()
    retR, right_frame = cap_right.read()
    if not retL or not retR:
        print("❌ Error: Could not read frames!")
        break

    # Convert to grayscale
    left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

    # Compute disparity map
    disparity = stereo.compute(left_gray, right_gray).astype(np.float32)
    disparity[disparity <= 0] = 0.1  # Avoid division by zero

    # Compute depth map
    depth_map = (focal_length * baseline) / disparity

    # Run YOLO object detection on the left image
    results = model(left_frame)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
            conf = box.conf[0].item()  # Confidence
            cls = int(box.cls[0].item())  # Class index
            label = model.names[cls]  # Get label

            # Get the disparity at the center of the bounding box
            disp_value = np.mean(disparity[y1:y2, x1:x2])
            distance = (focal_length * baseline) / disp_value if disp_value > 0 else 9999

            # Draw bounding box & distance
            cv2.rectangle(left_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(left_frame, f"{label}: {distance:.2f} cm", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Show depth map & object detection
    depth_norm = cv2.normalize(depth_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    depth_norm = np.uint8(np.nan_to_num(depth_norm))

    cv2.imshow("Left Camera - Object Detection", left_frame)
    cv2.imshow("Depth Map", depth_norm)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
