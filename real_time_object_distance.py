#real_time_object_distance.py
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")  # Ensure you have yolov8n.pt

# Open single camera
cap = cv2.VideoCapture(0)

# Camera parameters (adjust these based on calibration)
FOCAL_LENGTH = 700  # Adjust based on your camera
KNOWN_HEIGHTS = {0: 1.7, 67: 0.5}  # Example: 0=person (1.7m), 67=cell phone (0.5m)

if not cap.isOpened():
    print("❌ Error: Could not open camera!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Could not read frame!")
        break

    # Run YOLO object detection
    results = model(frame)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0].item())
            label = model.names[cls]

            # Object height in pixels
            object_height = y2 - y1
            real_height = KNOWN_HEIGHTS.get(cls, 1.0)  # Default to 1m if unknown

            # Estimate distance
            if object_height > 0:
                distance = (FOCAL_LENGTH * real_height) / object_height
                distance_text = f"{distance:.2f} m"
            else:
                distance_text = "N/A"

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {distance_text}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Show the result
    cv2.imshow("Object Detection & Distance", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
