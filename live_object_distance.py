#live_object_distance.py
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Load YOLOv8 model (pretrained)
model = YOLO("yolov8n.pt")

# Camera parameters (adjust based on your setup)
focal_length = 700  # pixels
baseline = 6  # cm

# Open webcam (adjust index if needed)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open camera!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Could not read frame!")
        break

    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Run YOLO object detection
    results = model(frame)
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
            conf = box.conf[0].item()  # Confidence
            cls = int(box.cls[0].item())  # Class index
            label = model.names[cls]  # Get label

            # Dummy disparity calculation (replace with real stereo processing)
            disparity = max(1, (x2 - x1) / 2)  

            # Compute depth
            distance = (focal_length * baseline) / disparity

            # Draw bounding box & distance
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {distance:.2f} cm", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Show frame
    cv2.imshow("Real-Time Object Detection & Distance", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
