#distance_prediction.py
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Define camera parameters (adjust based on your camera)
FOCAL_LENGTH = 700  # Example value (in pixels)
KNOWN_WIDTH = 15  # Example object width in cm (adjust based on the object)
REAL_DISTANCE = 50  # Known distance of the object in cm (for calibration)

# Open the camera
cap = cv2.VideoCapture(0)

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
            width = x2 - x1

            # Calculate distance using a known object width
            if width > 0:
                distance = (KNOWN_WIDTH * FOCAL_LENGTH) / width
                distance = round(distance, 2)

                # Draw bounding box and distance
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{distance} cm", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Show the result
    cv2.imshow("Object Detection & Distance", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
