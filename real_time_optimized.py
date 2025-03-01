#real_time_optimized.py
import cv2
import numpy as np
import threading
from ultralytics import YOLO
import pyttsx3  # Text-to-speech for voice output

# Load YOLO model
model = YOLO("yolov8n.pt")

# Initialize text-to-speech
engine = pyttsx3.init()
engine.setProperty("rate", 150)  # Adjust speech speed

# Camera parameters
FOCAL_LENGTH = 700  # Adjust based on your camera
KNOWN_HEIGHTS = {0: 1.7, 67: 0.5}  # Example: Person=1.7m, Cell phone=0.5m

class CameraStream:
    """ Threaded video capture for faster frame processing """
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.thread = threading.Thread(target=self.update, args=())

    def start(self):
        self.thread.start()
        return self

    def update(self):
        while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.thread.join()

# Start threaded camera stream
cap = CameraStream().start()

while True:
    frame = cap.read()
    if frame is None:
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
            real_height = KNOWN_HEIGHTS.get(cls, 1.0)  # Default to 1m

            # Estimate distance
            if object_height > 0:
                distance = (FOCAL_LENGTH * real_height) / object_height
                distance_text = f"{label}: {distance:.2f} m"

                # Speak out distance
                engine.say(distance_text)
                engine.runAndWait()
            else:
                distance_text = "N/A"

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, distance_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Show the result
    cv2.imshow("Object Detection & Distance", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.stop()
cv2.destroyAllWindows()
