import streamlit as st
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Ensure proper module import
sys.path.append(os.path.dirname(__file__))  # Adds current directory to the path
from distance_prediction import calculate_distance

# Load YOLO model
model = YOLO("yolov8n.pt")

def live_camera_feed():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Error: Could not open camera!")
        return

    stop_button_live = st.button("Stop Camera", key="stop_button_live_unique")
    distance_values = []
    frame_count = 0
    
    while cap.isOpened() and not stop_button_live:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Error: Could not read frame!")
            break
        
        # Object detection
        results = model(frame)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                label = model.names[cls]
                distance = calculate_distance(x1, y1, x2, y2)
                distance_values.append(distance)
                frame_count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {distance:.2f}m", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        st.image(frame, channels="BGR")
        
    cap.release()
    
    # Generate distance graph
    if distance_values:
        plt.figure()
        plt.plot(range(len(distance_values)), distance_values, marker='o', linestyle='-')
        plt.xlabel("Frame Count")
        plt.ylabel("Distance (m)")
        plt.title("Distance Over Time")
        st.pyplot(plt)

def process_uploaded_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    results = model(image)
    
    distance_values = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            distance = calculate_distance(x1, y1, x2, y2)
            distance_values.append(distance)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{distance:.2f}m", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    st.image(image, channels="BGR")
    
    # Generate distance graph
    if distance_values:
        plt.figure()
        plt.bar(range(len(distance_values)), distance_values)
        plt.xlabel("Detected Objects")
        plt.ylabel("Distance (m)")
        plt.title("Distance of Objects in Uploaded Image")
        st.pyplot(plt)

def main():
    st.title("AI-Powered Stereo Vision System")
    
    option = st.selectbox("Choose an operation:", ["Live Camera", "Upload Image"])
    
    if option == "Live Camera":
        live_camera_feed()
    elif option == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            process_uploaded_image(uploaded_file)
    
if __name__ == "__main__":
    main()
