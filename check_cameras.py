import cv2

# Try different camera indexes
for i in range(5):  # Check up to 5 camera indexes
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"✅ Camera found at index {i}")
        cap.release()
    else:
        print(f"❌ No camera found at index {i}")
