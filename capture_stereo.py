#capture_stero.py
import cv2

# Capture images from webcam
cap = cv2.VideoCapture(0)  # Use camera index 0 (change if needed)

# Read a frame
ret, frame = cap.read()

if ret:
    # Create a directory to store images
    import os
    if not os.path.exists("stereo_images"):
        os.makedirs("stereo_images")  # Create folder if it doesn't exist

    # Save two images as stereo pair
    cv2.imwrite("stereo_images/left.png", frame)
    cv2.imwrite("stereo_images/right.png", frame)
    print("Images saved successfully in 'stereo_images' folder!")
else:
    print("Failed to capture images.")

cap.release()
