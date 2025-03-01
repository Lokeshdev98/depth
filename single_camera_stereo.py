#single_camera_stereo.py
import cv2
import time

# Open the single camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Camera not found!")
    exit()

# Capture the first frame (left image)
ret, left_img = cap.read()
if not ret:
    print("❌ Error: Could not capture left image")
    cap.release()
    exit()

print("✅ Left image captured. Move the camera slightly to the right and wait...")

time.sleep(2)  # Small delay to move the camera slightly

# Capture the second frame (right image)
ret, right_img = cap.read()
if not ret:
    print("❌ Error: Could not capture right image")
    cap.release()
    exit()

# Save the images
cv2.imwrite("stereo_images/left.png", left_img)
cv2.imwrite("stereo_images/right.png", right_img)

print("✅ Stereo images saved successfully! Check 'stereo_images' folder.")

cap.release()
