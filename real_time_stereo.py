#real_time_stereo.py
import cv2
import numpy as np

# Open two cameras
cap_left = cv2.VideoCapture(0)  # Change index if needed
cap_right = cv2.VideoCapture(1)  # Ensure second camera is available

if not cap_left.isOpened() or not cap_right.isOpened():
    print("Error: One or both cameras not found!")
    exit()

# StereoBM settings
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)

while True:
    # Capture frames
    retL, frame_left = cap_left.read()
    retR, frame_right = cap_right.read()

    if not retL or not retR:
        print("Error: Failed to capture frames!")
        break

    # Convert to grayscale
    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

    # Compute disparity
    disparity = stereo.compute(gray_left, gray_right)

    # Normalize disparity for display
    disparity_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disparity_norm = np.uint8(disparity_norm)

    # Display results
    cv2.imshow("Left Camera", frame_left)
    cv2.imshow("Right Camera", frame_right)
    cv2.imshow("Disparity Map", disparity_norm)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
