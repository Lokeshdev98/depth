#real_time_depth.py
import cv2
import numpy as np

# Camera setup
cap_left = cv2.VideoCapture(0)
cap_right = cv2.VideoCapture(1)

if not cap_left.isOpened() or not cap_right.isOpened():
    print("Error: One or both cameras not found!")
    exit()

# StereoBM setup
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)

# Depth calculation parameters
focal_length = 800  # Adjust based on your camera
baseline = 0.1  # Distance between the cameras in meters

while True:
    retL, frame_left = cap_left.read()
    retR, frame_right = cap_right.read()

    if not retL or not retR:
        print("Error: Failed to capture frames!")
        break

    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

    disparity = stereo.compute(gray_left, gray_right)

    # Normalize disparity
    disparity_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disparity_norm = np.uint8(disparity_norm)

    # Avoid division by zero
    disparity[disparity == 0] = 0.1

    # Compute depth map
    depth_map = (focal_length * baseline) / disparity

    # Normalize for display
    depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_norm = np.uint8(depth_norm)

    # Display results
    cv2.imshow("Left Camera", frame_left)
    cv2.imshow("Right Camera", frame_right)
    cv2.imshow("Disparity Map", disparity_norm)
    cv2.imshow("Depth Map", depth_norm)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
