#distance_estimation_sgbm.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load stereo images
left_img = cv2.imread("stereo_images/left.png", cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread("stereo_images/right.png", cv2.IMREAD_GRAYSCALE)

# Check if images are loaded properly
if left_img is None or right_img is None:
    print("Error: Images not found! Please check the file path.")
    exit()

# Create StereoSGBM matcher
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=64,  # Increase for better accuracy
    blockSize=9,
    P1=8 * 3 * 9**2,   # Smoothing parameter 1
    P2=32 * 3 * 9**2,  # Smoothing parameter 2
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

# Compute the disparity map
disparity = stereo.compute(left_img, right_img).astype(np.float32) / 16.0

# Normalize for visualization
disparity_visual = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
disparity_visual = np.uint8(disparity_visual)

# Avoid division by zero in distance calculation
disparity[disparity <= 0] = 0.1  

# Camera parameters (Update with real values)
focal_length = 800  # Focal length in pixels
baseline = 0.1  # Baseline distance between cameras in meters

# Compute depth (distance) map
depth_map = (focal_length * baseline) / disparity

# Normalize depth map for visualization
depth_visual = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
depth_visual = np.uint8(depth_visual)

# Display disparity and depth maps
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(disparity_visual, cmap='jet')
plt.colorbar(label="Disparity")
plt.title("Improved Disparity Map (SGBM)")

plt.subplot(1, 2, 2)
plt.imshow(depth_visual, cmap='jet')
plt.colorbar(label="Estimated Distance (meters)")
plt.title("Improved Depth (Distance) Map")

plt.show()
