#distance_estimation.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the stereo images in grayscale
left_img = cv2.imread("stereo_images/left.png", cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread("stereo_images/right.png", cv2.IMREAD_GRAYSCALE)

# Check if images are loaded properly
if left_img is None or right_img is None:
    print("Error: Images not found! Please check the file path.")
    exit()

# Stereo block matching (SBM) for disparity calculation
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(left_img, right_img)

# Normalize the disparity map for visualization
disparity = np.float32(disparity)
disparity[disparity <= 0] = 0.1  # Avoid division by zero

# Camera parameters (You need to set these values correctly)
focal_length = 800  # Example focal length in pixels
baseline = 0.1  # Example baseline distance in meters

# Compute depth (distance) map
depth_map = (focal_length * baseline) / disparity

# Normalize for visualization
depth_map = cv2.normalize(depth_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
depth_map = np.uint8(depth_map)

# Display disparity and depth maps
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(disparity, cmap='jet')
plt.colorbar(label="Disparity")
plt.title("Disparity Map")

plt.subplot(1, 2, 2)
plt.imshow(depth_map, cmap='jet')
plt.colorbar(label="Estimated Distance (meters)")
plt.title("Depth (Distance) Map")

plt.show()
