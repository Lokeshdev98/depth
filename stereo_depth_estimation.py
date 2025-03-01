#stereo_depth_estimation.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load stereo images
left_img = cv2.imread("stereo_images/left.png", cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread("stereo_images/right.png", cv2.IMREAD_GRAYSCALE)

# Check if images are loaded
if left_img is None or right_img is None:
    print("❌ Error: Could not load stereo images!")
    exit()

# Create a StereoBM object
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)

# Compute disparity map
disparity = stereo.compute(left_img, right_img)

# Normalize for visualization
disparity_norm = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
disparity_norm = np.uint8(disparity_norm)

# Display the disparity map
plt.figure(figsize=(10, 5))
plt.imshow(disparity_norm, cmap="jet")
plt.colorbar(label="Disparity Value")
plt.title("Disparity Map (Depth Estimation)")
plt.show()
