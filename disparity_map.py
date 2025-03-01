#disparity_map.py
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

# Create StereoBM (Block Matching) object for disparity calculation
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

# Compute disparity map
disparity = stereo.compute(left_img, right_img)

# Normalize the disparity map for better visualization
disparity = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
disparity = np.uint8(disparity)

# Display the disparity map
plt.figure(figsize=(10, 5))
plt.imshow(disparity, cmap='jet')
plt.colorbar(label="Disparity Value")
plt.title("Disparity Map")
plt.show()
