#real_time_distance.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load stereo images
left_img = cv2.imread("stereo_images/left.png", cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread("stereo_images/right.png", cv2.IMREAD_GRAYSCALE)

if left_img is None or right_img is None:
    print("❌ Error: Could not load stereo images!")
    exit()

# StereoBM for disparity calculation
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
disparity = stereo.compute(left_img, right_img).astype(np.float32)

# Camera parameters (Adjust as needed)
focal_length = 700  # pixels
baseline = 6  # cm

# Avoid division by zero
disparity[disparity <= 0] = 0.1

# Compute depth map
depth_map = (focal_length * baseline) / disparity

# Normalize depth for visualization
depth_norm = cv2.normalize(depth_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
depth_norm = np.uint8(np.nan_to_num(depth_norm))

# Function to display distance on click
def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        distance = depth_map[y, x]  # Get distance at clicked point
        print(f"📏 Distance at ({x}, {y}): {distance:.2f} cm")

# Show depth map with click event
cv2.imshow("Depth Map", depth_norm)
cv2.setMouseCallback("Depth Map", on_mouse_click)

print("🖱️ Click on an object in the Depth Map to get its distance.")
cv2.waitKey(0)
cv2.destroyAllWindows()
