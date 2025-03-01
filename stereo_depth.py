#stereo_depth.py
import cv2
import numpy as np

# Load stereo images
left_img = cv2.imread("stereo_images/left.png", cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread("stereo_images/right.png", cv2.IMREAD_GRAYSCALE)

if left_img is None or right_img is None:
    print("❌ Error: Stereo images not found!")
    exit()

# Stereo Block Matching for Disparity Map
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(left_img, right_img)

# Normalize the disparity map
disparity = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
disparity = np.uint8(disparity)

# Show results
cv2.imshow("Disparity Map", disparity)
cv2.waitKey(0)
cv2.destroyAllWindows()
