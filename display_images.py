#display_images.py
import cv2

# Load the captured images
left_img = cv2.imread("stereo_images/left.png")
right_img = cv2.imread("stereo_images/right.png")

# Display images
cv2.imshow("Left Image", left_img)
cv2.imshow("Right Image", right_img)

# Wait for a key press and close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
