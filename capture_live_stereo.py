#capture_live_stero.py
import cv2

cap = cv2.VideoCapture(0)  # Single camera

if not cap.isOpened():
    print("Error: Camera not found!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame!")
        break

    cv2.imshow("Move Camera for Left/Right Images", frame)

    key = cv2.waitKey(1)
    if key == ord("l"):  # Press 'l' for left image
        cv2.imwrite("stereo_images/left.png", frame)
        print("Left image saved!")

    if key == ord("r"):  # Press 'r' for right image
        cv2.imwrite("stereo_images/right.png", frame)
        print("Right image saved!")

    if key == ord("q"):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
