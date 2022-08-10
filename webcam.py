import cv2
import time

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")
img_name = "input/2.jpg"
time_last_snapshot = 0
take_shot_every_seconds = 0.1

while True:
    # take a snapshot every 5 seconds

    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
        # SPACE pressed
    if time.perf_counter() - time_last_snapshot > take_shot_every_seconds:
        time_last_snapshot = time.perf_counter()
        cv2.imwrite(img_name, frame)
        print("{} written! at {:.4f}".format(img_name, time.time()))

cam.release()

cv2.destroyAllWindows()
