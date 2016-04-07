import numpy as np
import cv2

cv2.namedWindow("tracking")
camera = cv2.VideoCapture("E:/code/opencv/samples/data/768x576.avi")
bbox = (638.0,230.0,56.0,101.0)
tracker = cv2.Tracker_create("MIL")
init_once = False

while camera.isOpened():
    ok, image=camera.read()
    if not ok:
        print 'no image read'
        break

    if not init_once:
        ok = tracker.init(image, bbox)
        init_once = True

    ok, newbox = tracker.update(image)
    print ok, newbox

    if ok:
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        cv2.rectangle(image, p1, p2, (200,0,0))

    cv2.imshow("tracking", image)
    k = cv2.waitKey(1) & 0xff
    if k == 27 : break # esc pressed