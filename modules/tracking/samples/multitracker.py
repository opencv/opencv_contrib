import numpy as np
import cv2

cv2.namedWindow("tracking")
camera = cv2.VideoCapture("E:/code/opencv/samples/data/768x576.avi")
tracker = cv2.MultiTracker("MIL")
bbox1 = (638.0,230.0,56.0,101.0)
bbox2 = (240.0,210.0,60.0,104.0)
bbox3 = (486.0,149.0,54.0,83.0)
init_once = False

while camera.isOpened():
    ok, image=camera.read()
    if not ok:
        print 'no image read'
        break

    if not init_once:
        # add a list of boxes:
        ok = tracker.add(image, (bbox1,bbox2))
        # or add single box:
        ok = tracker.add(image, bbox3)
        init_once = True

    ok, boxes = tracker.update(image)
    print ok, boxes

    for newbox in boxes:
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        cv2.rectangle(image, p1, p2, (200,0,0))

    cv2.imshow("tracking", image)
    k = cv2.waitKey(1) & 0xff
    if k == 27 : break # esc pressed
