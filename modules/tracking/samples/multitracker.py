import numpy as np
import cv2 as cv
import sys

if len(sys.argv) != 2:
    print('Input video name is missing')
    exit()

print('Select 3 tracking targets')

cv.namedWindow("tracking")
camera = cv.VideoCapture(sys.argv[1])
tracker = cv.legacy.MultiTracker_create()
init_once = False

ok, image=camera.read()
if not ok:
    print('Failed to read video')
    exit()

bbox1 = cv.selectROI('tracking', image)
bbox2 = cv.selectROI('tracking', image)
bbox3 = cv.selectROI('tracking', image)

while camera.isOpened():
    ok, image=camera.read()
    if not ok:
        print('no image to read')
        break

    if not init_once:
        ok = tracker.add(cv.legacy.TrackerMIL_create(), image, bbox1)
        ok = tracker.add(cv.legacy.TrackerMIL_create(), image, bbox2)
        ok = tracker.add(cv.legacy.TrackerMIL_create(), image, bbox3)
        init_once = True

    ok, boxes = tracker.update(image)
    print(ok, boxes)

    for newbox in boxes:
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        cv.rectangle(image, p1, p2, (200,0,0))

    cv.imshow('tracking', image)
    k = cv.waitKey(1)
    if k == 27 : break # esc pressed
