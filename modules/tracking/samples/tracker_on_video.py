import numpy as np
import cv2 as cv
import sys
import glob

image_sequence = sorted(glob.glob('D:/LaSOTTesting/airplane-13/img/*.jpg'))
first_frame = cv.imread(image_sequence[0])

bbox = (540, 366, 151, 53)

tracker = cv.TrackerBoosting_create()
print("tracker", tracker)
#tracker = cv.TrackerMIL_create()
#tracker = cv.TrackerKCF_create()
#tracker = cv.TrackerMedianFlow_create()
#tracker = cv.TrackerGOTURN_create()
#tracker = cv.TrackerMOSSE_create()
#tracker = cv.TrackerCSRT_create()
init_once = False

for f, frame in enumerate(image_sequence):
    image = cv.imread(frame)

    if not init_once:
        ok = tracker.init(first_frame, bbox)
        print("status ", ok)
        init_once = True

    ok, newbox = tracker.update(image)

    if ok:
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        cv.rectangle(image, p1, p2, (200, 0, 0))

    cv.imshow("tracking", image)
    k = cv.waitKey(1) & 0xff
    if k == 27:
        break  # esc pressed

image_sequence = sorted(glob.glob('D:/LaSOTTesting/airplane-9/img/*.jpg'))
first_frame = cv.imread(image_sequence[0])

bbox = (234, 267, 303, 90)

tracker = cv.TrackerBoosting_create()
print("tracker", tracker)
#tracker = cv.TrackerMIL_create()
#tracker = cv.TrackerKCF_create()
#tracker = cv.TrackerMedianFlow_create()
#tracker = cv.TrackerGOTURN_create()
#tracker = cv.TrackerMOSSE_create()
#tracker = cv.TrackerCSRT_create()
init_once = False

for f, frame in enumerate(image_sequence):
    image = cv.imread(frame)

    if not init_once:
        ok = tracker.init(first_frame, bbox)
        print("status ", ok)
        init_once = True

    ok, newbox = tracker.update(image)

    if ok:
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        cv.rectangle(image, p1, p2, (200, 0, 0))

    cv.imshow("tracking", image)
    k = cv.waitKey(1) & 0xff
    if k == 27:
        break  # esc pressed
