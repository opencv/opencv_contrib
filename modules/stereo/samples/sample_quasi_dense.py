import numpy as np
import cv2 as cv

left_img = cv.imread("G:/Lib/opencv/samples/data/aloeL.jpg", cv.IMREAD_COLOR)
right_img = cv.imread("G:/Lib/opencv/samples/data/aloeR.jpg", cv.IMREAD_COLOR)

frame_size = leftImg.shape[0:2];

stereo = cv.stereo.QuasiDenseStereo_create(frame_size[::-1])
stereo.process(left_img,right_img)
disp = stereo.getDisparity(80)
cv.imshow("disparity", disp)
cv.waitKey()
dense_matches = stereo.getDenseMatches()
try:
    f = open("dense.txt", "wt")
    with f:
        for idx in range(0, min(10, len(dense_matches))):
            nb = f.write(str(dense_matches[idx].p0) + "\t" + str(dense_matches[idx].p1) + "\t" + str(dense_matches[idx].corr) + "\n")
except:
    print("Cannot open file")
