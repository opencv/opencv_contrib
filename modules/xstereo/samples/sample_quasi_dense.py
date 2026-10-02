import numpy as np
import cv2 as cv

left_img = cv.imread(cv.samples.findFile("aloeL.jpg"), cv.IMREAD_COLOR)
right_img = cv.imread(cv.samples.findFile("aloeR.jpg"), cv.IMREAD_COLOR)

frame_size = left_img.shape[0:2];

stereo = cv.stereo.QuasiDenseStereo_create(frame_size[::-1])
stereo.process(left_img, right_img)
disp = stereo.getDisparity()
cv.imshow("disparity", disp)
cv.waitKey()
dense_matches = stereo.getDenseMatches()
try:
    with open("dense.txt", "wt") as f:
        # if you want all matches use for idx in len(dense_matches): It can be a big file
        for idx in range(0, min(10, len(dense_matches))):
            nb = f.write(str(dense_matches[idx].p0) + "\t" + str(dense_matches[idx].p1) + "\t" + str(dense_matches[idx].corr) + "\n")
except:
    print("Cannot open file")
