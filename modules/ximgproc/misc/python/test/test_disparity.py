#!/usr/bin/env python
import cv2 as cv
import numpy as np

from tests_common import NewOpenCVTests

class disparity_test(NewOpenCVTests):
    def test_disp(self):
        # readGT
        ret,GT = cv.ximgproc.readGT(self.find_file("cv/disparityfilter/GT.png"))
        self.assertEqual(ret, 0) # returns 0 on success!
        self.assertFalse(np.shape(GT) == ())

        # computeMSE
        left = cv.imread(self.find_file("cv/disparityfilter/disparity_left_raw.png"), cv.IMREAD_UNCHANGED)
        self.assertFalse(np.shape(left) == ())
        left = np.asarray(left, dtype=np.int16)
        mse = cv.ximgproc.computeMSE(GT, left, (0, 0, GT.shape[1], GT.shape[0]))

        # computeBadPixelPercent
        bad = cv.ximgproc.computeBadPixelPercent(GT, left, (0, 0, GT.shape[1], GT.shape[0]), 24)

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
