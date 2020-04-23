#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function

import os
import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests

class sift_compatibility_test(NewOpenCVTests):

    def test_create(self):

        sift = cv.xfeatures2d.SIFT_create()
        self.assertFalse(sift is None)

        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        kp1_, des1_ = sift.detectAndCompute(img1, None)


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
