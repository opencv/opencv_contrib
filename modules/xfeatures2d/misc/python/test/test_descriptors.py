#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function

import os
import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests

class MSDDetector_test(NewOpenCVTests):

    def test_create(self):

        msd = cv.xfeatures2d.MSDDetector_create()
        self.assertFalse(msd is None)

        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        kp1_ = msd.detect(img1, None)

class matchLOGOS_test(NewOpenCVTests):

    def test_basic(self):

        frame = self.get_sample('python/images/baboon.png', cv.IMREAD_COLOR)
        detector = cv.AKAZE_create(threshold = 0.003)

        keypoints1, descrs1 = detector.detectAndCompute(frame, None)
        keypoints2, descrs2 = detector.detectAndCompute(frame, None)
        matches1to2 = cv.xfeatures2d.matchLOGOS(keypoints1, keypoints2, range(len(keypoints1)), range(len(keypoints2)))
        self.assertFalse(matches1to2 is None)


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
