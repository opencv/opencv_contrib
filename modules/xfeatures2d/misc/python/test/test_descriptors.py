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


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
