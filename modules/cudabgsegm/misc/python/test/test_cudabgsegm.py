#!/usr/bin/env python
import os
import cv2 as cv
import numpy as np

from tests_common import NewOpenCVTests, unittest

class cudabgsegm_test(NewOpenCVTests):
    def setUp(self):
        super(cudabgsegm_test, self).setUp()
        if not cv.cuda.getCudaEnabledDeviceCount():
            self.skipTest("No CUDA-capable device is detected")

    def test_existence(self):
        #Test at least the existence of wrapped functions for now

        _bgsub = cv.cuda.createBackgroundSubtractorMOG()
        _bgsub = cv.cuda.createBackgroundSubtractorMOG2()

        self.assertTrue(True) #It is sufficient that no exceptions have been there

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()