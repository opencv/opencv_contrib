#!/usr/bin/env python
import os
import cv2 as cv
import numpy as np

from tests_common import NewOpenCVTests, unittest

class cudaobjdetect_test(NewOpenCVTests):
    def setUp(self):
        super(cudaobjdetect_test, self).setUp()
        if not cv.cuda.getCudaEnabledDeviceCount():
            self.skipTest("No CUDA-capable device is detected")

    @unittest.skipIf('OPENCV_TEST_DATA_PATH' not in os.environ,
                        "OPENCV_TEST_DATA_PATH is not defined")
    def test_hog(self):
        img_path = os.environ['OPENCV_TEST_DATA_PATH'] + '/gpu/caltech/image_00000009_0.png'
        npMat = cv.cvtColor(cv.imread(img_path),cv.COLOR_BGR2BGRA)

        cuMat = cv.cuda_GpuMat(npMat)
        cuHog = cv.cuda.HOG_create()
        cuHog.setSVMDetector(cuHog.getDefaultPeopleDetector())

        loc, conf = cuHog.detect(cuMat)
        self.assertTrue(len(loc) == len(conf) and len(loc) > 0 and len(loc[0]) == 2)

        loc = cuHog.detectWithoutConf(cuMat)
        self.assertTrue(len(loc) > 0 and len(loc[0]) == 2)

        loc = cuHog.detectMultiScaleWithoutConf(cuMat)
        self.assertTrue(len(loc) > 0 and len(loc[0]) == 4)

        cuHog.setGroupThreshold(0)
        loc, conf = cuHog.detectMultiScale(cuMat)
        self.assertTrue(len(loc) == len(conf) and len(loc) > 0 and len(loc[0]) == 4)

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()