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

    def test_cudabgsegm(self):
        lr = 0.05
        sz = (128,128,1)
        npMat = (np.random.random(sz) * 255).astype(np.uint8)
        cuMat = cv.cuda_GpuMat(npMat)
        cuMatBg = cv.cuda_GpuMat(cuMat.size(),cuMat.type())
        cuMatFg = cv.cuda_GpuMat(cuMat.size(),cuMat.type())

        mog = cv.cuda.createBackgroundSubtractorMOG()
        mog.apply(cuMat, lr, cv.cuda.Stream_Null(), cuMatFg)
        mog.getBackgroundImage(cv.cuda.Stream_Null(),cuMatBg)
        self.assertTrue(sz[:2] == cuMatFg.size() == cuMatBg.size())
        self.assertTrue(sz[2] == cuMatFg.channels() == cuMatBg.channels())
        self.assertTrue(cv.CV_8UC1 == cuMatFg.type() == cuMatBg.type())
        mog = cv.cuda.createBackgroundSubtractorMOG()
        self.assertTrue(np.allclose(cuMatFg.download(),mog.apply(cuMat, lr, cv.cuda.Stream_Null()).download()))
        self.assertTrue(np.allclose(cuMatBg.download(),mog.getBackgroundImage(cv.cuda.Stream_Null()).download()))

        mog2 = cv.cuda.createBackgroundSubtractorMOG2()
        mog2.apply(cuMat, lr, cv.cuda.Stream_Null(), cuMatFg)
        mog2.getBackgroundImage(cv.cuda.Stream_Null(),cuMatBg)
        self.assertTrue(sz[:2] == cuMatFg.size() == cuMatBg.size())
        self.assertTrue(sz[2] == cuMatFg.channels() == cuMatBg.channels())
        self.assertTrue(cv.CV_8UC1 == cuMatFg.type() == cuMatBg.type())
        mog2 = cv.cuda.createBackgroundSubtractorMOG2()
        self.assertTrue(np.allclose(cuMatFg.download(),mog2.apply(cuMat, lr, cv.cuda.Stream_Null()).download()))
        self.assertTrue(np.allclose(cuMatBg.download(),mog2.getBackgroundImage(cv.cuda.Stream_Null()).download()))

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()