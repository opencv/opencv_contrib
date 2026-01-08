#!/usr/bin/env python
import os
import cv2 as cv
import numpy as np

from tests_common import NewOpenCVTests, unittest

class cudafilters_test(NewOpenCVTests):
    def setUp(self):
        super(cudafilters_test, self).setUp()
        if not cv.cuda.getCudaEnabledDeviceCount():
            self.skipTest("No CUDA-capable device is detected")

    def test_existence(self):
        #Test at least the existence of wrapped functions for now

        _filter = cv.cuda.createBoxFilter(cv.CV_8UC1, -1, (3, 3))
        _filter = cv.cuda.createLinearFilter(cv.CV_8UC4, -1, np.eye(3))
        _filter = cv.cuda.createLaplacianFilter(cv.CV_16UC1, -1, ksize=3)
        _filter = cv.cuda.createSeparableLinearFilter(cv.CV_8UC1, -1, np.eye(3), np.eye(3))
        _filter = cv.cuda.createDerivFilter(cv.CV_8UC1, -1, 1, 1, 3)
        _filter = cv.cuda.createSobelFilter(cv.CV_8UC1, -1, 1, 1)
        _filter = cv.cuda.createScharrFilter(cv.CV_8UC1, -1, 1, 0)
        _filter = cv.cuda.createGaussianFilter(cv.CV_8UC1, -1, (3, 3), 16)
        _filter = cv.cuda.createMorphologyFilter(cv.MORPH_DILATE, cv.CV_32FC1, np.eye(3))
        _filter = cv.cuda.createBoxMaxFilter(cv.CV_8UC1, (3, 3))
        _filter = cv.cuda.createBoxMinFilter(cv.CV_8UC1, (3, 3))
        _filter = cv.cuda.createRowSumFilter(cv.CV_8UC1, cv.CV_32FC1, 3)
        _filter = cv.cuda.createColumnSumFilter(cv.CV_8UC1, cv.CV_32FC1, 3)
        _filter = cv.cuda.createMedianFilter(cv.CV_8UC1, 3)

        self.assertTrue(True) #It is sufficient that no exceptions have been there

    def test_laplacian(self):
        npMat = (np.random.random((128, 128)) * 255).astype(np.uint16)
        cuMat = cv.cuda_GpuMat()
        cuMat.upload(npMat)

        self.assertTrue(np.allclose(cv.cuda.createLaplacianFilter(cv.CV_16UC1, -1, ksize=3).apply(cuMat).download(),
                                         cv.Laplacian(npMat, cv.CV_16UC1, ksize=3)))

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()