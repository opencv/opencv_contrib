#!/usr/bin/env python
import os
import cv2 as cv
import numpy as np

from tests_common import NewOpenCVTests, unittest

class cudafeatures2d_test(NewOpenCVTests):
    def setUp(self):
        super(cudafeatures2d_test, self).setUp()
        if not cv.cuda.getCudaEnabledDeviceCount():
            self.skipTest("No CUDA-capable device is detected")

    def test_cudafeatures2d(self):
        npMat1 = self.get_sample("samples/data/right01.jpg")
        npMat2 = self.get_sample("samples/data/right02.jpg")

        cuMat1 = cv.cuda_GpuMat()
        cuMat2 = cv.cuda_GpuMat()
        cuMat1.upload(npMat1)
        cuMat2.upload(npMat2)

        cuMat1 = cv.cuda.cvtColor(cuMat1, cv.COLOR_RGB2GRAY)
        cuMat2 = cv.cuda.cvtColor(cuMat2, cv.COLOR_RGB2GRAY)

        fast = cv.cuda_FastFeatureDetector.create()
        _kps = fast.detectAsync(cuMat1)

        orb = cv.cuda_ORB.create()
        _kps1, descs1 = orb.detectAndComputeAsync(cuMat1, None)
        _kps2, descs2 = orb.detectAndComputeAsync(cuMat2, None)

        self.assertTrue(len(orb.convert(_kps1)) == _kps1.size()[0])
        self.assertTrue(len(orb.convert(_kps2)) == _kps2.size()[0])

        bf = cv.cuda_DescriptorMatcher.createBFMatcher(cv.NORM_HAMMING)
        matches = bf.match(descs1, descs2)
        self.assertGreater(len(matches), 0)
        matches = bf.knnMatch(descs1, descs2, 2)
        self.assertGreater(len(matches), 0)
        matches = bf.radiusMatch(descs1, descs2, 0.1)
        self.assertGreater(len(matches), 0)

        self.assertTrue(True) #It is sufficient that no exceptions have been there

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()