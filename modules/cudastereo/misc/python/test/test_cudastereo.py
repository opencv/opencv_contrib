#!/usr/bin/env python
import os
import cv2 as cv
import numpy as np

from tests_common import NewOpenCVTests, unittest

class cudastereo_test(NewOpenCVTests):
    def setUp(self):
        super(cudastereo_test, self).setUp()
        if not cv.cuda.getCudaEnabledDeviceCount():
            self.skipTest("No CUDA-capable device is detected")

    def test_reprojectImageTo3D(self):
        # Test's the functionality but not the results from reprojectImageTo3D
        sz = (128,128)
        np_disparity = np.random.randint(0, 64, sz, dtype=np.int16)
        cu_disparity = cv.cuda_GpuMat(np_disparity)
        np_q = np.random.randint(0, 100, (4, 4)).astype(np.float32)
        stream = cv.cuda.Stream()
        cu_xyz = cv.cuda.reprojectImageTo3D(cu_disparity, np_q, stream = stream)
        self.assertTrue(cu_xyz.type() == cv.CV_32FC4 and cu_xyz.size() == sz)
        cu_xyz1 = cv.cuda.GpuMat(sz, cv.CV_32FC3)
        cv.cuda.reprojectImageTo3D(cu_disparity, np_q, cu_xyz1, 3, stream)
        self.assertTrue(cu_xyz1.type() == cv.CV_32FC3 and cu_xyz1.size() == sz)

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()