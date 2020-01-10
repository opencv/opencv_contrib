#!/usr/bin/env python
import os
import cv2 as cv
import numpy as np

from tests_common import NewOpenCVTests, unittest

class xfeatures2d_test(NewOpenCVTests):
    def setUp(self):
        super(xfeatures2d_test, self).setUp()
        if not cv.cuda.getCudaEnabledDeviceCount():
            self.skipTest("No CUDA-capable device is detected")

    @unittest.skipIf('OPENCV_TEST_DATA_PATH' not in os.environ,
                     "OPENCV_TEST_DATA_PATH is not defined")
    def test_surf(self):
        img_path = os.environ['OPENCV_TEST_DATA_PATH'] + "/gpu/features2d/aloe.png"
        hessianThreshold = 100
        nOctaves = 3
        nOctaveLayers = 2
        extended = False
        keypointsRatio = 0.05
        upright = False

        npMat = cv.cvtColor(cv.imread(img_path),cv.COLOR_BGR2GRAY)
        cuMat = cv.cuda_GpuMat(npMat)

        try:
            cuSurf = cv.cuda_SURF_CUDA.create(hessianThreshold,nOctaves,nOctaveLayers,extended,keypointsRatio,upright)
            surf = cv.xfeatures2d_SURF.create(hessianThreshold,nOctaves,nOctaveLayers,extended,upright)
        except cv.error as e:
            self.assertEqual(e.code, cv.Error.StsNotImplemented)
            self.skipTest("OPENCV_ENABLE_NONFREE is not enabled in this build.")

        cuKeypoints = cuSurf.detect(cuMat,cv.cuda_GpuMat())
        keypointsHost = cuSurf.downloadKeypoints(cuKeypoints)
        keypoints = surf.detect(npMat)
        self.assertTrue(len(keypointsHost) == len(keypoints))

        cuKeypoints, cuDescriptors = cuSurf.detectWithDescriptors(cuMat,cv.cuda_GpuMat(),cuKeypoints,useProvidedKeypoints=True)
        keypointsHost = cuSurf.downloadKeypoints(cuKeypoints)
        descriptorsHost = cuDescriptors.download()
        keypoints, descriptors = surf.compute(npMat,keypoints)

        self.assertTrue(len(keypointsHost) == len(keypoints) and descriptorsHost.shape == descriptors.shape)

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()