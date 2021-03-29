#!/usr/bin/env python
import os
import cv2 as cv
import numpy as np

from tests_common import NewOpenCVTests, unittest

class cudaimgproc_test(NewOpenCVTests):
    def setUp(self):
        super(cudaimgproc_test, self).setUp()
        if not cv.cuda.getCudaEnabledDeviceCount():
            self.skipTest("No CUDA-capable device is detected")

    def test_cudaimgproc(self):
        npC1 = (np.random.random((128, 128)) * 255).astype(np.uint8)
        npC3 = (np.random.random((128, 128, 3)) * 255).astype(np.uint8)
        npC4 = (np.random.random((128, 128, 4)) * 255).astype(np.uint8)
        cuC1 = cv.cuda_GpuMat()
        cuC3 = cv.cuda_GpuMat()
        cuC4 = cv.cuda_GpuMat()
        cuC1.upload(npC1)
        cuC3.upload(npC3)
        cuC4.upload(npC4)

        cv.cuda.cvtColor(cuC3, cv.COLOR_RGB2HSV)
        cv.cuda.demosaicing(cuC1, cv.cuda.COLOR_BayerGR2BGR_MHT)
        cv.cuda.gammaCorrection(cuC3)
        cv.cuda.alphaComp(cuC4, cuC4, cv.cuda.ALPHA_XOR)
        cv.cuda.calcHist(cuC1)
        cv.cuda.equalizeHist(cuC1)
        cv.cuda.evenLevels(3, 0, 255)
        cv.cuda.meanShiftFiltering(cuC4, 10, 5)
        cv.cuda.meanShiftProc(cuC4, 10, 5)
        cv.cuda.bilateralFilter(cuC3, 3, 16, 3)
        cv.cuda.blendLinear

        cuRes = cv.cuda.meanShiftSegmentation(cuC4, 10, 5, 5)
        cuDst = cv.cuda_GpuMat(cuC4.size(),cuC4.type())
        cv.cuda.meanShiftSegmentation(cuC4, 10, 5, 5, cuDst)
        self.assertTrue(np.allclose(cuRes.download(),cuDst.download()))

        clahe = cv.cuda.createCLAHE()
        clahe.apply(cuC1, cv.cuda_Stream.Null())

        histLevels = cv.cuda.histEven(cuC3, 20, 0, 255)
        cv.cuda.histRange(cuC1, histLevels)

        detector = cv.cuda.createCannyEdgeDetector(0, 100)
        detector.detect(cuC1)

        detector = cv.cuda.createHoughLinesDetector(3, np.pi / 180, 20)
        detector.detect(cuC1)

        detector = cv.cuda.createHoughSegmentDetector(3, np.pi / 180, 20, 5)
        detector.detect(cuC1)

        detector = cv.cuda.createHoughCirclesDetector(3, 20, 10, 10, 20, 100)
        detector.detect(cuC1)

        detector = cv.cuda.createGeneralizedHoughBallard()
        #BUG: detect accept only Mat!
        #Even if generate_gpumat_decls is set to True, it only wraps overload CUDA functions.
        #The problem is that Mat and GpuMat are not fully compatible to enable system-wide overloading
        #detector.detect(cuC1, cuC1, cuC1)

        detector = cv.cuda.createGeneralizedHoughGuil()
        #BUG: same as above..
        #detector.detect(cuC1, cuC1, cuC1)

        detector = cv.cuda.createHarrisCorner(cv.CV_8UC1, 15, 5, 1)
        detector.compute(cuC1)

        detector = cv.cuda.createMinEigenValCorner(cv.CV_8UC1, 15, 5, 1)
        detector.compute(cuC1)

        detector = cv.cuda.createGoodFeaturesToTrackDetector(cv.CV_8UC1)
        detector.detect(cuC1)

        matcher = cv.cuda.createTemplateMatching(cv.CV_8UC1, cv.TM_CCOEFF_NORMED)
        matcher.match(cuC3, cuC3)

        self.assertTrue(True) #It is sufficient that no exceptions have been there

    def test_cvtColor(self):
        npMat = (np.random.random((128, 128, 3)) * 255).astype(np.uint8)
        cuMat = cv.cuda_GpuMat()
        cuMat.upload(npMat)

        self.assertTrue(np.allclose(cv.cuda.cvtColor(cuMat, cv.COLOR_BGR2HSV).download(),
                                         cv.cvtColor(npMat, cv.COLOR_BGR2HSV)))

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()