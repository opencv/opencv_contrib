import os
import cv2 as cv
import numpy as np

from tests_common import NewOpenCVTests, unittest

class nvidiaopticalflow_test(NewOpenCVTests):
    def setUp(self):
        super(nvidiaopticalflow_test, self).setUp()
        if not cv.cuda.getCudaEnabledDeviceCount():
            self.skipTest("No CUDA-capable device is detected")

    @unittest.skipIf('OPENCV_TEST_DATA_PATH' not in os.environ,
                        "OPENCV_TEST_DATA_PATH is not defined")
    def test_calc(self):
        frame1 = os.environ['OPENCV_TEST_DATA_PATH'] + '/gpu/opticalflow/frame0.png'
        frame2 = os.environ['OPENCV_TEST_DATA_PATH'] + '/gpu/opticalflow/frame1.png'

        npMat1 = cv.cvtColor(cv.imread(frame1),cv.COLOR_BGR2GRAY)
        npMat2 = cv.cvtColor(cv.imread(frame2),cv.COLOR_BGR2GRAY)

        cuMat1 = cv.cuda_GpuMat(npMat1)
        cuMat2 = cv.cuda_GpuMat(npMat2)
        try:
            nvof = cv.cuda_NvidiaOpticalFlow_1_0.create((npMat1.shape[1], npMat1.shape[0]), 5, False, False, False, 0)
            flow, cost = nvof.calc(cuMat1, cuMat2, None)
            self.assertTrue(flow.size()[1] > 0 and flow.size()[0] > 0)
            flowUpSampled = nvof.upSampler(flow, (npMat1.shape[1], npMat1.shape[0]), nvof.getGridSize(), None)
            nvof.collectGarbage()
            self.assertTrue(flowUpSampled.size()[1] > 0 and flowUpSampled.size()[0] > 0)
        except cv.error as e:
            if e.code == cv.Error.StsBadFunc or e.code == cv.Error.StsBadArg or e.code == cv.Error.StsNullPtr:
                self.skipTest("Algorithm is not supported in the current environment")

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
