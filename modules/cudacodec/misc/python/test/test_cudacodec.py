#!/usr/bin/env python
import os
import cv2 as cv
import numpy as np

from tests_common import NewOpenCVTests, unittest

class cudacodec_test(NewOpenCVTests):
    def setUp(self):
        super(cudacodec_test, self).setUp()
        if not cv.cuda.getCudaEnabledDeviceCount():
            self.skipTest("No CUDA-capable device is detected")

    @unittest.skipIf('OPENCV_TEST_DATA_PATH' not in os.environ,
                     "OPENCV_TEST_DATA_PATH is not defined")
    def test_reader(self):
        #Test the functionality but not the results of the video reader

        vid_path = os.environ['OPENCV_TEST_DATA_PATH'] + '/cv/video/1920x1080.avi'
        try:
            reader = cv.cudacodec.createVideoReader(vid_path)
            ret, gpu_mat = reader.nextFrame()
            self.assertTrue(ret)
            self.assertTrue('GpuMat' in str(type(gpu_mat)), msg=type(gpu_mat))
            #TODO: print(cv.utils.dumpInputArray(gpu_mat)) # - no support for GpuMat

            # not checking output, therefore sepearate tests for different signatures is unecessary
            ret, _gpu_mat2 = reader.nextFrame(gpu_mat)
            #TODO: self.assertTrue(gpu_mat == gpu_mat2)
            self.assertTrue(ret)
        except cv.error as e:
            notSupported = (e.code == cv.Error.StsNotImplemented or e.code == cv.Error.StsUnsupportedFormat or e.code == cv.Error.GPU_API_CALL_ERROR)
            self.assertTrue(notSupported)
            if e.code == cv.Error.StsNotImplemented:
                self.skipTest("NVCUVID is not installed")
            elif e.code == cv.Error.StsUnsupportedFormat:
                self.skipTest("GPU hardware video decoder missing or video format not supported")
            elif e.code == cv.Error.GPU_API_CALL_ERRROR:
                self.skipTest("GPU hardware video decoder is missing")
            else:
                self.skipTest(e.err)

    def test_writer_existence(self):
        #Test at least the existence of wrapped functions for now

        try:
            _writer = cv.cudacodec.createVideoWriter("tmp", (128, 128), 30)
        except cv.error as e:
            self.assertEqual(e.code, cv.Error.StsNotImplemented)
            self.skipTest("NVCUVENC is not installed")

        self.assertTrue(True) #It is sufficient that no exceptions have been there

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()