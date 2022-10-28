#!/usr/bin/env python
import os
import cv2 as cv
import numpy as np
import tempfile
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
            format_info = reader.format()
            ret, gpu_mat = reader.nextFrame()
            self.assertTrue(ret)
            self.assertTrue('GpuMat' in str(type(gpu_mat)), msg=type(gpu_mat))
            #TODO: print(cv.utils.dumpInputArray(gpu_mat)) # - no support for GpuMat

            if(not format_info.valid):
              format_info = reader.format()
            sz = gpu_mat.size()
            self.assertTrue(sz[0] == format_info.width and sz[1] == format_info.height)

            # not checking output, therefore sepearate tests for different signatures is unecessary
            ret, _gpu_mat2 = reader.nextFrame(gpu_mat)
            #TODO: self.assertTrue(gpu_mat == gpu_mat2)
            self.assertTrue(ret)

            params = cv.cudacodec.VideoReaderInitParams()
            params.rawMode = True
            ms_gs = 1234
            reader = cv.cudacodec.createVideoReader(vid_path,[cv.CAP_PROP_OPEN_TIMEOUT_MSEC, ms_gs], params)
            ret, ms = reader.get(cv.CAP_PROP_OPEN_TIMEOUT_MSEC)
            self.assertTrue(ret and ms == ms_gs)
            ret, raw_mode = reader.getVideoReaderProps(cv.cudacodec.VideoReaderProps_PROP_RAW_MODE)
            self.assertTrue(ret and raw_mode)

            ret, colour_code = reader.getVideoReaderProps(cv.cudacodec.VideoReaderProps_PROP_COLOR_FORMAT)
            self.assertTrue(ret and colour_code == cv.cudacodec.ColorFormat_BGRA)
            colour_code_gs = cv.cudacodec.ColorFormat_GRAY
            reader.set(colour_code_gs)
            ret, colour_code = reader.getVideoReaderProps(cv.cudacodec.VideoReaderProps_PROP_COLOR_FORMAT)
            self.assertTrue(ret and colour_code == colour_code_gs)

            ret, i_base = reader.getVideoReaderProps(cv.cudacodec.VideoReaderProps_PROP_RAW_PACKAGES_BASE_INDEX)
            self.assertTrue(ret and i_base == 2.0)
            self.assertTrue(reader.grab())
            ret, gpu_mat3 = reader.retrieve()
            self.assertTrue(ret and isinstance(gpu_mat3,cv.cuda.GpuMat) and not gpu_mat3.empty())
            ret = reader.retrieve(gpu_mat3)
            self.assertTrue(ret and isinstance(gpu_mat3,cv.cuda.GpuMat) and not gpu_mat3.empty())
            ret, n_raw_packages_since_last_grab = reader.getVideoReaderProps(cv.cudacodec.VideoReaderProps_PROP_NUMBER_OF_RAW_PACKAGES_SINCE_LAST_GRAB)
            self.assertTrue(ret and n_raw_packages_since_last_grab > 0)
            ret, raw_data = reader.retrieve(int(i_base))
            self.assertTrue(ret and isinstance(raw_data,np.ndarray) and np.any(raw_data))

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
            fd, fname = tempfile.mkstemp(suffix=".h264")
            os.close(fd)
            encoder_params_in = cv.cudacodec.EncoderParams()
            encoder_params_in.gopLength = 10
            stream = cv.cuda.Stream()
            sz = (1920,1080)
            writer = cv.cudacodec.createVideoWriter(fname, sz, cv.cudacodec.H264, 30, cv.cudacodec.ColorFormat_BGR,
                                                    encoder_params_in, stream=stream)
            blankFrameIn = cv.cuda.GpuMat(sz,cv.CV_8UC3)
            writer.write(blankFrameIn)
            writer.release()
            encoder_params_out = writer.getEncoderParams()
            self.assert_true(encoder_params_in.gopLength == encoder_params_out.gopLength)
            cap = cv.VideoCapture(fname,cv.CAP_FFMPEG)
            self.assert_true(cap.isOpened())
            ret, blankFrameOut = cap.read()
            self.assert_true(ret and blankFrameOut.shape == blankFrameIn.download().shape)
        except cv.error as e:
            self.assertEqual(e.code, cv.Error.StsNotImplemented)
            self.skipTest("Either NVCUVENC or a GPU hardware encoder is missing or the encoding profile is not supported.")

        os.remove(fname)

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()