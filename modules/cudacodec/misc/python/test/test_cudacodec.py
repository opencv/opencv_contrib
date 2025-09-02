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
        # Test the functionality but not the results of the VideoReader

        vid_path = os.environ['OPENCV_TEST_DATA_PATH'] + '/highgui/video/big_buck_bunny.h264'
        try:
            reader = cv.cudacodec.createVideoReader(vid_path)
            format_info = reader.format()
            ret, gpu_mat = reader.nextFrame()
            self.assertTrue(ret)
            self.assertTrue(isinstance(gpu_mat, cv.cuda.GpuMat), msg=type(gpu_mat))
            #TODO: print(cv.utils.dumpInputArray(gpu_mat)) # - no support for GpuMat

            # Retrieve format info
            if(not format_info.valid):
              format_info = reader.format()
            sz = gpu_mat.size()
            self.assertTrue(sz[0] == format_info.width and sz[1] == format_info.height)

            # not checking output, therefore sepearate tests for different signatures is unecessary
            ret, gpu_mat_ = reader.nextFrame(gpu_mat)
            self.assertTrue(ret and gpu_mat_.cudaPtr() == gpu_mat.cudaPtr())

            # Pass VideoReaderInitParams to the decoder and initialization params to the source (cv::VideoCapture)
            params = cv.cudacodec.VideoReaderInitParams()
            params.rawMode = True
            params.enableHistogram = False
            ms_gs = 1234
            post_processed_sz = (gpu_mat.size()[0]*2, gpu_mat.size()[1]*2)
            params.targetSz = post_processed_sz
            reader = cv.cudacodec.createVideoReader(vid_path,[cv.CAP_PROP_OPEN_TIMEOUT_MSEC, ms_gs], params)
            ret, ms = reader.get(cv.CAP_PROP_OPEN_TIMEOUT_MSEC)
            self.assertTrue(ret and ms == ms_gs)
            ret, raw_mode = reader.getVideoReaderProps(cv.cudacodec.VideoReaderProps_PROP_RAW_MODE)
            self.assertTrue(ret and raw_mode)

            # Read raw encoded bitstream
            ret, i_base = reader.getVideoReaderProps(cv.cudacodec.VideoReaderProps_PROP_RAW_PACKAGES_BASE_INDEX)
            self.assertTrue(ret and i_base == 2.0)
            ret, gpu_mat_2 = reader.nextFrame()
            self.assertTrue(ret and isinstance(gpu_mat_2,cv.cuda.GpuMat) and not gpu_mat_2.empty())
            ret = reader.retrieve(gpu_mat_2)
            self.assertTrue(ret and isinstance(gpu_mat_2,cv.cuda.GpuMat) and not gpu_mat_2.empty())
            ret, n_raw_packages_since_last_grab = reader.getVideoReaderProps(cv.cudacodec.VideoReaderProps_PROP_NUMBER_OF_RAW_PACKAGES_SINCE_LAST_GRAB)
            self.assertTrue(ret and n_raw_packages_since_last_grab > 0)
            self.assertTrue(reader.rawPackageHasKeyFrame(int(i_base)))
            ret, raw_data = reader.retrieve(int(i_base))
            self.assertTrue(ret and isinstance(raw_data,np.ndarray) and np.any(raw_data))

            # Check post processing applied
            self.assertTrue(gpu_mat_2.size() == post_processed_sz)

            # Retrieve image histogram. Not all GPUs support histogram. Just check the method is called correctly
            ret, gpu_mat_3, hist = reader.nextFrameWithHist()
            self.assertTrue(ret and not gpu_mat_3.empty())
            ret, gpu_mat_3_, hist_ = reader.nextFrameWithHist(gpu_mat_3, hist)
            self.assertTrue(ret and not gpu_mat_3.empty())
            self.assertTrue(gpu_mat_3_.cudaPtr() == gpu_mat_3.cudaPtr())

            # Change color format
            ret, colour_code = reader.getVideoReaderProps(cv.cudacodec.VideoReaderProps_PROP_COLOR_FORMAT)
            self.assertTrue(ret and colour_code == cv.cudacodec.BGRA)
            colour_code_gs = cv.cudacodec.GRAY
            reader.set(colour_code_gs)
            ret, colour_code = reader.getVideoReaderProps(cv.cudacodec.VideoReaderProps_PROP_COLOR_FORMAT)
            self.assertTrue(ret and colour_code == colour_code_gs)

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

    def test_map_histogram(self):
        hist = cv.cuda_GpuMat((1,256), cv.CV_8UC1)
        hist.setTo(1)
        hist_host = cv.cudacodec.MapHist(hist)
        self.assertTrue(hist_host.shape == (256, 1) and isinstance(hist_host, np.ndarray))

    def test_writer(self):
        # Test the functionality but not the results of the VideoWriter

        try:
            fd, fname = tempfile.mkstemp(suffix=".h264")
            os.close(fd)
            encoder_params_in = cv.cudacodec.EncoderParams()
            encoder_params_in.gopLength = 10
            stream = cv.cuda.Stream()
            sz = (1920,1080)
            writer = cv.cudacodec.createVideoWriter(fname, sz, cv.cudacodec.H264, 30, cv.cudacodec.BGR,
                                                    encoder_params_in, stream=stream)
            blankFrameIn = cv.cuda.GpuMat(sz,cv.CV_8UC3)
            writer.write(blankFrameIn)
            writer.release()
            encoder_params_out = writer.getEncoderParams()
            self.assertTrue(encoder_params_in.gopLength == encoder_params_out.gopLength)
            cap = cv.VideoCapture(fname,cv.CAP_FFMPEG)
            self.assertTrue(cap.isOpened())
            ret, blankFrameOut = cap.read()
            self.assertTrue(ret and blankFrameOut.shape == blankFrameIn.download().shape)
            cap.release()
        except cv.error as e:
            self.assertEqual(e.code, cv.Error.StsNotImplemented)
            self.skipTest("Either NVCUVENC or a GPU hardware encoder is missing or the encoding profile is not supported.")

        os.remove(fname)

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
