#!/usr/bin/env python
import os
import cv2 as cv

from tests_common import NewOpenCVTests, unittest

class test_dnn_superres(NewOpenCVTests):

    @unittest.skipIf('OPENCV_TEST_DATA_PATH' not in os.environ,
                     "OPENCV_TEST_DATA_PATH is not defined")
    def test_single_output(self):
        # Get test data paths
        dnn_superres_test_path = os.environ['OPENCV_TEST_DATA_PATH'] + "/cv/dnn_superres/"
        img_path = dnn_superres_test_path + "butterfly.png"
        espcn_path = dnn_superres_test_path + "ESPCN_x2.pb"

        # Create an SR object
        sr = cv.dnn_superres.DnnSuperResImpl_create()

        # Read image
        image = cv.imread(img_path)
        inp_h, inp_w, inp_c = image.shape

        # Read the desired model
        sr.readModel(espcn_path)

        # Set the desired model and scale to get correct pre- and post-processing
        sr.setModel("espcn", 2)

        # Upscale the image
        result = sr.upsample(image)
        out_h, out_w, out_c = result.shape

        # CHECK...
        # if result is not empty
        self.assertFalse(result is None)

        # upsampled image is correct size
        self.assertEqual(out_h, inp_h*2)
        self.assertEqual(out_w, inp_w*2)
        self.assertEqual(out_c, inp_c)

        # get functions work
        self.assertEqual(sr.getScale(), 2)
        self.assertEqual(sr.getAlgorithm(), "espcn")

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()