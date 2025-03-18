#!/usr/bin/env python
# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.

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

    @unittest.skipIf('OPENCV_TEST_DATA_PATH' not in os.environ,
                     "OPENCV_TEST_DATA_PATH is not defined")
    def test_srgan(self):
        # Get test data paths
        dnn_superres_test_path = os.environ['OPENCV_TEST_DATA_PATH'] + "/cv/dnn_superres/"
        img_path = dnn_superres_test_path + "butterfly.png"
        srgan_path = dnn_superres_test_path + "SRGAN_x4.pb"

        # Create an SR object
        sr = cv.dnn_superres.DnnSuperResImpl_create()

        # Read image
        image = cv.imread(img_path)
        inp_h, inp_w, inp_c = image.shape

        # Read the desired model
        sr.readModel(srgan_path)

        # Set the desired model and scale to get correct pre- and post-processing
        sr.setModel("srgan", 4)

        # Upscale the image
        result = sr.upsample(image)
        out_h, out_w, out_c = result.shape

        # CHECK...
        # if result is not empty
        self.assertFalse(result is None)

        # upsampled image is correct size
        self.assertEqual(out_h, inp_h*4)
        self.assertEqual(out_w, inp_w*4)
        self.assertEqual(out_c, inp_c)

        # get functions work
        self.assertEqual(sr.getScale(), 4)
        self.assertEqual(sr.getAlgorithm(), "srgan")

    @unittest.skipIf('OPENCV_TEST_DATA_PATH' not in os.environ,
                     "OPENCV_TEST_DATA_PATH is not defined")
    def test_rdn(self):
        # Get test data paths
        dnn_superres_test_path = os.environ['OPENCV_TEST_DATA_PATH'] + "/cv/dnn_superres/"
        img_path = dnn_superres_test_path + "butterfly.png"
        rdn_path = dnn_superres_test_path + "RDN_x3.pb"

        # Create an SR object
        sr = cv.dnn_superres.DnnSuperResImpl_create()

        # Read image
        image = cv.imread(img_path)
        inp_h, inp_w, inp_c = image.shape

        # Read the desired model
        sr.readModel(rdn_path)

        # Set the desired model and scale to get correct pre- and post-processing
        sr.setModel("rdn", 3)

        # Upscale the image
        result = sr.upsample(image)
        out_h, out_w, out_c = result.shape

        # CHECK...
        # if result is not empty
        self.assertFalse(result is None)

        # upsampled image is correct size
        self.assertEqual(out_h, inp_h*3)
        self.assertEqual(out_w, inp_w*3)
        self.assertEqual(out_c, inp_c)

        # get functions work
        self.assertEqual(sr.getScale(), 3)
        self.assertEqual(sr.getAlgorithm(), "rdn")

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
