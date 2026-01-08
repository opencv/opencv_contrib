#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function

import os
import cv2 as cv
import numpy as np

from tests_common import NewOpenCVTests, unittest

class intensity_transform_test(NewOpenCVTests):
    def setUp(self):
        super(intensity_transform_test, self).setUp()
        try:
            result_ = cv.intensity_transform.BIMEF(None)
        except cv.error as e:
            if e.code == cv.Error.StsNotImplemented:
                self.skipTest('BIMEF is not implemented (missing Eigen dependency)')

    @unittest.skipIf('OPENCV_TEST_DATA_PATH' not in os.environ,
                     "OPENCV_TEST_DATA_PATH is not defined")
    def test_BIMEF(self):
        filenames = ['P1000205_resize', 'P1010676_resize', 'P1010815_resize']

        for f in filenames:
            img = self.get_sample('cv/intensity_transform/BIMEF/{}.png'.format(f))
            self.assertTrue(img.size > 0)

            img_ref = self.get_sample('cv/intensity_transform/BIMEF/{}_ref.png'.format(f))
            self.assertTrue(img_ref.size > 0)

            img_BIMEF = cv.intensity_transform.BIMEF(img)
            self.assertTrue(img_BIMEF.size > 0)
            self.assertTrue(img_BIMEF.shape == img_ref.shape)
            self.assertTrue(img_BIMEF.dtype == img_ref.dtype)

            RMSE = np.sqrt(cv.norm(img_BIMEF, img_ref, cv.NORM_L2SQR) / (img_ref.shape[0]*img_ref.shape[1]*img_ref.shape[2]))
            max_RMSE_threshold = 9.0
            self.assertLessEqual(RMSE, max_RMSE_threshold)
            print('BIMEF RMSE:', RMSE)

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
