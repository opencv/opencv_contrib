#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function

import os

import cv2 as cv

from tests_common import NewOpenCVTests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_DIR = os.path.join(SCRIPT_DIR, '../../../')

class shape_test(NewOpenCVTests):

    def test_computeDistance(self):

        a = cv.imread(os.path.join(MODULE_DIR, 'samples/data/shape_sample/1.png'), cv.IMREAD_GRAYSCALE)
        b = cv.imread(os.path.join(MODULE_DIR, 'samples/data/shape_sample/2.png'), cv.IMREAD_GRAYSCALE)
        if a is None or b is None:
            raise unittest.SkipTest("Missing files with test data")

        ca, _ = cv.findContours(a, cv.RETR_CCOMP, cv.CHAIN_APPROX_TC89_KCOS)
        cb, _ = cv.findContours(b, cv.RETR_CCOMP, cv.CHAIN_APPROX_TC89_KCOS)

        hd = cv.createHausdorffDistanceExtractor()
        sd = cv.createShapeContextDistanceExtractor()

        d1 = hd.computeDistance(ca[0], cb[0])
        d2 = sd.computeDistance(ca[0], cb[0])

        self.assertAlmostEqual(d1, 26.4196891785, 3, "HausdorffDistanceExtractor")
        self.assertAlmostEqual(d2, 0.25804194808, 3, "ShapeContextDistanceExtractor")

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
