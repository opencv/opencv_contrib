#!/usr/bin/env python
import os
import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests, unittest

class tracking_contrib_test(NewOpenCVTests):

    def test_createTracker(self):

        t = cv.TrackerMIL_create()
        t = cv.TrackerKCF_create()
        try:
            t = cv.TrackerGOTURN_create()
        except cv.error as e:
            pass  # may fail due to missing DL model files

    def test_createLegacyTracker(self):

        t = cv.legacy.TrackerBoosting_create()
        t = cv.legacy.TrackerMIL_create()
        t = cv.legacy.TrackerKCF_create()
        t = cv.legacy.TrackerMedianFlow_create()
        #t = cv.legacy.TrackerGOTURN_create()
        t = cv.legacy.TrackerMOSSE_create()
        t = cv.legacy.TrackerCSRT_create()


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
