#!/usr/bin/env python
import os
import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests
#from unittest import TestCase as NewOpenCVTests


class ovis_contrib_test(NewOpenCVTests):

    def setUp(self):
        super().setUp()
        # use software rendering
        os.environ["OPENCV_OVIS_RENDERSYSTEM"] = "Tiny Rendering Subsystem"
        # in case something goes wrong
        os.environ["OPENCV_OVIS_VERBOSE_LOG"] = "1"

    def test_multiWindow(self):
        win0 = cv.ovis.createWindow("main", (1, 1))
        win1 = cv.ovis.createWindow("other", (1, 1))
        del win1
        win1 = cv.ovis.createWindow("other", (1, 1))
        del win1

    def test_addResourceLocation(self):
        win0 = cv.ovis.createWindow("main", (1, 1))
        with self.assertRaises(cv.error):
            # must be called before the first createWindow
            cv.ovis.addResourceLocation(".")

    def test_texStride(self):
        win = cv.ovis.createWindow("main", (1, 1))
        data = np.zeros((200, 200), dtype=np.uint8)
        cv.ovis.createPlaneMesh("plane", (1, 1), data[50:-50, 50:-50])


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
