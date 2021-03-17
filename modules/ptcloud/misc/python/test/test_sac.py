#!/usr/bin/env python

'''
sac segmentation
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests

class sac_test(NewOpenCVTests):

    def test_plane(self):
        N = 64;
        plane = np.zeros((N,N,3),np.float32)
        for i in range(0,N):
            for j in range(0,N):
                 plane[i,j] = (i,j,0)

        fit = cv.ptcloud.SACModelFitting_create(plane)
        mdl,left = fit.segment()

        self.assertEqual(len(mdl), 1)
        self.assertEqual(len(mdl[0].indices), N*N)
        self.assertEqual(len(mdl[0].points), N*N)
        self.assertEqual(len(mdl[0].coefficients), 4)
        self.assertEqual(np.shape(left), ())


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
