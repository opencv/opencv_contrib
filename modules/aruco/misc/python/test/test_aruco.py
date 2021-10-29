#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function

import os, numpy as np

import cv2 as cv

from tests_common import NewOpenCVTests

class aruco_test(NewOpenCVTests):

    def test_idsAccessibility(self):

        ids = np.arange(17)
        rev_ids = ids[::-1]

        aruco_dict  = cv.aruco.Dictionary_get(cv.aruco.DICT_5X5_250)
        board = cv.aruco.CharucoBoard_create(7, 5, 1, 0.5, aruco_dict)

        np.testing.assert_array_equal(board.ids.squeeze(), ids)

        board.ids = rev_ids
        np.testing.assert_array_equal(board.ids.squeeze(), rev_ids)

        board.setIds(ids)
        np.testing.assert_array_equal(board.ids.squeeze(), ids)

        with self.assertRaises(cv.error):
            board.setIds(np.array([0]))

    def test_drawCharucoDiamond(self):
        aruco_dict = cv.aruco.Dictionary_get(cv.aruco.DICT_4X4_50)
        img = cv.aruco.drawCharucoDiamond(aruco_dict, np.array([0, 1, 2, 3]), 100, 80)
        self.assertTrue(img is not None)

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
