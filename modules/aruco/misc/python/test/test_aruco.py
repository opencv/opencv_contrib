#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function

import os, numpy as np

import cv2 as cv

from tests_common import NewOpenCVTests

class aruco_test(NewOpenCVTests):

    def test_idsAccessibility(self):

        ids = np.array([[elem] for elem in range(17)])
        rev_ids = np.array(list(reversed(ids)))

        aruco_dict  = cv.aruco.Dictionary_get(cv.aruco.DICT_5X5_250)
        board = cv.aruco.CharucoBoard_create(7, 5, 1, 0.5, aruco_dict)

        self.assertTrue(np.equal(board.ids, ids).all())

        board.ids = rev_ids
        self.assertTrue(np.equal(board.ids, rev_ids).all())

        board.setIds(ids)
        self.assertTrue(np.equal(board.ids, ids).all())

        with self.assertRaises(cv.error):
            board.setIds(np.array([0]))

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
