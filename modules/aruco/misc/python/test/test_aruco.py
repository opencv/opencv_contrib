#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function

import os, numpy as np

import cv2 as cv

from tests_common import NewOpenCVTests

class aruco_test(NewOpenCVTests):

    def test_aruco_detect_markers(self):
        """Original test — new API, basic detection."""
        aruco_params = cv.aruco.DetectorParameters()
        aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250)
        id = 2
        marker_size = 100
        offset = 10
        img_marker = cv.aruco.generateImageMarker(aruco_dict, id, marker_size, aruco_params.markerBorderBits)
        img_marker = np.pad(img_marker, pad_width=offset, mode='constant', constant_values=255)
        gold_corners = np.array([[offset, offset], [marker_size+offset-1.0, offset],
                                 [marker_size+offset-1.0, marker_size+offset-1.0],
                                 [offset, marker_size+offset-1.0]], dtype=np.float32)
        expected_corners, expected_ids, expected_rejected = cv.aruco.detectMarkers(
            img_marker, aruco_dict, parameters=aruco_params)

        self.assertEqual(1, len(expected_ids))
        self.assertEqual(id, expected_ids[0])
        for i in range(0, len(expected_corners)):
            np.testing.assert_array_equal(gold_corners, expected_corners[i].reshape(4, 2))

    def test_drawCharucoDiamond(self):
        """Original test — draw a charuco diamond."""
        aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
        img = cv.aruco.drawCharucoDiamond(aruco_dict, np.array([0, 1, 2, 3]), 100, 80)
        self.assertTrue(img is not None)

    # ------------------------------------------------------------------
    # ARM / Raspberry Pi regression tests for issue #3938
    # These verify the SIGSEGV fix: detectMarkers must NOT crash when
    # called with the new API objects (getPredefinedDictionary /
    # DetectorParameters) on ARM/aarch64 platforms.
    # ------------------------------------------------------------------

    def test_aruco_detect_markers_new_api_no_crash(self):
        """Regression: new API must not SIGSEGV on ARM (issue #3938)."""
        gray = np.zeros((480, 640), dtype=np.uint8)
        aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_50)
        parameters = cv.aruco.DetectorParameters()
        # Must not crash — if we reach the assertions below, fix is working
        corners, ids, rejected = cv.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        self.assertIsNotNone(corners)
        self.assertIsNotNone(rejected)

    def test_aruco_detect_markers_old_api_no_crash(self):
        """Regression: old API must still work (original workaround path)."""
        gray = np.zeros((480, 640), dtype=np.uint8)
        dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_5X5_50)
        parameters = cv.aruco.DetectorParameters_create()
        corners, ids, rejected = cv.aruco.detectMarkers(gray, dictionary, parameters=parameters)
        self.assertIsNotNone(corners)
        self.assertIsNotNone(rejected)

    def test_aruco_detect_markers_both_apis_consistent(self):
        """Regression: new API and old API must produce identical results."""
        aruco_dict_new = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250)
        params_new = cv.aruco.DetectorParameters()

        aruco_dict_old = cv.aruco.Dictionary_get(cv.aruco.DICT_4X4_250)
        params_old = cv.aruco.DetectorParameters_create()

        id = 5
        marker_size = 120
        offset = 15
        img_marker = cv.aruco.generateImageMarker(aruco_dict_new, id, marker_size,
                                                  params_new.markerBorderBits)
        img_marker = np.pad(img_marker, pad_width=offset, mode='constant', constant_values=255)

        corners_new, ids_new, _ = cv.aruco.detectMarkers(img_marker, aruco_dict_new,
                                                          parameters=params_new)
        corners_old, ids_old, _ = cv.aruco.detectMarkers(img_marker, aruco_dict_old,
                                                          parameters=params_old)

        self.assertEqual(len(ids_new), len(ids_old))
        self.assertEqual(ids_new[0], ids_old[0])
        np.testing.assert_array_almost_equal(corners_new[0], corners_old[0], decimal=1)

    def test_detect_markers_null_params_raises(self):
        """Null parameters must raise cv2.error, not SIGSEGV."""
        gray = np.zeros((480, 640), dtype=np.uint8)
        aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_50)
        # Passing None as parameters — must raise, not crash
        with self.assertRaises((cv.error, Exception)):
            cv.aruco.detectMarkers(gray, aruco_dict, parameters=None)

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
