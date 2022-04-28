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

    def test_write_read_dict(self):

        try:
            aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_50)
            markers_gold = aruco_dict.bytesList

            # write aruco_dict
            filename = "test_dict.yml"
            fs_write = cv.FileStorage(filename, cv.FileStorage_WRITE)
            aruco_dict.writeDictionary(fs_write)
            fs_write.release()

            # reset aruco_dict
            aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)

            # read aruco_dict
            fs_read = cv.FileStorage(filename, cv.FileStorage_READ)
            aruco_dict.readDictionary(fs_read.root())
            fs_read.release()

            # check equal
            self.assertEqual(aruco_dict.markerSize, 5)
            self.assertEqual(aruco_dict.maxCorrectionBits, 3)
            np.testing.assert_array_equal(aruco_dict.bytesList, markers_gold)

        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_identify(self):
        aruco_dict = cv.aruco.Dictionary_get(cv.aruco.DICT_4X4_50)
        expected_idx = 9
        expected_rotation = 2
        bit_marker = np.array([[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 1, 1], [0, 0, 1, 1]], dtype=np.uint8)

        check, idx, rotation = aruco_dict.identify(bit_marker, 0)

        self.assertTrue(check, True)
        self.assertEqual(idx, expected_idx)
        self.assertEqual(rotation, expected_rotation)

    def test_getDistanceToId(self):
        aruco_dict = cv.aruco.Dictionary_get(cv.aruco.DICT_4X4_50)
        idx = 7
        rotation = 3
        bit_marker = np.array([[0, 1, 0, 1], [0, 1, 1, 1], [1, 1, 0, 0], [0, 1, 0, 0]], dtype=np.uint8)
        dist = aruco_dict.getDistanceToId(bit_marker, idx)

        self.assertEqual(dist, 0)

    def test_aruco_detector(self):
        aruco_params = cv.aruco.DetectorParameters_create()
        aruco_dict = cv.aruco.Dictionary_get(cv.aruco.DICT_4X4_250)
        aruco_detector = cv.aruco.ArucoDetector_create(aruco_dict, aruco_params)
        id = 2
        marker_size = 100
        offset = 10
        img_marker = cv.aruco.drawMarker(aruco_dict, id, marker_size, aruco_params.markerBorderBits)
        img_marker = np.pad(img_marker, pad_width=offset, mode='constant', constant_values=255)
        gold_corners = np.array([[offset, offset],[marker_size+offset-1.0,offset],
                                 [marker_size+offset-1.0,marker_size+offset-1.0],
                                 [offset, marker_size+offset-1.0]], dtype=np.float32)
        expected_corners, expected_ids, expected_rejected = cv.aruco.detectMarkers(img_marker, aruco_dict,
                                                                                   parameters=aruco_params)

        corners, ids, rejected = aruco_detector.detectMarkers(img_marker)

        self.assertEqual(1, len(ids))
        self.assertEqual(id, ids[0])
        for i in range(0, len(ids)):
            np.testing.assert_array_equal(expected_corners[i], corners[i])
            np.testing.assert_array_equal(gold_corners, corners[i].reshape(4, 2))

    def test_aruco_detector_refine(self):
        aruco_params = cv.aruco.DetectorParameters_create()
        aruco_dict = cv.aruco.Dictionary_get(cv.aruco.DICT_4X4_250)
        aruco_detector = cv.aruco.ArucoDetector_create(aruco_dict, aruco_params)
        board_size = (3, 4)
        board = cv.aruco.GridBoard_create(board_size[0], board_size[1], 5.0, 1.0, aruco_dict)
        board_image = board.draw((board_size[0]*50, board_size[1]*50), marginSize=10)

        corners, ids, rejected = aruco_detector.detectMarkers(board_image)
        self.assertEqual(board_size[0]*board_size[1], len(ids))

        part_corners, part_ids, part_rejected = corners[:-1], ids[:-1], list(rejected)
        part_rejected.append(corners[-1])

        refine_corners, refine_ids, refine_rejected, recovered_ids = aruco_detector.refineDetectedMarkers(board_image, board, part_corners, part_ids, part_rejected)
        refine_corners_c, _, _, _ = cv.aruco.refineDetectedMarkers(board_image, board, part_corners, part_ids, part_rejected)

        self.assertEqual(board_size[0] * board_size[1], len(refine_ids))
        self.assertEqual(1, len(recovered_ids))

        for i in range(0, len(ids)):
            np.testing.assert_array_equal(refine_corners_c[i], refine_corners[i])
        #self.assertEqual(ids[-1], recovered_ids[0])
        self.assertEqual(ids[-1], refine_ids[-1])
        self.assertEqual((1, 4, 2), refine_corners[0].shape)
        np.testing.assert_array_equal(corners, refine_corners)

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
