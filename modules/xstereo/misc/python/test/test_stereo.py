#!/usr/bin/env python
import cv2 as cv

from tests_common import NewOpenCVTests

class quasi_dense_stereo_test(NewOpenCVTests):

    def test_simple(self):

        stereo = cv.stereo.QuasiDenseStereo_create((100, 100))
        self.assertIsNotNone(stereo)

        dense_matches = cv.stereo_MatchQuasiDense()
        self.assertIsNotNone(dense_matches)

        parameters = cv.stereo_PropagationParameters()
        self.assertIsNotNone(parameters)

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
