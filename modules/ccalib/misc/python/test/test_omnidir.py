#!/usr/bin/env python
import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests

class omnidir_test(NewOpenCVTests):
    def test_non_contiguous_points_issue_3625(self):
        K = np.array([[500.0, 0.0, 500.0], [0.0, 500.0, 500.0], [0.0, 0.0, 1.0]])
        D = np.array([[0.1, -0.05, 0.001, 0.002]])
        xi = 0.8
        rvec = np.array([[0.1, -0.2, 0.05]])
        tvec = np.array([[0.2, 0.1, 4.0]])
        grid = np.mgrid[100:900:5j, 100:900:4j].T.reshape(-1, 2)

        # slicing away an extra column gives a non-contiguous view, as in the issue
        image_points = np.hstack([grid, np.ones((len(grid), 1))])[:, :2].reshape(-1, 1, 2)
        self.assertFalse(image_points.flags['C_CONTIGUOUS'])
        undistorted = cv.omnidir.undistortPoints(image_points, K, D, np.array([[xi]]), np.eye(3))
        expected = cv.omnidir.undistortPoints(image_points.copy(), K, D, np.array([[xi]]), np.eye(3))
        self.assertLess(cv.norm(undistorted, expected, cv.NORM_INF), 1e-9)

        object_points = np.hstack([grid / 1000.0 - 0.5, np.zeros((len(grid), 2))])[:, :3].reshape(-1, 1, 3)
        self.assertFalse(object_points.flags['C_CONTIGUOUS'])
        projected, _ = cv.omnidir.projectPoints(object_points, rvec, tvec, K, xi, D)
        expected, _ = cv.omnidir.projectPoints(object_points.copy(), rvec, tvec, K, xi, D)
        self.assertLess(cv.norm(projected, expected, cv.NORM_INF), 1e-9)


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
