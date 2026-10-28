#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests

class surface_matching_test(NewOpenCVTests):

    def test_point_cloud_layouts_issue_3981(self):
        u, v = np.meshgrid(np.linspace(0.3, 1.3, 13), np.linspace(0.2, 1.4, 15))
        a, b, c = 10.0, 17.0, 25.0
        pts = np.stack([a * np.sin(u) * np.cos(v), b * np.sin(u) * np.sin(v), c * np.cos(u)], axis=-1).reshape(-1, 3)
        normals = pts / np.array([a * a, b * b, c * c])
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        model = np.hstack([pts, normals]).astype(np.float32)

        detector = cv.ppf_match_3d_PPF3DDetector(0.08, 0.08)
        detector.trainModel(model)
        ref = detector.match(model, 0.2, 0.08)
        self.assertTrue(len(ref) > 0)

        for layout in [(-1, 1, 6), (1, -1, 6)]:
            pc = model.reshape(layout)
            detector = cv.ppf_match_3d_PPF3DDetector(0.08, 0.08)
            detector.trainModel(pc)
            poses = detector.match(pc, 0.2, 0.08)
            self.assertEqual(len(ref), len(poses))
            self.assertLess(cv.norm(ref[0].pose, poses[0].pose, cv.NORM_INF), 1e-5)

            icp = cv.ppf_match_3d_ICP(50, 0.0001, 2.5, 3)
            retval, residual, pose = icp.registerModelToScene(pc, pc)
            self.assertEqual(0, retval)


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
