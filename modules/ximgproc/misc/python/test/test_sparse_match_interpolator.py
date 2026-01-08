#!/usr/bin/env python
import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests

class Interpolator_test(NewOpenCVTests):
    def test_edgeaware_interpolator(self):
        # readGT
        MAX_DIF = 1.0
        MAX_MEAN_DIF = 1.0 / 256.0

        src = cv.imread(self.find_file("cv/optflow/RubberWhale1.png"), cv.IMREAD_COLOR)
        self.assertFalse(src is None)

        ref_flow = cv.readOpticalFlow(self.find_file("cv/sparse_match_interpolator/RubberWhale_reference_result.flo"))
        self.assertFalse(ref_flow is None)

        matches = np.genfromtxt(self.find_file("cv/sparse_match_interpolator/RubberWhale_sparse_matches.txt")).astype(np.float32)
        from_points = matches[:,0:2]
        to_points = matches[:,2:4]
        interpolator = cv.ximgproc.createEdgeAwareInterpolator()
        interpolator.setK(128)
        interpolator.setSigma(0.05)
        interpolator.setUsePostProcessing(True)
        interpolator.setFGSLambda(500.0)
        interpolator.setFGSSigma(1.5)

        dense_flow = interpolator.interpolate(src, from_points, src, to_points)

        self.assertTrue(cv.norm(dense_flow, ref_flow, cv.NORM_INF) <= MAX_DIF)
        self.assertTrue(cv.norm(dense_flow, ref_flow, cv.NORM_L1) <= (MAX_MEAN_DIF * dense_flow.shape[0] * dense_flow.shape[1]))

    def test_ric_interpolator(self):
        # readGT
        MAX_DIF = 6.0
        MAX_MEAN_DIF = 60.0 / 256.0

        src0 = cv.imread(self.find_file("cv/optflow/RubberWhale1.png"), cv.IMREAD_COLOR)
        self.assertFalse(src0 is None)

        src1 = cv.imread(self.find_file("cv/optflow/RubberWhale2.png"), cv.IMREAD_COLOR)
        self.assertFalse(src1 is None)

        ref_flow = cv.readOpticalFlow(self.find_file("cv/sparse_match_interpolator/RubberWhale_reference_result.flo"))
        self.assertFalse(ref_flow is None)

        matches = np.genfromtxt(self.find_file("cv/sparse_match_interpolator/RubberWhale_sparse_matches.txt")).astype(np.float32)
        from_points = matches[:,0:2]
        to_points = matches[:,2:4]

        interpolator = cv.ximgproc.createRICInterpolator()
        interpolator.setK(32)
        interpolator.setSuperpixelSize(15)
        interpolator.setSuperpixelNNCnt(150)
        interpolator.setSuperpixelRuler(15.0)
        interpolator.setSuperpixelMode(cv.ximgproc.SLIC)
        interpolator.setAlpha(0.7)
        interpolator.setModelIter(4)
        interpolator.setRefineModels(True)
        interpolator.setMaxFlow(250)
        interpolator.setUseVariationalRefinement(True)
        interpolator.setUseGlobalSmootherFilter(True)
        interpolator.setFGSLambda(500.0)
        interpolator.setFGSSigma(1.5)
        dense_flow = interpolator.interpolate(src0, from_points, src1, to_points)
        self.assertTrue(cv.norm(dense_flow, ref_flow, cv.NORM_INF) <= MAX_DIF)
        self.assertTrue(cv.norm(dense_flow, ref_flow, cv.NORM_L1) <= (MAX_MEAN_DIF * dense_flow.shape[0] * dense_flow.shape[1]))

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()