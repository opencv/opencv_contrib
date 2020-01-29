#!/usr/bin/env python
import os
import cv2 as cv
import numpy as np

from tests_common import NewOpenCVTests, unittest

def create_affine_transform_matrix(size,angle):
    return np.array([[np.cos(angle), -np.sin(angle), size[1]/2], [np.sin(angle), np.cos(angle), 0]])
def create_perspective_transform_matrix(size,angle):
    return np.vstack([create_affine_transform_matrix(size,angle),[0, 0, 1]])

class cudawarping_test(NewOpenCVTests):
    def setUp(self):
        super(cudawarping_test, self).setUp()
        if not cv.cuda.getCudaEnabledDeviceCount():
            self.skipTest("No CUDA-capable device is detected")

    def test_resize(self):
        dstSz = (256,256)
        interp = cv.INTER_NEAREST
        npMat = (np.random.random((128,128,3))*255).astype(np.uint8)

        cuMat = cv.cuda_GpuMat(npMat)
        cuMatDst = cv.cuda_GpuMat(dstSz,cuMat.type())

        self.assertTrue(np.allclose(cv.cuda.resize(cuMat,dstSz,interpolation=interp).download(),
            cv.resize(npMat,dstSz,interpolation=interp)))

        cv.cuda.resize(cuMat,dstSz,cuMatDst,interpolation=interp)
        self.assertTrue(np.allclose(cuMatDst.download(),cv.resize(npMat,dstSz,interpolation=interp)))

    def test_warp(self):
        npMat = (np.random.random((128,128,3))*255).astype(np.uint8)
        size = npMat.shape[:2]
        M1 = create_affine_transform_matrix(size,np.pi/2)

        cuMat = cv.cuda_GpuMat(npMat)
        cuMatDst = cv.cuda_GpuMat(size,cuMat.type())

        borderType = cv.BORDER_REFLECT101
        self.assertTrue(np.allclose(cv.cuda.warpAffine(cuMat,M1,size,borderMode=borderType).download(),
            cv.warpAffine(npMat,M1,size, borderMode=borderType)))
        cv.cuda.warpAffine(cuMat,M1,size,cuMatDst,borderMode=borderType)
        self.assertTrue(np.allclose(cuMatDst.download(),cv.warpAffine(npMat,M1,size,borderMode=borderType)))

        interpolation = cv.INTER_NEAREST
        flags = interpolation | cv.WARP_INVERSE_MAP
        dst_gold = cv.warpAffine(npMat, M1, size, flags = flags)
        cuMaps = cv.cuda.buildWarpAffineMaps(M1,True,size)
        dst = cv.remap(npMat, cuMaps[0].download(), cuMaps[1].download(),interpolation)
        self.assertTrue(np.allclose(dst,dst_gold))

        xmap = cv.cuda_GpuMat(size,cv.CV_32FC1)
        ymap = cv.cuda_GpuMat(size,cv.CV_32FC1)
        cv.cuda.buildWarpAffineMaps(M1,True,size,xmap,ymap)
        dst = cv.remap(npMat, xmap.download(), ymap.download(),interpolation)
        self.assertTrue(np.allclose(dst,dst_gold))

        M2 = create_perspective_transform_matrix(size,np.pi/2)
        np.allclose(cv.cuda.warpPerspective(cuMat,M2,size,borderMode=borderType).download(),
                    cv.warpPerspective(npMat,M2,size,borderMode=borderType))
        cv.cuda.warpPerspective(cuMat,M2,size,cuMatDst,borderMode=borderType)
        self.assertTrue(np.allclose(cuMatDst.download(),cv.warpPerspective(npMat,M2,size,borderMode=borderType)))

        dst_gold = cv.warpPerspective(npMat, M2, size, flags = flags)
        cuMaps = cv.cuda.buildWarpPerspectiveMaps(M2,True,size)
        dst = cv.remap(npMat, cuMaps[0].download(), cuMaps[1].download(),interpolation)
        self.assertTrue(np.allclose(dst,dst_gold))

        cv.cuda.buildWarpPerspectiveMaps(M2,True,size,xmap,ymap)
        dst = cv.remap(npMat, xmap.download(), ymap.download(),interpolation)
        self.assertTrue(np.allclose(dst,dst_gold))

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()