#!/usr/bin/env python
import os
import cv2 as cv
import numpy as np

from tests_common import NewOpenCVTests, unittest

class cudaarithm_test(NewOpenCVTests):
    def setUp(self):
        super(cudaarithm_test, self).setUp()
        if not cv.cuda.getCudaEnabledDeviceCount():
            self.skipTest("No CUDA-capable device is detected")

    def test_cudaarithm(self):
        npMat = (np.random.random((128, 128, 3)) * 255).astype(np.uint8)

        cuMat = cv.cuda_GpuMat(npMat)
        cuMatDst = cv.cuda_GpuMat(cuMat.size(),cuMat.type())
        cuMatB = cv.cuda_GpuMat(cuMat.size(),cv.CV_8UC1)
        cuMatG = cv.cuda_GpuMat(cuMat.size(),cv.CV_8UC1)
        cuMatR = cv.cuda_GpuMat(cuMat.size(),cv.CV_8UC1)

        self.assertTrue(np.allclose(cv.cuda.merge(cv.cuda.split(cuMat)),npMat))

        cv.cuda.split(cuMat,[cuMatB,cuMatG,cuMatR])
        cv.cuda.merge([cuMatB,cuMatG,cuMatR],cuMatDst)
        self.assertTrue(np.allclose(cuMatDst.download(),npMat))

        shift = (np.random.random((cuMat.channels(),)) * 8).astype(np.uint8).tolist()
        self.assertTrue(np.allclose(cv.cuda.rshift(cuMat,shift).download(),npMat  >> shift))
        cv.cuda.rshift(cuMat,shift,cuMatDst)
        self.assertTrue(np.allclose(cuMatDst.download(),npMat >> shift))

        self.assertTrue(np.allclose(cv.cuda.lshift(cuMat,shift).download(),(npMat << shift).astype('uint8')))
        cv.cuda.lshift(cuMat,shift,cuMatDst)
        self.assertTrue(np.allclose(cuMatDst.download(),(npMat << shift).astype('uint8')))

    def test_arithmetic(self):
        npMat1 = np.random.random((128, 128, 3)) - 0.5
        npMat2 = np.random.random((128, 128, 3)) - 0.5

        cuMat1 = cv.cuda_GpuMat()
        cuMat2 = cv.cuda_GpuMat()
        cuMat1.upload(npMat1)
        cuMat2.upload(npMat2)
        cuMatDst = cv.cuda_GpuMat(cuMat1.size(),cuMat1.type())

        self.assertTrue(np.allclose(cv.cuda.add(cuMat1, cuMat2).download(),
                                         cv.add(npMat1, npMat2)))

        cv.cuda.add(cuMat1, cuMat2, cuMatDst)
        self.assertTrue(np.allclose(cuMatDst.download(),cv.add(npMat1, npMat2)))

        self.assertTrue(np.allclose(cv.cuda.subtract(cuMat1, cuMat2).download(),
                                         cv.subtract(npMat1, npMat2)))

        cv.cuda.subtract(cuMat1, cuMat2, cuMatDst)
        self.assertTrue(np.allclose(cuMatDst.download(),cv.subtract(npMat1, npMat2)))

        self.assertTrue(np.allclose(cv.cuda.multiply(cuMat1, cuMat2).download(),
                                         cv.multiply(npMat1, npMat2)))

        cv.cuda.multiply(cuMat1, cuMat2, cuMatDst)
        self.assertTrue(np.allclose(cuMatDst.download(),cv.multiply(npMat1, npMat2)))

        self.assertTrue(np.allclose(cv.cuda.divide(cuMat1, cuMat2).download(),
                                         cv.divide(npMat1, npMat2)))

        cv.cuda.divide(cuMat1, cuMat2, cuMatDst)
        self.assertTrue(np.allclose(cuMatDst.download(),cv.divide(npMat1, npMat2)))

        self.assertTrue(np.allclose(cv.cuda.absdiff(cuMat1, cuMat2).download(),
                                         cv.absdiff(npMat1, npMat2)))

        cv.cuda.absdiff(cuMat1, cuMat2, cuMatDst)
        self.assertTrue(np.allclose(cuMatDst.download(),cv.absdiff(npMat1, npMat2)))

        self.assertTrue(np.allclose(cv.cuda.compare(cuMat1, cuMat2, cv.CMP_GE).download(),
                                         cv.compare(npMat1, npMat2, cv.CMP_GE)))

        cuMatDst1 = cv.cuda_GpuMat(cuMat1.size(),cv.CV_8UC3)
        cv.cuda.compare(cuMat1, cuMat2, cv.CMP_GE, cuMatDst1)
        self.assertTrue(np.allclose(cuMatDst1.download(),cv.compare(npMat1, npMat2, cv.CMP_GE)))

        self.assertTrue(np.allclose(cv.cuda.abs(cuMat1).download(),
                                         np.abs(npMat1)))

        cv.cuda.abs(cuMat1, cuMatDst)
        self.assertTrue(np.allclose(cuMatDst.download(),np.abs(npMat1)))

        self.assertTrue(np.allclose(cv.cuda.sqrt(cv.cuda.sqr(cuMat1)).download(),
                                    cv.cuda.abs(cuMat1).download()))

        cv.cuda.sqr(cuMat1, cuMatDst)
        cv.cuda.sqrt(cuMatDst, cuMatDst)
        self.assertTrue(np.allclose(cuMatDst.download(),cv.cuda.abs(cuMat1).download()))

        self.assertTrue(np.allclose(cv.cuda.log(cv.cuda.exp(cuMat1)).download(),
                                                            npMat1))

        cv.cuda.exp(cuMat1, cuMatDst)
        cv.cuda.log(cuMatDst, cuMatDst)
        self.assertTrue(np.allclose(cuMatDst.download(),npMat1))

        self.assertTrue(np.allclose(cv.cuda.pow(cuMat1, 2).download(),
                                         cv.pow(npMat1, 2)))

        cv.cuda.pow(cuMat1, 2, cuMatDst)
        self.assertTrue(np.allclose(cuMatDst.download(),cv.pow(npMat1, 2)))

    def test_logical(self):
        npMat1 = (np.random.random((128, 128)) * 255).astype(np.uint8)
        npMat2 = (np.random.random((128, 128)) * 255).astype(np.uint8)

        cuMat1 = cv.cuda_GpuMat()
        cuMat2 = cv.cuda_GpuMat()
        cuMat1.upload(npMat1)
        cuMat2.upload(npMat2)
        cuMatDst = cv.cuda_GpuMat(cuMat1.size(),cuMat1.type())

        self.assertTrue(np.allclose(cv.cuda.bitwise_or(cuMat1, cuMat2).download(),
                                         cv.bitwise_or(npMat1, npMat2)))

        cv.cuda.bitwise_or(cuMat1, cuMat2, cuMatDst)
        self.assertTrue(np.allclose(cuMatDst.download(),cv.bitwise_or(npMat1, npMat2)))

        self.assertTrue(np.allclose(cv.cuda.bitwise_and(cuMat1, cuMat2).download(),
                                         cv.bitwise_and(npMat1, npMat2)))

        cv.cuda.bitwise_and(cuMat1, cuMat2, cuMatDst)
        self.assertTrue(np.allclose(cuMatDst.download(),cv.bitwise_and(npMat1, npMat2)))

        self.assertTrue(np.allclose(cv.cuda.bitwise_xor(cuMat1, cuMat2).download(),
                                         cv.bitwise_xor(npMat1, npMat2)))

        cv.cuda.bitwise_xor(cuMat1, cuMat2, cuMatDst)
        self.assertTrue(np.allclose(cuMatDst.download(),cv.bitwise_xor(npMat1, npMat2)))

        self.assertTrue(np.allclose(cv.cuda.bitwise_not(cuMat1).download(),
                                         cv.bitwise_not(npMat1)))

        cv.cuda.bitwise_not(cuMat1, cuMatDst)
        self.assertTrue(np.allclose(cuMatDst.download(),cv.bitwise_not(npMat1)))

        self.assertTrue(np.allclose(cv.cuda.min(cuMat1, cuMat2).download(),
                                         cv.min(npMat1, npMat2)))

        cv.cuda.min(cuMat1, cuMat2, cuMatDst)
        self.assertTrue(np.allclose(cuMatDst.download(),cv.min(npMat1, npMat2)))

        self.assertTrue(np.allclose(cv.cuda.max(cuMat1, cuMat2).download(),
                                         cv.max(npMat1, npMat2)))

        cv.cuda.max(cuMat1, cuMat2, cuMatDst)
        self.assertTrue(np.allclose(cuMatDst.download(),cv.max(npMat1, npMat2)))

        self.assertTrue(cv.cuda.minMax(cuMat1),cv.minMaxLoc(npMat1)[:2])
        self.assertTrue(cv.cuda.minMaxLoc(cuMat1),cv.minMaxLoc(npMat1))

    def test_convolution(self):
        npMat = (np.random.random((128, 128)) * 255).astype(np.float32)
        npDims = np.array(npMat.shape)
        kernel = (np.random.random((3, 3)) * 1).astype(np.float32)
        kernelDims = np.array(kernel.shape)
        iS = (kernelDims/2).astype(int)
        iE = npDims - kernelDims + iS

        cuMat = cv.cuda_GpuMat(npMat)
        cuKernel= cv.cuda_GpuMat(kernel)
        cuMatDst = cv.cuda_GpuMat(tuple(npDims - kernelDims + 1), cuMat.type())
        conv = cv.cuda.createConvolution()

        self.assertTrue(np.allclose(conv.convolve(cuMat,cuKernel,ccorr=True).download(),
                    cv.filter2D(npMat,-1,kernel,anchor=(-1,-1))[iS[0]:iE[0]+1,iS[1]:iE[1]+1]))

        conv.convolve(cuMat,cuKernel,cuMatDst,True)
        self.assertTrue(np.allclose(cuMatDst.download(),
                    cv.filter2D(npMat,-1,kernel,anchor=(-1,-1))[iS[0]:iE[0]+1,iS[1]:iE[1]+1]))

    def test_inrange(self):
        npMat = (np.random.random((128, 128, 3)) * 255).astype(np.float32)

        bound1 = np.random.random((4,)) * 255
        bound2 = np.random.random((4,)) * 255
        lowerb = np.minimum(bound1, bound2).tolist()
        upperb = np.maximum(bound1, bound2).tolist()

        cuMat = cv.cuda_GpuMat()
        cuMat.upload(npMat)

        self.assertTrue((cv.cuda.inRange(cuMat, lowerb, upperb).download() ==
                         cv.inRange(npMat, np.array(lowerb), np.array(upperb))).all())

        cuMatDst = cv.cuda_GpuMat(cuMat.size(), cv.CV_8UC1)
        cv.cuda.inRange(cuMat, lowerb, upperb, cuMatDst)
        self.assertTrue((cuMatDst.download() ==
                         cv.inRange(npMat, np.array(lowerb), np.array(upperb))).all())

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
