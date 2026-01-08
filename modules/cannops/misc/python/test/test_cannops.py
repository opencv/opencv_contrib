# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.

import cv2 as cv
from tests_common import NewOpenCVTests
import numpy as np

def genMask(mask, listx, listy):
    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            if (row in listx and col in listx) or (row in listy and col in listy):
                mask[row][col] = 1
    mask = mask.astype(np.uint8)
    return mask


mask = np.zeros((5, 5))
listx = [0, 1]
listy = [1, 2]
mask = genMask(mask, listx, listy)


class cannop_test(NewOpenCVTests):
    def test_ascend(self):
        cv.cann.initAcl()
        cv.cann.initDvpp()
        cv.cann.getDevice()
        cv.cann.setDevice(0)
        stream = cv.cann.AscendStream_Null()
        cv.cann.wrapStream(id(stream))
        cv.cann.resetDevice()

    def test_arithmetic(self):
        # input data
        npMat1 = np.random.random((5, 5, 3)).astype(int)
        npMat2 = np.random.random((5, 5, 3)).astype(int)
        cv.cann.setDevice(0)

        # ACLMat input data
        aclMat1 = cv.cann.AscendMat()
        aclMat1.upload(npMat1)
        aclMat2 = cv.cann.AscendMat()
        aclMat2.upload(npMat2)
        aclMask = cv.cann.AscendMat()
        aclMask.upload(mask)
        aclMatDst = cv.cann.AscendMat(aclMat1.size(), aclMat1.type())

        # InputArray interface test
        self.assertTrue(np.allclose(cv.cann.add(
            npMat1, npMat2), cv.add(npMat1, npMat2)))
        self.assertTrue(np.allclose(cv.cann.subtract(
            npMat1, npMat2), cv.subtract(npMat1, npMat2)))
        self.assertTrue(np.allclose(cv.cann.multiply(
            npMat1, npMat2, scale=2), cv.multiply(npMat1, npMat2, scale=2)))
        self.assertTrue(np.allclose(cv.cann.divide(
            npMat1, npMat2, scale=2), cv.divide(npMat1, npMat2, scale=2)))

        # AscendMat interface test
        self.assertTrue(np.allclose(cv.cann.add(aclMat1, aclMat2).download(),
                                    cv.add(npMat1, npMat2)))
        self.assertTrue(np.allclose(cv.cann.subtract(aclMat1, aclMat2).download(),
                                    cv.subtract(npMat1, npMat2)))
        self.assertTrue(np.allclose(cv.cann.multiply(aclMat1, aclMat2, scale=2).download(),
                                    cv.multiply(npMat1, npMat2, scale=2)))
        self.assertTrue(np.allclose(cv.cann.divide(aclMat1, aclMat2, scale=2).download(),
                                    cv.divide(npMat1, npMat2, scale=2)))

        # mask
        self.assertTrue(np.allclose(cv.cann.add(
            npMat1, npMat2, mask=mask), cv.add(npMat1, npMat2, mask=mask)))
        self.assertTrue(np.allclose(cv.cann.subtract(
            npMat1, npMat2, mask=mask), cv.subtract(npMat1, npMat2, mask=mask)))
        self.assertTrue(np.allclose(cv.cann.multiply(npMat1, npMat2, scale=2),
                                    cv.multiply(npMat1, npMat2, scale=2)))
        self.assertTrue(np.allclose(cv.cann.divide(npMat1, npMat2, scale=2),
                                    cv.divide(npMat1, npMat2, scale=2)))
        self.assertTrue(np.allclose(cv.cann.addWeighted(npMat1, 2, npMat2, 4, 3),
                                    cv.addWeighted(npMat1, 2, npMat2, 4, 3)))

        self.assertTrue(np.allclose(cv.cann.add(aclMat1, aclMat2, mask=aclMask).download(),
                                    cv.add(npMat1, npMat2, mask=mask)))
        self.assertTrue(np.allclose(cv.cann.subtract(aclMat1, aclMat2, mask=aclMask).download(),
                                    cv.subtract(npMat1, npMat2, mask=mask)))
        self.assertTrue(np.allclose(cv.cann.multiply(aclMat1, aclMat2, scale=2).download(),
                                    cv.multiply(npMat1, npMat2, scale=2)))
        self.assertTrue(np.allclose(cv.cann.divide(aclMat1, aclMat2, scale=2).download(),
                                    cv.divide(npMat1, npMat2, scale=2)))
        self.assertTrue(np.allclose(cv.cann.addWeighted(aclMat1, 2, aclMat2, 4, 3).download(),
                                    cv.addWeighted(npMat1, 2, npMat2, 4, 3)))

        # stream
        stream = cv.cann.AscendStream()
        matDst = cv.cann.add(npMat1, npMat2, stream=stream)
        stream.waitForCompletion()
        self.assertTrue(np.allclose(matDst, cv.add(npMat1, npMat2)))
        matDst = cv.cann.add(npMat1, npMat2, mask=mask, stream=stream)
        stream.waitForCompletion()
        self.assertTrue(np.allclose(matDst, cv.add(npMat1, npMat2, mask=mask)))
        matDst = cv.cann.subtract(npMat1, npMat2, mask=mask, stream=stream)
        stream.waitForCompletion()
        self.assertTrue(np.allclose(
            matDst, cv.subtract(npMat1, npMat2, mask=mask)))

        # stream AsceendMat
        aclMatDst = cv.cann.add(aclMat1, aclMat2, stream=stream)
        stream.waitForCompletion()
        self.assertTrue(np.allclose(aclMatDst.download(),
                        cv.add(npMat1, npMat2)))

        aclMatDst = cv.cann.add(aclMat1, aclMat2, mask=aclMask, stream=stream)
        stream.waitForCompletion()
        self.assertTrue(np.allclose(aclMatDst.download(),
                        cv.add(npMat1, npMat2, mask=mask)))

        aclMatDst = cv.cann.subtract(aclMat1, aclMat2, mask=aclMask, stream=stream)
        stream.waitForCompletion()
        self.assertTrue(np.allclose(aclMatDst.download(),
                        cv.subtract(npMat1, npMat2, mask=mask)))

        cv.cann.resetDevice()

    def test_logical(self):
        npMat1 = np.random.random((5, 5, 3)).astype(np.uint16)
        npMat2 = np.random.random((5, 5, 3)).astype(np.uint16)
        cv.cann.setDevice(0)

        # ACLMat input data
        aclMat1 = cv.cann.AscendMat()
        aclMat1.upload(npMat1)
        aclMat2 = cv.cann.AscendMat()
        aclMat2.upload(npMat2)
        aclMask = cv.cann.AscendMat()
        aclMask.upload(mask)

        self.assertTrue(np.allclose(cv.cann.bitwise_or(npMat1, npMat2),
                                    cv.bitwise_or(npMat1, npMat2)))
        self.assertTrue(np.allclose(cv.cann.bitwise_or(
            npMat1, npMat2), cv.bitwise_or(npMat1, npMat2)))
        self.assertTrue(np.allclose(cv.cann.bitwise_and(npMat1, npMat2),
                                    cv.bitwise_and(npMat1, npMat2)))
        self.assertTrue(np.allclose(cv.cann.bitwise_and(
            npMat1, npMat2), cv.bitwise_and(npMat1, npMat2)))
        self.assertTrue(np.allclose(cv.cann.bitwise_xor(npMat1, npMat2),
                                    cv.bitwise_xor(npMat1, npMat2)))
        self.assertTrue(np.allclose(cv.cann.bitwise_xor(
            npMat1, npMat2), cv.bitwise_xor(npMat1, npMat2)))
        self.assertTrue(np.allclose(cv.cann.bitwise_not(npMat1),
                                    cv.bitwise_not(npMat1)))
        self.assertTrue(np.allclose(
            cv.cann.bitwise_not(npMat1), cv.bitwise_not(npMat1)))
        self.assertTrue(np.allclose(cv.cann.bitwise_and(npMat1, npMat2, mask=mask),
                                    cv.bitwise_and(npMat1, npMat2, mask=mask)))
        self.assertTrue(np.allclose(cv.cann.bitwise_or(npMat1, npMat2, mask=mask),
                                    cv.bitwise_or(npMat1, npMat2, mask=mask)))
        self.assertTrue(np.allclose(cv.cann.bitwise_not(npMat1, mask=mask),
                                    cv.bitwise_not(npMat1, mask=mask)))
        self.assertTrue(np.allclose(cv.cann.bitwise_xor(npMat1, npMat2, mask=mask),
                                    cv.bitwise_xor(npMat1, npMat2, mask=mask)))

        # AscendMat interface
        self.assertTrue(np.allclose(cv.cann.bitwise_or(aclMat1, aclMat2).download(),
                                    cv.bitwise_or(npMat1, npMat2)))
        self.assertTrue(np.allclose(cv.cann.bitwise_or(aclMat1, aclMat2).download(),
                                    cv.bitwise_or(npMat1, npMat2)))
        self.assertTrue(np.allclose(cv.cann.bitwise_and(aclMat1, aclMat2).download(),
                                    cv.bitwise_and(npMat1, npMat2)))
        self.assertTrue(np.allclose(cv.cann.bitwise_and(
            aclMat1, aclMat2).download(), cv.bitwise_and(npMat1, npMat2)))
        self.assertTrue(np.allclose(cv.cann.bitwise_xor(aclMat1, aclMat2).download(),
                                    cv.bitwise_xor(npMat1, npMat2)))
        self.assertTrue(np.allclose(cv.cann.bitwise_xor(
            aclMat1, aclMat2).download(), cv.bitwise_xor(npMat1, npMat2)))
        self.assertTrue(np.allclose(cv.cann.bitwise_not(aclMat1).download(),
                                    cv.bitwise_not(npMat1)))
        self.assertTrue(np.allclose(cv.cann.bitwise_not(aclMat1).download(),
                                    cv.bitwise_not(npMat1)))
        self.assertTrue(np.allclose(cv.cann.bitwise_and(aclMat1, aclMat2, mask=aclMask).download(),
                                    cv.bitwise_and(npMat1, npMat2, mask=mask)))
        self.assertTrue(np.allclose(cv.cann.bitwise_or(aclMat1, aclMat2, mask=aclMask).download(),
                                    cv.bitwise_or(npMat1, npMat2, mask=mask)))
        self.assertTrue(np.allclose(cv.cann.bitwise_not(aclMat1, mask=aclMask).download(),
                                    cv.bitwise_not(npMat1, mask=mask)))
        self.assertTrue(np.allclose(cv.cann.bitwise_xor(aclMat1, aclMat2, mask=aclMask).download(),
                                    cv.bitwise_xor(npMat1, npMat2, mask=mask)))
        cv.cann.resetDevice()

    def test_imgproc(self):
        npMat = (np.random.random((128, 128, 3)) * 255).astype(np.uint8)
        cv.cann.setDevice(0)
        aclMat = cv.cann.AscendMat()
        aclMatDst = aclMat
        aclMat.upload(npMat)

        # TODO try pass out param, not use return value.
        # merge & split
        self.assertTrue(np.allclose(
            cv.cann.merge(cv.cann.split(npMat)).download(), npMat))
        self.assertTrue(np.allclose(
            cv.cann.merge(cv.cann.split(aclMat)).download(), npMat))

        # transpose
        self.assertTrue(np.allclose(
            cv.cann.transpose(npMat), cv.transpose(npMat)))
        self.assertTrue(np.allclose(
            cv.cann.transpose(aclMat).download(), cv.transpose(npMat)))

        # crop
        w_off, h_off, crop_w, crop_h = 0, 0, 64, 64
        roi = [w_off, h_off, crop_w, crop_h]
        self.assertTrue(np.allclose(
            cv.cann.crop(npMat, roi).download(), npMat[w_off:crop_w, h_off:crop_h]))
        self.assertTrue(np.allclose(
            cv.cann.crop(aclMat, roi).download(), npMat[w_off:crop_w, h_off:crop_h]))

        # resize
        dstSize = np.array([crop_w, crop_h])
        aclMat32F = cv.cann.AscendMat()
        aclMat32F.upload(npMat.astype(np.float32))
        self.assertTrue(np.allclose(cv.cann.resize(npMat.astype(np.float32), dstSize, 0, 0, 3),
                        cv.resize(npMat.astype(np.float32), dstSize, 0, 0, 3)))
        self.assertTrue(np.allclose(cv.cann.resize(aclMat32F, dstSize, 0, 0, 3).download(),
                        cv.resize(npMat.astype(np.float32), dstSize, 0, 0, 3)))
        # flip
        flipMode = [0, 1, -1]
        for fMode in flipMode:
            self.assertTrue(np.allclose(cv.cann.flip(
                npMat, fMode), cv.flip(npMat, fMode)))
            self.assertTrue(np.allclose(cv.cann.flip(
                aclMat, fMode).download(), cv.flip(npMat, fMode)))

        # rotate
        rotateMode = [0, 1, 2]
        for rMode in rotateMode:
            self.assertTrue(np.allclose(cv.cann.rotate(
                npMat, rMode), cv.rotate(npMat, rMode)))
            self.assertTrue(np.allclose(cv.cann.rotate(
                aclMat, rMode).download(), cv.rotate(npMat, rMode)))

        # cvtColcor
        cvtModeC1 = [cv.COLOR_GRAY2BGR, cv.COLOR_GRAY2BGRA]
        cvtModeC3 = [cv.COLOR_BGR2GRAY, cv.COLOR_BGRA2BGR, cv.COLOR_BGR2RGBA, cv.COLOR_RGBA2BGR,
                     cv.COLOR_BGR2RGB, cv.COLOR_BGRA2RGBA, cv.COLOR_RGB2GRAY, cv.COLOR_BGRA2GRAY,
                     cv.COLOR_RGBA2GRAY, cv.COLOR_BGR2BGRA, cv.COLOR_BGR2YUV, cv.COLOR_RGB2YUV,
                     cv.COLOR_YUV2BGR, cv.COLOR_YUV2RGB, cv.COLOR_BGR2YCrCb, cv.COLOR_RGB2YCrCb,
                     cv.COLOR_YCrCb2BGR, cv.COLOR_YCrCb2RGB, cv.COLOR_BGR2XYZ, cv.COLOR_RGB2XYZ,
                     cv.COLOR_XYZ2BGR, cv.COLOR_XYZ2RGB,]
        for cvtM in cvtModeC3:
            self.assertTrue(np.allclose(cv.cann.cvtColor(
                npMat, cvtM), cv.cvtColor(npMat, cvtM), 1))
            self.assertTrue(np.allclose(cv.cann.cvtColor(
                aclMat, cvtM).download(), cv.cvtColor(npMat, cvtM), 1))

        npMatC1 = (np.random.random((128, 128, 1)) * 255).astype(np.uint8)
        aclMatC1 = cv.cann.AscendMat()
        aclMatC1.upload(npMatC1)
        for cvtM in cvtModeC1:
            self.assertTrue(np.allclose(cv.cann.cvtColor(
                npMatC1, cvtM), cv.cvtColor(npMatC1, cvtM), 1))
            self.assertTrue(np.allclose(cv.cann.cvtColor(
                aclMatC1, cvtM).download(), cv.cvtColor(npMatC1, cvtM), 1))

        # threshold
        threshType = [cv.THRESH_BINARY, cv.THRESH_BINARY_INV,
                      cv.THRESH_TRUNC, cv.THRESH_TOZERO, cv.THRESH_TOZERO_INV]
        for tType in threshType:
            cvRet, cvThresh = cv.threshold(
                npMat.astype(np.uint8), 127, 255, tType)
            cannRet, cannThresh = cv.cann.threshold(
                npMat.astype(np.float32), 127, 255, tType)
            self.assertTrue(np.allclose(cvThresh, cannThresh))
            self.assertTrue(np.allclose(cvRet, cannRet))

            aclMat.upload(npMat.astype(np.float32))
            cannRet, cannThresh = cv.cann.threshold(
                aclMat, 127, 255, tType)
            self.assertTrue(np.allclose(cvThresh, cannThresh.download()))
            self.assertTrue(np.allclose(cvRet, cannRet))

        npMat = (np.random.random((1280, 1024, 3)) * 255).astype(np.uint8)
        w_off, h_off, crop_w, crop_h = 0, 0, 512, 384
        roi = [w_off, h_off, crop_w, crop_h]
        aclMat = cv.cann.AscendMat()
        aclMat.upload(npMat)

        # resize
        dstSize = np.array([crop_w, crop_h])
        self.assertTrue(np.allclose(cv.cann.resize(npMat, dstSize, 0, 0, 1),
                        cv.resize(npMat, dstSize, 0, 0, 1)))
        self.assertTrue(np.allclose(cv.cann.resize(aclMat, dstSize, 0, 0, 1).download(),
                        cv.resize(npMat, dstSize, 0, 0, 1)))
        # cropResize
        self.assertTrue(np.allclose(cv.cann.cropResize(npMat, roi, dstSize, 0, 0, 1),
                        cv.resize(npMat[h_off:crop_h, w_off:crop_w], dstSize, 0, 0, 1)), 0)
        self.assertTrue(np.allclose(cv.cann.cropResize(aclMat, roi, dstSize, 0, 0, 1).download(),
                        cv.resize(npMat[h_off:crop_h, w_off:crop_w], dstSize, 0, 0, 1)), 0)

        # cropResizeMakeBorder
        # TODO cv.copyMakeBorder ignores borderColorValue param; find the reason and fix it
        borderColorValue = (100, 0, 255)
        top, bottom, left, right = 32, 0, 10, 0
        borderTypes = [0, 1]

        for borderType in borderTypes:
            self.assertTrue(np.allclose(cv.cann.cropResizeMakeBorder(npMat, roi, dstSize,
                                0, 0, 1, top, left, borderType),
                            cv.copyMakeBorder(cv.resize(npMat[h_off:crop_h, w_off:crop_w],
                                dstSize, 0, 0, 1), top, bottom, left, right, borderType), 1))
            self.assertTrue(np.allclose(cv.cann.cropResizeMakeBorder(aclMat, roi, dstSize,
                                0, 0, 1, top, left, borderType).download(),
                            cv.copyMakeBorder(cv.resize(npMat[h_off:crop_h, w_off:crop_w],
                                dstSize, 0, 0, 1), top, bottom, left, right, borderType), 1))

        # copyMakeBorder
        for borderType in borderTypes:
            self.assertTrue(np.allclose(cv.cann.copyMakeBorder(npMat, top, bottom, left, right,
                                                               borderType),
                            cv.copyMakeBorder(npMat, top, bottom, left, right, borderType)))
            self.assertTrue(np.allclose(cv.cann.copyMakeBorder(aclMat, top, bottom, left, right,
                                                               borderType).download(),
                            cv.copyMakeBorder(npMat, top, bottom, left, right, borderType)))

        cv.cann.resetDevice()

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
