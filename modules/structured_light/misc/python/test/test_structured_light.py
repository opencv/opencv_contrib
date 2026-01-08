#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function

import os, numpy

import cv2 as cv

from tests_common import NewOpenCVTests

class structured_light_test(NewOpenCVTests):

    def test_unwrap(self):
        paramsPsp = cv.structured_light_SinusoidalPattern_Params();
        paramsFtp = cv.structured_light_SinusoidalPattern_Params();
        paramsFaps = cv.structured_light_SinusoidalPattern_Params();
        paramsPsp.methodId = cv.structured_light.PSP;
        paramsFtp.methodId = cv.structured_light.FTP;
        paramsFaps.methodId = cv.structured_light.FAPS;

        sinusPsp = cv.structured_light.SinusoidalPattern_create(paramsPsp)
        sinusFtp = cv.structured_light.SinusoidalPattern_create(paramsFtp)
        sinusFaps = cv.structured_light.SinusoidalPattern_create(paramsFaps)

        captures = []
        for i in range(0,3):
            capture = self.get_sample('/cv/structured_light/data/capture_sin_%d.jpg'%i, cv.IMREAD_GRAYSCALE)
            if capture is None:
                raise unittest.SkipTest("Missing files with test data")
            captures.append(capture)

        rows,cols = captures[0].shape

        unwrappedPhaseMapPspRef = self.get_sample('/cv/structured_light/data/unwrappedPspTest.jpg',
                                                  cv.IMREAD_GRAYSCALE)
        unwrappedPhaseMapFtpRef = self.get_sample('/cv/structured_light/data/unwrappedFtpTest.jpg',
                                                  cv.IMREAD_GRAYSCALE)
        unwrappedPhaseMapFapsRef = self.get_sample('/cv/structured_light/data/unwrappedFapsTest.jpg',
                                                  cv.IMREAD_GRAYSCALE)

        wrappedPhaseMap,shadowMask = sinusPsp.computePhaseMap(captures);
        unwrappedPhaseMap = sinusPsp.unwrapPhaseMap(wrappedPhaseMap, (cols, rows), shadowMask=shadowMask)
        unwrappedPhaseMap8 = unwrappedPhaseMap*1 + 128
        unwrappedPhaseMap8 = numpy.uint8(unwrappedPhaseMap8)

        sumOfDiff = 0
        count = 0
        for i in range(rows):
            for j in range(cols):
                ref = int(unwrappedPhaseMapPspRef[i, j])
                comp = int(unwrappedPhaseMap8[i, j])
                sumOfDiff += (ref - comp)
                count += 1

        ratio = sumOfDiff/float(count)
        self.assertLessEqual(ratio, 0.2)

        wrappedPhaseMap,shadowMask = sinusFtp.computePhaseMap(captures);
        unwrappedPhaseMap = sinusFtp.unwrapPhaseMap(wrappedPhaseMap, (cols, rows), shadowMask=shadowMask)
        unwrappedPhaseMap8 = unwrappedPhaseMap*1 + 128
        unwrappedPhaseMap8 = numpy.uint8(unwrappedPhaseMap8)

        sumOfDiff = 0
        count = 0
        for i in range(rows):
            for j in range(cols):
                ref = int(unwrappedPhaseMapFtpRef[i, j])
                comp = int(unwrappedPhaseMap8[i, j])
                sumOfDiff += (ref - comp)
                count += 1

        ratio = sumOfDiff/float(count)
        self.assertLessEqual(ratio, 0.2)

        wrappedPhaseMap,shadowMask2 = sinusFaps.computePhaseMap(captures);
        unwrappedPhaseMap = sinusFaps.unwrapPhaseMap(wrappedPhaseMap, (cols, rows), shadowMask=shadowMask)
        unwrappedPhaseMap8 = unwrappedPhaseMap*1 + 128
        unwrappedPhaseMap8 = numpy.uint8(unwrappedPhaseMap8)

        sumOfDiff = 0
        count = 0
        for i in range(rows):
            for j in range(cols):
                ref = int(unwrappedPhaseMapFapsRef[i, j])
                comp = int(unwrappedPhaseMap8[i, j])
                sumOfDiff += (ref - comp)
                count += 1

        ratio = sumOfDiff/float(count)
        self.assertLessEqual(ratio, 0.2)

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
