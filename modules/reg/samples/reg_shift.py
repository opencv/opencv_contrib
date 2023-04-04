#!/usr/bin/python

import cv2 as cv
import numpy as np
import sys

img1 = cv.imread(sys.argv[1])
img1 = img1.astype(np.float32)
shift = np.array([5., 5.])

#substituted dot (.) with an underscore (_)

mapTest = cv.reg_MapShift(shift)        #line 1

img2 = mapTest.warp(img1)

mapper = cv.reg_MapperGradShift()       #line2
mappPyr = cv.reg_MapperPyramid(mapper)          #line 3

resMap = mappPyr.calculate(img1, img2)
mapShift = cv.reg.MapTypeCaster_toShift(resMap)

print(mapShift.getShift())
