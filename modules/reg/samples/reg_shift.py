#!/usr/bin/python

import cv2
import numpy as np
import sys

img1 = cv2.imread(sys.argv[1])
img1 = img1.astype(np.float32)
shift = np.array([5., 5.])
mapTest = cv2.reg.MapShift(shift)

img2 = mapTest.warp(img1)

mapper = cv2.reg.MapperGradShift()
mappPyr = cv2.reg.MapperPyramid(mapper)

resMap = mappPyr.calculate(img1, img2)
mapShift = cv2.reg.MapTypeCaster_toShift(resMap)

print(mapShift.getShift())
