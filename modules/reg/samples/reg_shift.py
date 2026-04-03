#!/usr/bin/python

import cv2 as cv
import numpy as np
import sys

img1 = cv.imread(sys.argv[1])
img1 = img1.astype(np.float32)

shift = np.array([5.0, 5.0], dtype=np.float32)

# Prefer dot-notation (cv.reg.*), fallback to underscore bindings when needed
MapShift = getattr(cv.reg, "MapShift", getattr(cv, "reg_MapShift"))
MapperGradShift = getattr(cv.reg, "MapperGradShift", getattr(cv, "reg_MapperGradShift"))
MapperPyramid = getattr(cv.reg, "MapperPyramid", getattr(cv, "reg_MapperPyramid"))
MapTypeCaster_toShift = cv.reg.MapTypeCaster_toShift

mapTest = MapShift(shift)
img2 = mapTest.warp(img1)

# Avoid nested construction (reported to segfault in some builds)
mapper = MapperGradShift()
mappPyr = MapperPyramid(mapper)

resMap = mappPyr.calculate(img1, img2)
mapShift = MapTypeCaster_toShift(resMap)

print(mapShift.getShift())
