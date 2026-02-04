#!/usr/bin/env python

import sys
import cv2 as cv
import numpy as np


def _pick_api():
    """Prefer cv.reg.*; fallback to underscore-style bindings when needed."""
    reg = getattr(cv, "reg", None)

    if reg is not None and all(
        hasattr(reg, n)
        for n in ("MapShift", "MapperGradShift", "MapperPyramid", "MapTypeCaster_toShift")
    ):
        return reg.MapShift, reg.MapperGradShift, reg.MapperPyramid, reg.MapTypeCaster_toShift

    if (
        reg is not None
        and hasattr(reg, "MapTypeCaster_toShift")
        and all(hasattr(cv, n) for n in ("reg_MapShift", "reg_MapperGradShift", "reg_MapperPyramid"))
    ):
        return cv.reg_MapShift, cv.reg_MapperGradShift, cv.reg_MapperPyramid, reg.MapTypeCaster_toShift

    return None


# keep original behavior: expects an image path argument
img1 = cv.imread(sys.argv[1], cv.IMREAD_COLOR)
if img1 is None:
    raise FileNotFoundError(f"Could not read image: {sys.argv[1]}")

api = _pick_api()
if api is None:
    raise RuntimeError("Required OpenCV reg bindings are not available (install opencv-contrib).")

MapShift, MapperGradShift, MapperPyramid, MapTypeCaster_toShift = api

img1 = img1.astype(np.float32)
shift = np.array([5.0, 5.0], dtype=np.float32)

map_test = MapShift(shift)
img2 = map_test.warp(img1)

# Avoid nested construction (reported to segfault in some builds)
mapper = MapperGradShift()
mapp_pyr = MapperPyramid(mapper)

res_map = mapp_pyr.calculate(img1, img2)
map_shift = MapTypeCaster_toShift(res_map)

print(map_shift.getShift())
