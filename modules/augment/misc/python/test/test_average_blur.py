import cv2 as cv
import numpy as np
from tests_common import NewOpenCVTests
import utils

## for consistency
np.random.seed(seed=1)
cv.setRNGSeed(seed=1)

transformations = [cv.augment_AverageBlur(minKernelSize=3, maxKernelSize=15),
                   cv.augment_AverageBlur(minKernelSize=(5,7), maxKernelSize=(13,15)),
                   cv.augment_AverageBlur(kernelSize=7),
                   cv.augment_AverageBlur(kernelSize=(5,7))]

class averageBlurTest(NewOpenCVTests):
    def test_image(self):
        utils.test_image(transformations)


    def test_point(self):
        utils.test_points(transformations)


    def test_rectangle(self):
        utils.test_rectangles(transformations)


    def test_polygon(self):
        utils.test_polygons(transformations)