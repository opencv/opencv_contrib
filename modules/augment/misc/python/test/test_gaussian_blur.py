import cv2 as cv
import numpy as np
from tests_common import NewOpenCVTests
import utils

## for consistency
np.random.seed(seed=1)
cv.setRNGSeed(seed=1)

transformations = [cv.augment_GaussianBlur(minKernelSize=3, maxKernelSize=15, minSigma=0.1, maxSigma=1),
                   cv.augment_GaussianBlur(minKernelSize=(5,7), maxKernelSize=(13,15), minSigmaX=0.1, maxSigmaX=1, minSigmaY=0.1, maxSigmaY=1),
                   cv.augment_GaussianBlur(kernelSize=7, sigma =0.2),
                   cv.augment_GaussianBlur(kernelSize=(5,7), sigmaX=0.2, sigmaY=0.3)]

class gaussianBlurTest(NewOpenCVTests):
    def test_image(self):
        utils.test_image(transformations)


    def test_point(self):
        utils.test_points(transformations)


    def test_rectangle(self):
        utils.test_rectangles(transformations)


    def test_polygon(self):
        utils.test_polygons(transformations)

