import cv2 as cv
import numpy as np
from tests_common import NewOpenCVTests
import utils

## for consistency
np.random.seed(seed=1)
cv.setRNGSeed(seed=1)

transformations = [cv.augment_Crop(size=(30,40)),
                   cv.augment_Crop(size=(30, 40), origin=(15,20)),
                   cv.augment_Crop(minSize=(15, 20), maxSize=(30,40)),
                   cv.augment_Crop(minSize=(15, 20), maxSize=(30, 40), origin=(15,15))]

class cropTest(NewOpenCVTests):
    def test_image(self):
        utils.test_image(transformations)


    def test_point(self):
        utils.test_points(transformations)


    def test_rectangle(self):
        utils.test_rectangles(transformations)


    def test_polygon(self):
        utils.test_polygons(transformations)