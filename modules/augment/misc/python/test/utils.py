import cv2 as cv
import numpy as np
from config import MIN_NUMBER_OF_TESTS, MAX_NUMBER_OF_TESTS, MIN_IMAGE_DIM_SIZE, MAX_IMAGE_DIM_SIZE, MIN_NUMBER_OF_GROUND_TRUTH_DATA, MAX_NUMBER_OF_GROUND_TRUTH_DATA

def test_image(transformations):
    numberOfImages = np.random.randint(MIN_NUMBER_OF_TESTS, MAX_NUMBER_OF_TESTS)

    for i in range(numberOfImages):
        widthOfImage = np.random.randint(MIN_IMAGE_DIM_SIZE, MAX_IMAGE_DIM_SIZE)
        heightOfImage = np.random.randint(MIN_IMAGE_DIM_SIZE, MAX_IMAGE_DIM_SIZE)

        img = np.random.rand(heightOfImage, widthOfImage)
        for t in transformations:
            t.init(img)
            t.image(img)


def test_points(transformations):
    numberOfImages = np.random.randint(MIN_NUMBER_OF_TESTS, MAX_NUMBER_OF_TESTS)

    for i in range(numberOfImages):
        widthOfImage = np.random.randint(MIN_IMAGE_DIM_SIZE, MAX_IMAGE_DIM_SIZE)
        heightOfImage = np.random.randint(MIN_IMAGE_DIM_SIZE, MAX_IMAGE_DIM_SIZE)
        numberOfPoints = np.random.randint(MIN_NUMBER_OF_GROUND_TRUTH_DATA, MAX_NUMBER_OF_GROUND_TRUTH_DATA)

        img = np.random.rand(heightOfImage, widthOfImage)
        points = np.random.rand(numberOfPoints, 2)

        for t in transformations:
            t.init(img)
            t.points(points)


def test_rectangles(transformations):
    numberOfImages = np.random.randint(10, 100)

    for i in range(numberOfImages):
        widthOfImage = np.random.randint(MIN_IMAGE_DIM_SIZE, MAX_IMAGE_DIM_SIZE)
        heightOfImage = np.random.randint(MIN_IMAGE_DIM_SIZE, MAX_IMAGE_DIM_SIZE)
        numberOfRects = np.random.randint(MIN_NUMBER_OF_GROUND_TRUTH_DATA, MAX_NUMBER_OF_GROUND_TRUTH_DATA)

        img = np.random.rand(heightOfImage, widthOfImage)
        rects = np.random.rand(numberOfRects, 4)

        for t in transformations:
            t.init(img)
            t.rectangles(rects)


def test_polygons(transformations):
    numberOfImages = np.random.randint(10, 100)

    for i in range(numberOfImages):
        widthOfImage = np.random.randint(MIN_IMAGE_DIM_SIZE, MAX_IMAGE_DIM_SIZE)
        heightOfImage = np.random.randint(MIN_IMAGE_DIM_SIZE, MAX_IMAGE_DIM_SIZE)
        numberOfPolys = np.random.randint(MIN_NUMBER_OF_GROUND_TRUTH_DATA, MAX_NUMBER_OF_GROUND_TRUTH_DATA)

        polys = []
        for i in range(numberOfPolys):
            numberOfPoints = np.random.randint(MIN_NUMBER_OF_GROUND_TRUTH_DATA, MAX_NUMBER_OF_GROUND_TRUTH_DATA)
            poly = np.random.rand(numberOfPoints, 2)
            polys.append(poly)

        img = np.random.rand(heightOfImage, widthOfImage)

        for t in transformations:
            t.init(img)
            t.polygons(polys)