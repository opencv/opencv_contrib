import cv2 as cv
import numpy as np
from tests_common import NewOpenCVTests
from config import MIN_NUMBER_OF_TESTS, MAX_NUMBER_OF_TESTS, MIN_IMAGE_DIM_SIZE, MAX_IMAGE_DIM_SIZE, MIN_NUMBER_OF_GROUND_TRUTH_DATA, MAX_NUMBER_OF_GROUND_TRUTH_DATA

## for consistency
np.random.seed(seed=1)
cv.setRNGSeed(seed=1)


class augmenter_test(NewOpenCVTests):
    def test_augmenter_images(self):
        numberOfImages = np.random.randint(MIN_NUMBER_OF_TESTS, MAX_NUMBER_OF_TESTS)

        aug = cv.augment_Augmenter()
        aug.add(t=cv.augment_FlipHorizontal(), prob=0.7)
        aug.add(t=cv.augment_FlipVertical(), prob=0.5)
        aug.add(t=cv.augment_GaussianBlur(kernelSize=5, sigma=12), prob=0.7)
        aug.add(t=cv.augment_Rotate(minAngle=0, maxAngle=180), prob=0.3)
        aug.add(t=cv.augment_Resize(size=(1200, 900)), prob=0.4)

        imgs = []

        for i in range(numberOfImages):
            widthOfImages = np.random.randint(MIN_IMAGE_DIM_SIZE, MAX_IMAGE_DIM_SIZE)
            heightOfImages = np.random.randint(MIN_IMAGE_DIM_SIZE, MAX_IMAGE_DIM_SIZE)

            img = np.random.rand(heightOfImages, widthOfImages)
            imgs.append(img)

        imgs = imgs
        aug.applyImages(imgs)

    def test_augmenter_images_with_masks(self):
        numberOfImages = np.random.randint(MIN_NUMBER_OF_TESTS, MAX_NUMBER_OF_TESTS)

        aug = cv.augment_Augmenter()
        aug.add(t=cv.augment_FlipHorizontal(), prob=0.7)
        aug.add(t=cv.augment_FlipVertical(), prob=0.5)
        aug.add(t=cv.augment_GaussianBlur(kernelSize=5, sigma=12), prob=0.7)
        aug.add(t=cv.augment_Rotate(minAngle=0, maxAngle=180), prob=0.3)
        aug.add(t=cv.augment_Resize(size=(1200, 900)), prob=0.4)

        imgs = []
        masks = []

        for i in range(numberOfImages):
            widthOfImages = np.random.randint(MIN_IMAGE_DIM_SIZE, MAX_IMAGE_DIM_SIZE)
            heightOfImages = np.random.randint(MIN_IMAGE_DIM_SIZE, MAX_IMAGE_DIM_SIZE)

            img = np.random.rand(heightOfImages, widthOfImages)
            imgs.append(img)
            mask = np.random.rand(heightOfImages, widthOfImages)
            masks.append(mask)

        aug.applyImagesWithMasks(imgs, masks)

    def test_augmenter_images_with_points(self):
        numberOfImages = np.random.randint(MIN_NUMBER_OF_TESTS, MAX_NUMBER_OF_TESTS)

        aug = cv.augment_Augmenter()
        aug.add(t=cv.augment_FlipHorizontal(), prob=0.7)
        aug.add(t=cv.augment_FlipVertical(), prob=0.5)
        aug.add(t=cv.augment_GaussianBlur(kernelSize=5, sigma=12), prob=0.7)
        aug.add(t=cv.augment_Rotate(minAngle=0, maxAngle=180), prob=0.3)
        aug.add(t=cv.augment_Resize(size=(1200, 900)), prob=0.4)

        imgs = []
        pointsArr = []

        for i in range(numberOfImages):
            widthOfImages = np.random.randint(MIN_IMAGE_DIM_SIZE, MAX_IMAGE_DIM_SIZE)
            heightOfImages = np.random.randint(MIN_IMAGE_DIM_SIZE, MAX_IMAGE_DIM_SIZE)
            numberOfPoints = np.random.randint(MIN_NUMBER_OF_GROUND_TRUTH_DATA, MAX_NUMBER_OF_GROUND_TRUTH_DATA)

            img = np.random.rand(heightOfImages, widthOfImages)
            imgs.append(img)
            points = np.random.rand(numberOfPoints, 2)
            pointsArr.append(points)

        aug.applyImagesWithPoints(imgs, pointsArr)

    def test_augmenter_images_with_rectangles(self):
        numberOfImages = np.random.randint(MIN_NUMBER_OF_TESTS, MAX_NUMBER_OF_TESTS)

        aug = cv.augment_Augmenter()
        aug.add(t=cv.augment_FlipHorizontal(), prob=0.7)
        aug.add(t=cv.augment_FlipVertical(), prob=0.5)
        aug.add(t=cv.augment_GaussianBlur(kernelSize=5, sigma=12), prob=0.7)
        aug.add(t=cv.augment_Rotate(minAngle=0, maxAngle=180), prob=0.3)
        aug.add(t=cv.augment_Resize(size=(1200, 900)), prob=0.4)

        imgs = []
        rectsArr = []

        for i in range(numberOfImages):
            widthOfImages = np.random.randint(MIN_IMAGE_DIM_SIZE, MAX_IMAGE_DIM_SIZE)
            heightOfImages = np.random.randint(MIN_IMAGE_DIM_SIZE, MAX_IMAGE_DIM_SIZE)
            numberOfRects = np.random.randint(MIN_NUMBER_OF_GROUND_TRUTH_DATA, MAX_NUMBER_OF_GROUND_TRUTH_DATA)

            img = np.random.rand(heightOfImages, widthOfImages)
            imgs.append(img)
            rects = np.random.rand(numberOfRects, 4)
            rectsArr.append(rects)

        aug.applyImagesWithRectangles(imgs, rectsArr)


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()

