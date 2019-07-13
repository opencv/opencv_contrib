import cv2 as cv
import numpy as np
from tests_common import NewOpenCVTests

## for consistency
np.random.seed(seed=1)
cv.setRNGSeed(seed=1)


class augmenter_test(NewOpenCVTests):
    def test_augmenter_images(self):
        numberOfImages = np.random.randint(low=10, high=100)

        aug = cv.augment_Augmenter()
        aug.add(t=cv.augment_FlipHorizontal(), prob=0.7)
        aug.add(t=cv.augment_FlipVertical(), prob=0.5)
        aug.add(t=cv.augment_GaussianBlur(5, 12), prob=0.7)
        aug.add(t=cv.augment_Rotate(0, 180), prob=0.3)
        aug.add(t=cv.augment_Resize((1200, 900)), prob=0.4)

        imgs = []

        for i in range(numberOfImages):
            widthOfImages = np.random.randint(low=400, high=2000)
            heightOfImages = np.random.randint(low=400, high=2000)

            img = np.random.rand(heightOfImages, widthOfImages)
            imgs.append(img)

        imgs = imgs
        aug.applyImages(imgs)

    def test_augmenter_images_with_masks(self):
        numberOfImages = np.random.randint(low=10, high=100)

        aug = cv.augment_Augmenter()
        aug.add(t=cv.augment_FlipHorizontal(), prob=0.7)
        aug.add(t=cv.augment_FlipVertical(), prob=0.5)
        aug.add(t=cv.augment_GaussianBlur(5, 12), prob=0.7)
        aug.add(t=cv.augment_Rotate(0, 180), prob=0.3)
        aug.add(t=cv.augment_Resize((1200, 900)), prob=0.4)

        imgs = []
        masks = []

        for i in range(numberOfImages):
            widthOfImages = np.random.randint(low=400, high=2000)
            heightOfImages = np.random.randint(low=400, high=2000)

            img = np.random.rand(heightOfImages, widthOfImages)
            imgs.append(img)
            mask = np.random.rand(heightOfImages, widthOfImages)
            masks.append(mask)

        aug.applyImagesWithMasks(imgs, masks)

    def test_augmenter_images_with_points(self):
        numberOfImages = np.random.randint(low=10, high=100)

        aug = cv.augment_Augmenter()
        aug.add(t=cv.augment_FlipHorizontal(), prob=0.7)
        aug.add(t=cv.augment_FlipVertical(), prob=0.5)
        aug.add(t=cv.augment_GaussianBlur(5, 12), prob=0.7)
        aug.add(t=cv.augment_Rotate(0, 180), prob=0.3)
        aug.add(t=cv.augment_Resize((1200, 900)), prob=0.4)

        imgs = []
        pointsArr = []

        for i in range(numberOfImages):
            widthOfImages = np.random.randint(low=400, high=2000)
            heightOfImages = np.random.randint(low=400, high=2000)
            numberOfPoints = np.random.randint(low=1, high=100)

            img = np.random.rand(heightOfImages, widthOfImages)
            imgs.append(img)
            points = np.random.rand(numberOfPoints, 2)
            pointsArr.append(points)

        aug.applyImagesWithPoints(imgs, pointsArr)

    def test_augmenter_with_rectangles(self):
        numberOfImages = np.random.randint(10, 100)

        aug = cv.augment_Augmenter()
        aug.add(t=cv.augment_FlipHorizontal(), prob=0.7)
        aug.add(t=cv.augment_FlipVertical(), prob=0.5)
        aug.add(t=cv.augment_GaussianBlur(5, 12), prob=0.7)
        aug.add(t=cv.augment_Rotate(0, 180), prob=0.3)
        aug.add(t=cv.augment_Resize((1200, 900)), prob=0.4)

        imgs = []
        rectsArr = []

        for i in range(numberOfImages):
            widthOfImages = np.random.randint(low=400, high=2000)
            heightOfImages = np.random.randint(low=400, high=2000)
            numberOfRects = np.random.randint(low=1, high=100)

            img = np.random.rand(heightOfImages, widthOfImages)
            imgs.append(img)
            rects = np.random.rand(numberOfRects, 4)
            rectsArr.append(rects)

        aug.applyImagesWithRectangles(imgs, rectsArr)


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()

