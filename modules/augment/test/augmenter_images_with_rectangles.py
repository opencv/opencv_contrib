import cv2 as cv
import numpy as np


if __name__ == "__main__":
    ## for consistency
    np.random.seed(seed=1)
    cv.setRNGSeed(seed=1)

    numberOfImages = np.random.randint(10,100)

    aug = cv.augment_Augmenter()
    aug.add(cv.augment_FlipHorizontal(), 0.7)
    aug.add(cv.augment_FlipVertical(), 0.5)
    aug.add(cv.augment_GaussianBlur(5,12), .7)
    aug.add(cv.augment_Rotate(0,180), 0.3)
    aug.add(cv.augment_Resize((1200,900)),0.4)

    imgs = []
    rectsArr = []

    for i in range(numberOfImages):
        widthOfImages = np.random.randint(400, 2000)
        heightOfImages = np.random.randint(400, 2000)
        numberOfRects = np.random.randint(1,100)

        img = np.random.rand(heightOfImages, widthOfImages)
        imgs.append(img)
        rects = np.random.rand(numberOfRects, 4)
        rectsArr.append(rects)

    imgs2, rectsArr2 = aug.applyImagesWithRectangles(imgs, rectsArr)
