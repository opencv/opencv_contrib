import cv2 as cv
import numpy as np


if __name__ == "__main__":
    ## for consistency
    np.random.seed(seed=1)
    cv.setRNGSeed(seed=1)

    numberOfImages = np.random.randint(10,100)

    aug = cv.augment_Augmenter()
    aug.add(t=cv.augment_FlipHorizontal(), prob=0.7)
    aug.add(t=cv.augment_FlipVertical(), prob=0.5)
    aug.add(t=cv.augment_GaussianBlur(5,12), prob=0.7)
    aug.add(t=cv.augment_Rotate(0,180), prob=0.3)
    aug.add(t=cv.augment_Resize((1200,900)), prob=0.4)

    imgs = []
    masks = []

    for i in range(numberOfImages):
        widthOfImages = np.random.randint(400, 2000)
        heightOfImages = np.random.randint(400, 2000)

        img = np.random.rand(heightOfImages, widthOfImages)
        imgs.append(img)
        mask = np.random.rand(heightOfImages, widthOfImages)
        masks.append(mask)

    imgs2, masks2 = aug.applyImagesWithMasks(imgs, masks)
