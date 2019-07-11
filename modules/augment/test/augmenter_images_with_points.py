try:
    import cv2 as cv
except ImportError:
    raise ImportError('Can\'t find OpenCV Python module. If you\'ve built it from sources without installation, '
                      'configure environment variable PYTHONPATH to "opencv_build_dir/lib" directory (with "python3" subdirectory if required)')
try:
    import numpy as np
except ImportError:
    raise ImportError('Can\'t find numpy library, please make sure it is installed')

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
    pointsArr = []

    for i in range(10):
        widthOfImages = np.random.randint(400, 2000)
        heightOfImages = np.random.randint(400, 2000)
        numberOfPoints = np.random.randint(1,100)

        img = np.random.rand(heightOfImages, widthOfImages)
        imgs.append(img)
        points = np.random.rand(numberOfPoints, 2)
        pointsArr.append(points)

    imgs2, pointsArr2 = aug.applyImagesWithPoints(imgs, pointsArr)
