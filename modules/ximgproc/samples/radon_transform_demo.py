# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.

import numpy as np
import cv2 as cv

if __name__ == "__main__":
    src = cv.imread("peilin_plane.png", cv.IMREAD_GRAYSCALE)
    radon = cv.ximgproc.RadonTransform(src).astype(np.float32)
    cv.imshow("src image", src)
    cv.imshow("Radon transform", radon)
    cv.waitKey()
