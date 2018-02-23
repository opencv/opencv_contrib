# USAGE - How to run this code ?
# python find_shapes.py --image shapes.png
#python findredlinedpolygonfromgooglemaps.py --image stanford.png 

import numpy as np
import argparse
import cv2 as cv

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image file")
args = vars(ap.parse_args())

# load the image
image = cv.imread(args["image"])
lower = np.array([20,0,155])
upper = np.array([255,120,250])
shapeMask = cv.inRange(image, lower, upper)

# find the contours in the mask
(cnts, _) = cv.findContours(shapeMask.copy(), cv.RETR_EXTERNAL,
    cv.CHAIN_APPROX_SIMPLE)
cv.imshow("Mask", shapeMask)

# loop over the contours
for c in cnts:
    cv.drawContours(image, [c], -1, (0, 255, 0), 2)
    cv.imshow("Image", image)
    cv.waitKey(0)