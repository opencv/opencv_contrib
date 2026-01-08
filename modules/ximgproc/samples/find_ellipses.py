#!/usr/bin/python

'''
This example illustrates how to use cv.ximgproc.findEllipses function.

Usage:
    find_ellipses.py [<image_name>]
    image argument defaults to stuff.jpg
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import sys
import math

def main():
    try:
        fn = sys.argv[1]
    except IndexError:
        fn = 'stuff.jpg'

    src = cv.imread(cv.samples.findFile(fn))
    cv.imshow("source", src)

    ells = cv.ximgproc.findEllipses(src,scoreThreshold = 0.4, reliabilityThreshold = 0.7, centerDistanceThreshold = 0.02)

    if ells is not None:
        for i in range(len(ells)):
            center = (int(ells[i][0][0]), int(ells[i][0][1]))
            axes = (int(ells[i][0][2]),int(ells[i][0][3]))
            angle = ells[i][0][5] * 180 / math.pi
            color = (0, 0, 255)
            cv.ellipse(src, center, axes, angle,0, 360, color, 2, cv.LINE_AA)

    cv.imshow("detected ellipses", src)
    cv.waitKey(0)
    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
