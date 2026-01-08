#!/usr/bin/env python

'''
This example shows the functionalities of lines extraction finished by LSDDetector class.

USAGE: lsd_lines_extraction.py [<path_to_input_image>]
'''

import sys
import cv2 as cv

if __name__ == '__main__':
    print(__doc__)

    if len(sys.argv) > 1:
        fname = sys.argv[1]
    else :
        fname = '../data/corridor.jpg'

    img = cv.imread(fname)

    if img is None:
        print('Failed to load image file:', fname)
        sys.exit(1)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    lsd = cv.line_descriptor_LSDDetector.createLSDDetector()

    lines = lsd.detect(gray, 2, 1)
    for kl in lines:
        if kl.octave == 0:
            # cv.line only accepts integer coordinate
            pt1 = (int(kl.startPointX), int(kl.startPointY))
            pt2 = (int(kl.endPointX), int(kl.endPointY))
            cv.line(img, pt1, pt2, [255, 0, 0], 2)

    cv.imshow('output', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
