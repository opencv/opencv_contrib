#!/usr/bin/env python

'''
This sample demonstrates SEEDS Superpixels segmentation
Use [space] to toggle output mode

Usage:
  seeds.py [<video source>]

'''

import numpy as np
import cv2 as cv

# built-in module
import sys


if __name__ == '__main__':
    print(__doc__)

    try:
        fn = sys.argv[1]
    except:
        fn = 0

    def nothing(*arg):
        pass

    cv.namedWindow('SEEDS')
    cv.createTrackbar('Number of Superpixels', 'SEEDS', 400, 1000, nothing)
    cv.createTrackbar('Iterations', 'SEEDS', 4, 12, nothing)

    seeds = None
    display_mode = 0
    num_superpixels = 400
    prior = 2
    num_levels = 4
    num_histogram_bins = 5

    cap = cv.VideoCapture(fn)
    while True:
        flag, img = cap.read()
        converted_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        height,width,channels = converted_img.shape
        num_superpixels_new = cv.getTrackbarPos('Number of Superpixels', 'SEEDS')
        num_iterations = cv.getTrackbarPos('Iterations', 'SEEDS')

        if not seeds or num_superpixels_new != num_superpixels:
            num_superpixels = num_superpixels_new
            seeds = cv.ximgproc.createSuperpixelSEEDS(width, height, channels,
                    num_superpixels, num_levels, prior, num_histogram_bins)
            color_img = np.zeros((height,width,3), np.uint8)
            color_img[:] = (0, 0, 255)

        seeds.iterate(converted_img, num_iterations)

        # retrieve the segmentation result
        labels = seeds.getLabels()


        # labels output: use the last x bits to determine the color
        num_label_bits = 2
        labels &= (1<<num_label_bits)-1
        labels *= 1<<(16-num_label_bits)


        mask = seeds.getLabelContourMask(False)

        # stitch foreground & background together
        mask_inv = cv.bitwise_not(mask)
        result_bg = cv.bitwise_and(img, img, mask=mask_inv)
        result_fg = cv.bitwise_and(color_img, color_img, mask=mask)
        result = cv.add(result_bg, result_fg)

        if display_mode == 0:
            cv.imshow('SEEDS', result)
        elif display_mode == 1:
            cv.imshow('SEEDS', mask)
        else:
            cv.imshow('SEEDS', labels)

        ch = cv.waitKey(1)
        if ch == 27:
            break
        elif ch & 0xff == ord(' '):
            display_mode = (display_mode + 1) % 3
    cv.destroyAllWindows()
