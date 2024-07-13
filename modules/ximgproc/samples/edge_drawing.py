#!/usr/bin/python

'''
This example script illustrates how to use cv.ximgproc.EdgeDrawing class.

It uses the OpenCV library to load an image, and then use the EdgeDrawing class
to detect edges, lines, and ellipses. The detected features are then drawn and displayed.

The main loop allows the user changing parameters of EdgeDrawing by pressing following keys:

to toggle the grayscale conversion press 'space' key
to increase MinPathLength value press '/' key
to decrease MinPathLength value press '*' key
to increase MinLineLength value press '+' key
to decrease MinLineLength value press '-' key
to toggle NFAValidation value press 'n' key
to toggle PFmode value press 'p' key
to save parameters to file press 's' key
to load parameters from file press 'l' key

The program exits when the Esc key is pressed.

Usage:
    ed.py [<image_name>]
    image argument defaults to board.jpg
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import random as rng
import sys

def EdgeDrawingDemo(src, ed, EDParams, convert_to_gray):
    rng.seed(12345)
    ssrc = np.zeros_like(src)
    lsrc = src.copy()
    esrc = src.copy()

    img_to_detect = cv.cvtColor(src, cv.COLOR_BGR2GRAY) if convert_to_gray else src

    cv.imshow("source image", img_to_detect)

    print("")
    print("convert_to_gray:", convert_to_gray)
    print("MinPathLength:", EDParams.MinPathLength)
    print("MinLineLength:", EDParams.MinLineLength)
    print("PFmode:", EDParams.PFmode)
    print("NFAValidation:", EDParams.NFAValidation)

    tm = cv.TickMeter()
    tm.start()

    # Detect edges
    # you should call this before detectLines() and detectEllipses()
    ed.detectEdges(img_to_detect)

    segments = ed.getSegments()
    lines = ed.detectLines()
    ellipses = ed.detectEllipses()

    tm.stop()

    print("Detection time : {:.2f} ms. using the parameters above".format(tm.getTimeMilli()))

    # Draw detected edge segments
    for segment in segments:
        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        cv.polylines(ssrc, [segment], False, color, 1, cv.LINE_8)

    cv.imshow("detected edge segments", ssrc)

    # Draw detected lines
    if lines is not None:  # Check if the lines have been found and only then iterate over these and add them to the image
        lines = np.uint16(np.around(lines))
        for line in lines:
            cv.line(lsrc, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (0, 0, 255), 1, cv.LINE_AA)

    cv.imshow("detected lines", lsrc)

    # Draw detected circles and ellipses
    if ellipses is not None:  # Check if circles and ellipses have been found and only then iterate over these and add them to the image
        for ellipse in ellipses:
            center = (int(ellipse[0][0]), int(ellipse[0][1]))
            axes = (int(ellipse[0][2] + ellipse[0][3]), int(ellipse[0][2] + ellipse[0][4]))
            angle = ellipse[0][5]

            color = (0, 255, 0) if ellipse[0][2] == 0 else (0, 0, 255)

            cv.ellipse(esrc, center, axes, angle, 0, 360, color, 2, cv.LINE_AA)

    cv.imshow("detected circles and ellipses", esrc)

def main():
    try:
        fn = sys.argv[1]
    except IndexError:
        fn = 'board.jpg'
    src = cv.imread(cv.samples.findFile(fn))
    if src is None:
        print("Error loading image")
        return

    ed = cv.ximgproc.createEdgeDrawing()

    # Set parameters (refer to the documentation for all parameters)
    EDParams = cv.ximgproc_EdgeDrawing_Params()
    EDParams.MinPathLength = 10     # try changing this value by pressing '/' and '*' keys
    EDParams.MinLineLength = 10     # try changing this value by pressing '+' and '-' keys
    EDParams.PFmode = False         # default value is False, try switching by pressing 'p' key
    EDParams.NFAValidation = True   # default value is True, try switching by pressing 'n' key

    convert_to_gray = True
    key = 0

    while key != 27:
        ed.setParams(EDParams)
        EdgeDrawingDemo(src, ed, EDParams, convert_to_gray)
        key = cv.waitKey()
        if key == 32:  # space key
            convert_to_gray = not convert_to_gray
        if key == 112: # 'p' key
            EDParams.PFmode = not EDParams.PFmode
        if key == 110: # 'n' key
            EDParams.NFAValidation = not EDParams.NFAValidation
        if key == 43: # '+' key
            EDParams.MinLineLength = EDParams.MinLineLength + 5
        if key == 45: # '-' key
            EDParams.MinLineLength = max(0, EDParams.MinLineLength - 5)
        if key == 47: # '/' key
            EDParams.MinPathLength = EDParams.MinPathLength + 20
        if key == 42: # '*' key
            EDParams.MinPathLength = max(0, EDParams.MinPathLength - 20)
        if key == 115: # 's' key
            fs = cv.FileStorage("ed-params.xml",cv.FileStorage_WRITE)
            EDParams.write(fs)
            fs.release()
            print("parameters saved to ed-params.xml")
        if key == 108: # 'l' key
            fs = cv.FileStorage("ed-params.xml",cv.FileStorage_READ)
            if fs.isOpened():
                EDParams.read(fs.root())
                fs.release()
                print("parameters loaded from ed-params.xml")

if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
