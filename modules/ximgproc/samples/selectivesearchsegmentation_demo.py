#!/usr/bin/env python

'''
A program demonstrating the use and capabilities of a particular image segmentation algorithm described
in Jasper R. R. Uijlings, Koen E. A. van de Sande, Theo Gevers, Arnold W. M. Smeulders:
    "Selective Search for Object Recognition"
International Journal of Computer Vision, Volume 104 (2), page 154-171, 2013
Usage:
    ./selectivesearchsegmentation_demo.py input_image (single|fast|quality)
Use "a" to display less rects, 'd' to display more rects, "q" to quit.
'''

import cv2
import sys

if __name__ == '__main__':
    img = cv2.imread(sys.argv[1])

    cv2.setUseOptimized(True)
    cv2.setNumThreads(8)

    gs = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    gs.setBaseImage(img)

    if (sys.argv[2][0] == 's'):
        gs.switchToSingleStrategy()

    elif (sys.argv[2][0] == 'f'):
        gs.switchToSelectiveSearchFast()

    elif (sys.argv[2][0] == 'q'):
        gs.switchToSelectiveSearchQuality()
    else:
        print(__doc__)
        sys.exit(1)

    rects = gs.process()
    nb_rects = 10

    while True:
        wimg = img.copy()

        for i in range(len(rects)):
            if (i < nb_rects):
                x, y, w, h = rects[i]
                cv2.rectangle(wimg, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow("Output", wimg);
        c = cv2.waitKey()

        if (c == 100):
            nb_rects += 10

        elif (c == 97 and nb_rects > 10):
            nb_rects -= 10

        elif (c == 113):
            break

    cv2.destroyAllWindows()
