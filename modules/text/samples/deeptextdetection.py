# -*- coding: utf-8 -*-
#!/usr/bin/python
import sys
import os
import cv2
import numpy as np

def main():
    print('\nDeeptextdetection.py')
    print('       A demo script of text box alogorithm of the paper:')
    print('       * Minghui Liao et al.: TextBoxes: A Fast Text Detector with a Single Deep Neural Network https://arxiv.org/abs/1611.06779\n')

    if (len(sys.argv) < 2):
        print(' (ERROR) You must call this script with an argument (path_to_image_to_be_processed)\n')
        quit()

    if not os.path.isfile('TextBoxes_icdar13.caffemodel') or not os.path.isfile('textbox.prototxt'):
        print " Model files not found in current directory. Aborting"
        print " See the documentation of text::TextDetectorCNN class to get download links."
        quit()

    img = cv2.imread(str(sys.argv[1]))
    textSpotter = cv2.text.TextDetectorCNN_create("textbox.prototxt", "TextBoxes_icdar13.caffemodel")
    rects, outProbs = textSpotter.detect(img);
    vis = img.copy()
    thres = 0.6

    for r in range(np.shape(rects)[0]):
        if outProbs[r] > thres:
            rect = rects[r]
            cv2.rectangle(vis, (rect[0],rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0), 2)

    cv2.imshow("Text detection result", vis)
    cv2.waitKey()

if __name__ == "__main__":
    main()
