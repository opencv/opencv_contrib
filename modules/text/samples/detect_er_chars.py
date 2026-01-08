#!/usr/bin/python

import sys
import os

import cv2 as cv
import numpy as np

print('\ndetect_er_chars.py')
print('       A simple demo script using the Extremal Region Filter algorithm described in:')
print('       Neumann L., Matas J.: Real-Time Scene Text Localization and Recognition, CVPR 2012\n')


if (len(sys.argv) < 2):
  print(' (ERROR) You must call this script with an argument (path_to_image_to_be_processed)\n')
  quit()

pathname = os.path.dirname(sys.argv[0])

img  = cv.imread(str(sys.argv[1]))
gray = cv.imread(str(sys.argv[1]),0)

erc1 = cv.text.loadClassifierNM1(pathname+'/trained_classifierNM1.xml')
er1 = cv.text.createERFilterNM1(erc1)

erc2 = cv.text.loadClassifierNM2(pathname+'/trained_classifierNM2.xml')
er2 = cv.text.createERFilterNM2(erc2)

regions = cv.text.detectRegions(gray,er1,er2)

#Visualization
rects = [cv.boundingRect(p.reshape(-1, 1, 2)) for p in regions]
for rect in rects:
  cv.rectangle(img, rect[0:2], (rect[0]+rect[2],rect[1]+rect[3]), (0, 0, 0), 2)
for rect in rects:
  cv.rectangle(img, rect[0:2], (rect[0]+rect[2],rect[1]+rect[3]), (255, 255, 255), 1)
cv.imshow("Text detection result", img)
cv.waitKey(0)
