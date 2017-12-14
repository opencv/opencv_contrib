#!/usr/bin/python

import sys
import os

import cv2 as cv
import numpy as np

print('\ntextdetection.py')
print('       A demo script of the Extremal Region Filter algorithm described in:')
print('       Neumann L., Matas J.: Real-Time Scene Text Localization and Recognition, CVPR 2012\n')


if (len(sys.argv) < 2):
  print(' (ERROR) You must call this script with an argument (path_to_image_to_be_processed)\n')
  quit()

pathname = os.path.dirname(sys.argv[0])


img      = cv.imread(str(sys.argv[1]))
# for visualization
vis      = img.copy()


# Extract channels to be processed individually
channels = cv.text.computeNMChannels(img)
# Append negative channels to detect ER- (bright regions over dark background)
cn = len(channels)-1
for c in range(0,cn):
  channels.append((255-channels[c]))

# Apply the default cascade classifier to each independent channel (could be done in parallel)
print("Extracting Class Specific Extremal Regions from "+str(len(channels))+" channels ...")
print("    (...) this may take a while (...)")
for channel in channels:

  erc1 = cv.text.loadClassifierNM1(pathname+'/trained_classifierNM1.xml')
  er1 = cv.text.createERFilterNM1(erc1,16,0.00015,0.13,0.2,True,0.1)

  erc2 = cv.text.loadClassifierNM2(pathname+'/trained_classifierNM2.xml')
  er2 = cv.text.createERFilterNM2(erc2,0.5)

  regions = cv.text.detectRegions(channel,er1,er2)

  rects = cv.text.erGrouping(img,channel,[r.tolist() for r in regions])
  #rects = cv.text.erGrouping(img,channel,[x.tolist() for x in regions], cv.text.ERGROUPING_ORIENTATION_ANY,'../../GSoC2014/opencv_contrib/modules/text/samples/trained_classifier_erGrouping.xml',0.5)

  #Visualization
  for r in range(0,np.shape(rects)[0]):
    rect = rects[r]
    cv.rectangle(vis, (rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), (0, 0, 0), 2)
    cv.rectangle(vis, (rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), (255, 255, 255), 1)


#Visualization
cv.imshow("Text detection result", vis)
cv.waitKey(0)
