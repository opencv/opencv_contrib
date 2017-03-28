#!/usr/bin/python

import sys
import os

import cv2
import numpy as np

print('\ntextdetection.py')
print('       A demo script of the Extremal Region Filter algorithm described in:')
print('       Neumann L., Matas J.: Real-Time Scene Text Localization and Recognition, CVPR 2012\n')


if (len(sys.argv) < 2):
  print(' (ERROR) You must call this script with an argument (path_to_image_to_be_processed)\n')
  quit()

# Grouping overlapping rects into one larger rect
def rectGrouping(rects):
    # Create buckets equal to number of rect
    buckets = [[] for i in range(len(rects))]
    # Initialising first bucket
    buckets[0].append(0)
    # Implemented an O(n^3) algorithm to merge rects, possible to do it more efficiently
    for i in range(1,len(rects)):
        flag = False
        for bucket in buckets:
            for j in range(len(bucket)):
                if rectOverlap(rects[bucket[j]], rects[i]):
                    bucket.append(i)
                    flag = True
                    break
            if flag:
                break
        if not flag:
            buckets[i].append(i)
    # Creating bigger rects from the already grouped smaller rects
    new_rects = list()
    for bucket in buckets:
        if len(bucket) > 0:
            rect_bucket = [rects[i] for i in bucket]
            new_rect = {
                "left" : min([r["left"] for r in rect_bucket]),
                "top" : min([r["top"] for r in rect_bucket]),
                "right" : max([r["right"] for r in rect_bucket]),
                "bottom" : max([r["bottom"] for r in rect_bucket]),
            }
            new_rects.append(new_rect)
    return new_rects
# Checks whether two given rects overlap or not
def rectOverlap(r1 , r2):
    h_over = (r1["left"] <= r2["left"] and r1["right"] >= r2["left"]) or (r2["left"] <= r1["left"] and r2["right"] >= r1["left"])
    v_over = (r1["top"] <= r2["top"] and r1["bottom"] >= r2["top"]) or (r2["top"] <= r1["top"] and r2["bottom"] >= r1["top"])
    return h_over and v_over



# Path is extracted for loading the trained ER filters, can be given explicitly
pathname = os.path.dirname(sys.argv[0])

img = cv2.imread(str(sys.argv[1]))
# Copying image for visualization
vis = img.copy()

# Extracting the channels to process individually
channels = cv2.text.computeNMChannels(img)
# Adding negative channels to detect bright regions over dark background, ER-
cn = len(channels)-1
for c in range(0,cn):
  channels.append((255-channels[c]))

# Applying both ER Filter classifiers to each independent channel, (could do it in parallel)

print("Extracting Class Specific Extremal Regions from " + str(len(channels)) + " channels ...")

# Creating lists to store all regions and rects to feed into the Tesseract OCR
rects = list()
regions = list()

for channel in channels:

  erc1 = cv2.text.loadClassifierNM1(pathname + '/trained_classifierNM1.xml')
  er1 = cv2.text.createERFilterNM1(erc1, 16, 0.00015, 0.13, 0.2, True, 0.1)

  erc2 = cv2.text.loadClassifierNM2(pathname + '/trained_classifierNM2.xml')
  er2 = cv2.text.createERFilterNM2(erc2, 0.5)

  channel_regions = cv2.text.detectRegions(channel, er1, er2)
  channel_rects = cv2.text.erGrouping( img, channel, [region.tolist() for region in channel_regions] )

  rects.append(channel_rects)

nm_rects = list()
for channel_rects in rects:
    for rect in channel_rects:
        # NOTE: numpy uses (y,x) addressing rather than (x,y)
        Rect = {
            "top" : rect[1],
            "bottom" : rect[1] + rect[3],
            "left" : rect[0],
            "right" : rect[0] + rect[2],
        }
        nm_rects.append(Rect)

new_rects = rectGrouping(nm_rects)

# Visualization
for rect in new_rects:
    cv2.rectangle(vis, (rect["left"], rect["top"]), (rect["right"],rect["bottom"]), (0,255,0), 2)


#Visualization
cv2.imshow("Text detection result", vis)
cv2.waitKey(0)
