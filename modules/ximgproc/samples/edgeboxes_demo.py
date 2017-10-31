#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This sample demonstrates structured edge detection and edgeboxes.
Usage:
  edgeboxes_demo.py [<model>] [<input_image>]
'''

import cv2
import numpy as np
import sys

if __name__ == '__main__':
    print(__doc__)

    model = sys.argv[1]
    im = cv2.imread(sys.argv[2])

    edge_detection = cv2.ximgproc.createStructuredEdgeDetection(model)
    rgb_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    edges = edge_detection.detectEdges(np.float32(rgb_im) / 255.0)

    orimap = edge_detection.computeOrientation(edges)
    edges = edge_detection.edgesNms(edges, orimap)

    edge_boxes = cv2.ximgproc.createEdgeBoxes()
    edge_boxes.setMaxBoxes(30)
    boxes = edge_boxes.getBoundingBoxes(edges, orimap)

    for b in boxes:
        x, y, w, h = b
        cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow("edges", edges)
    cv2.imshow("edgeboxes", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
