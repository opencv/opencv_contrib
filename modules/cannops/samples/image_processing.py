# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.

import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser(description='This is a sample for image processing with Ascend NPU.')
parser.add_argument('image', help='path to input image')
parser.add_argument('output', help='path to output image')
args = parser.parse_args()

# read input image and generate guass noise
#! [input_noise]
img = cv2.imread(args.image)
# Generate gauss noise that will be added into the input image
gaussNoise = np.random.normal(0, 25,(img.shape[0], img.shape[1], img.shape[2])).astype(img.dtype)
#! [input_noise]

# setup cann
#! [setup]
cv2.cann.initAcl()
cv2.cann.setDevice(0)
#! [setup]

#! [image-process]
# add gauss noise to the image
output = cv2.cann.add(img, gaussNoise)
# rotate the image with a certain mode (0, 1 and 2, correspond to rotation of 90, 180
# and 270 degrees clockwise respectively)
output = cv2.cann.rotate(output, 0)
# flip the image with a certain mode (0, positive and negative number, correspond to flipping
# around the x-axis, y-axis and both axes respectively)
output = cv2.cann.flip(output, 0)
#! [image-process]

cv2.imwrite(args.output, output)

#! [tear-down-cann]
cv2.cann.finalizeAcl()
#! [tear-down-cann]
