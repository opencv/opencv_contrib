#
#  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
#
#  By downloading, copying, installing or using the software you agree to this license.
#  If you do not agree to this license, do not download, install,
#  copy or use the software.
#
#
#                          License Agreement
#                For Open Source Computer Vision Library
#
# Copyright (C) 2014, OpenCV Foundation, all rights reserved.
# Third party copyrights are property of their respective owners.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#   * Redistribution's of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#   * Redistribution's in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#   * The name of the copyright holders may not be used to endorse or promote products
#     derived from this software without specific prior written permission.
#
# This software is provided by the copyright holders and contributors "as is" and
# any express or implied warranties, including, but not limited to, the implied
# warranties of merchantability and fitness for a particular purpose are disclaimed.
# In no event shall the Intel Corporation or contributors be liable for any direct,
# indirect, incidental, special, exemplary, or consequential damages
# (including, but not limited to, procurement of substitute goods or services;
# loss of use, data, or profits; or business interruption) however caused
# and on any theory of liability, whether in contract, strict liability,
# or tort (including negligence or otherwise) arising in any way out of
# the use of this software, even if advised of the possibility of such damage.
#
# Module author: Tolga Birdal <tbirdal AT gmail.com>
# Wrapper author: Hamdi Sahloul <hamdisahloul AT hotmail.com>

import sys
import cv2
import numpy as np

def help(errorMessage):
    print("Program init error : %s" % errorMessage)
    print("\nUsage : python ppf_normal_computation.py [input model file] [output model file]")
    print("\nPlease start again with new parameters")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        help("Not enough input arguments")
        sys.exit(1)

    modelFileName = sys.argv[1]
    outputFileName = sys.argv[2]

    print("Loading points")
    points = cv2.ppf_match_3d.loadPLYSimple(modelFileName, 1)

    print("Computing normals")
    viewpoint = (0, 0, 0)
    _, pointsAndNormals = cv2.ppf_match_3d.computeNormalsPC3d(points, 6, False, viewpoint)

    print("Writing points")
    cv2.ppf_match_3d.writePLY(pointsAndNormals, outputFileName)
    #the following function can also be used for debugging purposes
    #cv2.ppf_match_3d.writePLYVisibleNormals(pointsAndNormals, outputFileName)

    print("Done")
