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
    print("\nUsage : python ppf_load_match.py [input model file] [input scene file]")
    print("\nPlease start again with new parameters")

def printPose(pose3D):
    print("-- Pose to Model Index %d: NumVotes = %d, Residual = %g" % (pose3D['modelIndex'], pose3D['numVotes'], pose3D['residual']))
    print(pose3D['pose'])
    print("")

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    # welcome message
    print("****************************************************")
    print("* Surface Matching demonstration : demonstrates the use of surface matching"
            " using point pair features.")
    print("* The sample loads a model and a scene, where the model lies in a different"
            " pose than the training.\n* It then trains the model and searches for it in the"
            " input scene. The detected poses are further refined by ICP\n* and printed to the "
            " standard output.")
    print("****************************************************")

    if len(sys.argv) < 3:
        help("Not enough input arguments")
        sys.exit(1)

    modelFileName = sys.argv[1]
    sceneFileName = sys.argv[2]

    pc = cv2.ppf_match_3d.loadPLYSimple(modelFileName, 1)

    # Now train the model
    print("Training...")
    tick1 = cv2.getTickCount()
    detector = cv2.ppf_match_3d.PPF3DDetector(1.0 / 40.0, 0.05)
    detector.trainModel(pc)
    tick2 = cv2.getTickCount()
    print("")
    print("Training complete in %g sec" % ((tick2 - tick1) / cv2.getTickFrequency()))
    print("Loading model...")

    # Read the scene
    pcTest = cv2.ppf_match_3d.loadPLYSimple(sceneFileName, 1)

    # Match the model to the scene and get the pose
    print("")
    print("Starting matching...")
    tick1 = cv2.getTickCount()
    results = detector.match(pcTest, 1.0 / 40.0, 0.05)
    tick2 = cv2.getTickCount()
    print("")
    print("PPF Elapsed Time %g sec" % ((tick2 - tick1) / cv2.getTickFrequency()))

    #check results size from match call above
    print("Number of matching poses: %u" % len(results))
    if len(results) == 0:
        print("")
        print("No matching poses found. Exiting.")
        sys.exit(0)

    # Get only first N results - but adjust to results size if num of results are less than that specified by N
    N = 2
    if len(results) < N:
        print("")
        print("Reducing matching poses to be reported (as specified in code): %u to the number of matches found: %u" % (N, len(results)))
        N = len(results)

    # Create an instance of ICP
    icp = cv2.ppf_match_3d.ICP(100, 0.005, 2.5, 8)
    t1 = cv2.getTickCount()

    # Register for all selected poses
    print("")
    print("Performing ICP on %u poses..." % N)
    _, results[:N] = icp.registerModelToScene(pc, pcTest, results[:N])
    t2 = cv2.getTickCount()

    print("")
    print("ICP Elapsed Time %g sec" % ((t2 - t1) / cv2.getTickFrequency()))

    print("Poses: ")
    # debug first five poses
    for i, result in enumerate(results[:N]):
        print("Pose Result %u" % i)
        printPose(result)
        if i == 0:
            pct = cv2.ppf_match_3d.transformPCPose(pc, result['pose'])
            cv2.ppf_match_3d.writePLY(pct, "para6700PCTrans.ply")
