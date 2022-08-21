// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#ifndef __OPENCV_ARUCO_HPP__
#define __OPENCV_ARUCO_HPP__

#include "opencv2/aruco_detector.hpp"
#include "opencv2/aruco/aruco_calib_pose.hpp"

namespace cv {
namespace aruco {


/**
@deprecated Use class ArucoDetector
*/
CV_EXPORTS_W void detectMarkers(InputArray image, const Ptr<Dictionary> &dictionary, OutputArrayOfArrays corners,
                                OutputArray ids, const Ptr<DetectorParameters> &parameters = DetectorParameters::create(),
                                OutputArrayOfArrays rejectedImgPoints = noArray());

/**
@deprecated Use class ArucoDetector
*/
CV_EXPORTS_W void refineDetectedMarkers(InputArray image,const  Ptr<Board> &board,
                                        InputOutputArrayOfArrays detectedCorners,
                                        InputOutputArray detectedIds, InputOutputArrayOfArrays rejectedCorners,
                                        InputArray cameraMatrix = noArray(), InputArray distCoeffs = noArray(),
                                        float minRepDistance = 10.f, float errorCorrectionRate = 3.f,
                                        bool checkAllOrders = true, OutputArray recoveredIdxs = noArray(),
                                        const Ptr<DetectorParameters> &parameters = DetectorParameters::create());

}
}

#endif
