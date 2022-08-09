// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_ELLIPSE_DETECTOR_HPP__
#define __OPENCV_ELLIPSE_DETECTOR_HPP__

#include <opencv2/core.hpp>

namespace cv {
namespace ximgproc {

//! @addtogroup ximgproc_ellipse_detector
//! @{

/**
@brief finds ellipses fastly in an image using projective invariant pruning.

@param image input image, could be gray or color.
@param ellipses output vector of found ellipses. each vector is encoded as five float $x, y, a, b, radius, score$.
@param scoreThreshold float, the threshold of ellipse score.
@param reliabilityThreshold float, the threshold of reliability.
@param centerDistanceThreshold float, the threshold of center distance.
*/
CV_EXPORTS_W void ellipseDetector(
    InputArray _image, OutputArray _ellipses,
    float scoreThreshold = 0.7f, float reliabilityThreshold = 0.5f,
    float centerDistanceThreshold = 0.05f
);
//! @} ximgproc_ellipse_detector
}
}
#endif