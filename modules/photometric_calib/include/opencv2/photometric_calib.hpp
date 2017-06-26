// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_PHOTOMETRIC_CALIB_HPP__
#define __OPENCV_PHOTOMETRIC_CALIB_HPP__

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

#include <vector>
#include <string>
#include <iostream>
#include <fstream>

/** @defgroup photometric_calib Photometric Calibration
*/

namespace cv { namespace photometric_calib{

//! @addtogroup photometric_calib
//! @{

class CV_EXPORTS PhotometricCalibrator : public Algorithm
{
public:
    bool validImgs(const std::vector <Mat> &inputImgs, const std::vector<double> &exposureTime);
};

//! @}

}} // namespace photometric_calib, cv

#endif