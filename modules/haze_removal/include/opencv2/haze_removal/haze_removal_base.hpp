// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_HAZE_REMOVAL_HAZE_REMOVAL_BASE_HPP
#define OPENCV_HAZE_REMOVAL_HAZE_REMOVAL_BASE_HPP

#include "opencv2/core.hpp"

namespace cv {
namespace haze_removal {

//! @addtogroup haze_removal
//! @{

/** @brief The base class for haze removal algorithm
 */

class CV_EXPORTS_W HazeRemovalBase : public Algorithm
{
public:
    class HazeRemovalImpl;

    ~HazeRemovalBase();
    /** @brief Dehazes a given image
        @param _src hazy image as input
        @param _dst output image with haze removed
    */
    CV_WRAP void dehaze(cv::InputArray _src, cv::OutputArray _dst);

protected:
    HazeRemovalBase();
    Ptr<HazeRemovalImpl> pImpl;
};

//! @}

}} // cv::haze_removal::

#endif
