// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_HAZE_REMOVAL_PRECOMP_H
#define OPENCV_HAZE_REMOVAL_PRECOMP_H

#include "opencv2/core.hpp"
#include "opencv2/haze_removal.hpp"


namespace cv{ namespace haze_removal {

class HazeRemovalBase::HazeRemovalImpl
{
public:
    virtual void dehaze(cv::InputArray _src, cv::OutputArray _dst) = 0;
    virtual ~HazeRemovalImpl() {}
};

}} // cv::haze_removal::

#endif // OPENCV_HAZE_REMOVAL_PRECOMP_H
