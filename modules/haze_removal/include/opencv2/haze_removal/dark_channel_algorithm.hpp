// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_HAZE_REMOVAL_HE_ALGORITHM_HPP
#define OPENCV_HAZE_REMOVAL_HE_ALGORITHM_HPP

#include "haze_removal_base.hpp"

namespace cv {
namespace haze_removal {

//! @addtogroup haze_removal
//! @{

/** @brief Dehazes a image using Dark Channel Prior

For details refer to @cite kaiminghe_cvpr09
*/
class CV_EXPORTS_W DarkChannelPriorHazeRemoval : public HazeRemovalBase
{
public:
    CV_WRAP void setKernel(int _erosionSize, int _erosionType);
    CV_WRAP void setKernel(InputArray _kernelForEroding);
    CV_WRAP void setPercentageBrightestPixelsForAtmoLight(float _percentageBrightestPixelsForAtmoLight);
    CV_WRAP void setOmega(float _omega);
    CV_WRAP void setGuidedFilterRadius(float _guidedFilterRadius);
    CV_WRAP void setGuidedFilterEps(float _guidedFilterEps);
    CV_WRAP void setTransmissionLowerBound(float _transmissionLowerBoundsetPlotLineColor);

    CV_WRAP static Ptr<DarkChannelPriorHazeRemoval> create();

protected:
    DarkChannelPriorHazeRemoval() {}
};

/** @brief Dehazes using haze_removal::DarkChannelPriorHazeRemoval in one call
@param _src input image you want to dehaze, must be a CV_8UC3  image
@param _dst dehazed image with same number of rows and columns as input in CV_8UC3 format
*/

CV_EXPORTS_W void darkChannelPriorHazeRemoval(InputArray _src, OutputArray _dst);

//! @}

}} // cv::haze_removal::

#endif
