// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_XIMGPROC_RIDGEFILTER_HPP__
#define __OPENCV_XIMGPROC_RIDGEFILTER_HPP__

#include <opencv2/core.hpp>

namespace cv
{
namespace ximgproc
{
//! @addtogroup ximgproc_filters
//! @{
/** @brief Ridge detection algorithm based on the Hessian Matrix.
 */
class CV_EXPORTS_W RidgeDetectionFilter : public Algorithm
{
public:
    /** @brief Apply ridge detection filter.
    @param img input image. Should be gray or BGR.
    @param out 32FC1 image with ridges.
    */
    CV_WRAP virtual void getRidges(InputArray img, OutputArray out) = 0;

    /** @brief Factory method, creates instance of RidgeDetectionFilter
    */
    CV_WRAP static Ptr<RidgeDetectionFilter> create();
};

//! @}
}
}
#endif
