// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#ifndef __OPENCV_OIL_PAINTING_HPP__
#define __OPENCV_OIL_PAINTING_HPP__

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace cv
{
namespace xphoto
{

//! @addtogroup xphoto
//! @{

/** @brief oilPainting
See the book @cite Holzmann1988 for details.
@param src Input three-channel or one channel image (either CV_8UC3 or CV_8UC1)
@param dst Output image of the same size and type as src.
@param size neighbouring size is 2-size+1
@param dynRatio image is divided by dynRatio before histogram processing
@param code	color space conversion code(see ColorConversionCodes). Histogram will used only first plane
*/
CV_EXPORTS_W void oilPainting(InputArray src, OutputArray dst, int size, int dynRatio, int code);
/** @brief oilPainting
See the book @cite Holzmann1988 for details.
@param src Input three-channel or one channel image (either CV_8UC3 or CV_8UC1)
@param dst Output image of the same size and type as src.
@param size neighbouring size is 2-size+1
@param dynRatio image is divided by dynRatio before histogram processing
*/
CV_EXPORTS_W void oilPainting(InputArray src, OutputArray dst, int size, int dynRatio);
//! @}
}
}

#endif // __OPENCV_OIL_PAINTING_HPP__
