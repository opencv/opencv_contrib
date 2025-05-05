/*
 * Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#ifndef OPENCV_FASTCV_THRESH_DSP_HPP
#define OPENCV_FASTCV_THRESH_DSP_HPP

#include <opencv2/core.hpp>

namespace cv {
namespace fastcv {
namespace dsp {

//! @addtogroup fastcv
//! @{

/**
 * @brief Binarizes a grayscale image using Otsu's method.
 *        Sets the pixel to max(255) if it's value is greater than the threshold;
 *        else, set the pixel to min(0). The threshold is searched that minimizes
 *        the intra-class variance (the variance within the class).
 * 
 * @param _src Input 8-bit grayscale image. Size of buffer is srcStride*srcHeight bytes.
 * @param _dst Output 8-bit binarized image. Size of buffer is dstStride*srcHeight bytes.
 * @param type Threshold type that can be either 0 or 1.
 *             NOTE: For threshold type=0, the pixel is set as
 *             maxValue if it's value is greater than the threshold; else, it is set as zero.
 *             For threshold type=1, the pixel is set as zero if it's
 *             value is greater than the threshold; else, it is set as maxValue.
 */
CV_EXPORTS void thresholdOtsu(InputArray _src, OutputArray _dst, bool type);

//! @}
} // dsp::
} // fastcv::
} // cv::

#endif // OPENCV_FASTCV_THRESH_DSP_HPP