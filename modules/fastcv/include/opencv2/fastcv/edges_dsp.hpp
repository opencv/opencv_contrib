/*
 * Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#ifndef OPENCV_FASTCV_EDGES_DSP_HPP
#define OPENCV_FASTCV_EDGES_DSP_HPP

#include "opencv2/core/mat.hpp"

namespace cv {
namespace fastcv {
namespace dsp {

/**
* @defgroup fastcv Module-wrapper for FastCV hardware accelerated functions
*/

//! @addtogroup fastcv
//! @{

/**
 * @brief Canny edge detector applied to a 8 bit grayscale image
 * @param _src          Input image with type CV_8UC1
 * @param _dst          Output 8-bit image containing the edge detection results
 * @param lowThreshold  First threshold
 * @param highThreshold Second threshold
 * @param apertureSize  The Sobel kernel size for calculating gradient. Supported sizes are 3, 5 and 7.
 * @param L2gradient    L2 Gradient or L1 Gradient
*/
CV_EXPORTS void Canny(InputArray _src, OutputArray _dst, int lowThreshold, int highThreshold, int apertureSize = 3, bool L2gradient = false);
//! @}

} // dsp::
} // fastcv::
} // cv::

#endif //OPENCV_FASTCV_EDGES_DSP_HPP
