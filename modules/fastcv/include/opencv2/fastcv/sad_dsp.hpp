/*
 * Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#ifndef OPENCV_FASTCV_SAD_HPP
#define OPENCV_FASTCV_SAD_HPP

#include <opencv2/core.hpp>

namespace cv {
namespace fastcv {
namespace dsp {

/**
 * @defgroup fastcv Module-wrapper for FastCV hardware accelerated functions
 */

//! @addtogroup fastcv
//! @{
/**
 * @brief Sum of absolute differences of an image against an 8x8 template.
 * @param _patch The first input image data, type CV_8UC1
 * @param _src The input image data, type CV_8UC1
 * @param _dst The output image data, type CV_16UC1
*/
CV_EXPORTS void sumOfAbsoluteDiffs(cv::InputArray _patch, cv::InputArray _src, cv::OutputArray _dst);
//! @}

} // dsp::
} // fastcv::
} // cv::

#endif // OPENCV_FASTCV_SAD_HPP
