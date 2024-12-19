/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#ifndef OPENCV_FASTCV_SMOOTH_HPP
#define OPENCV_FASTCV_SMOOTH_HPP

#include <opencv2/core.hpp>

namespace cv {
namespace fastcv {

//! @addtogroup fastcv
//! @{

/**
 * @brief Recursive Bilateral Filtering

Different from traditional bilateral filtering, here the smoothing is actually performed in gradient domain.
The algorithm claims that it's more efficient than the original bilateral filtering in both image quality and computation.
See algorithm description in the paper Recursive Bilateral Filtering, ECCV2012 by Prof Yang Qingxiong
This function isn't bit-exact with cv::bilateralFilter but provides improved latency on Snapdragon processors.
 * @param src Input image, should have one CV_8U channel
 * @param dst Output array having one CV_8U channel
 * @param sigmaColor Sigma in the color space, the bigger the value the more color difference is smoothed by the algorithm
 * @param sigmaSpace Sigma in the coordinate space, the bigger the value the more distant pixels are smoothed
 */
CV_EXPORTS_W void bilateralRecursive(cv::InputArray src, cv::OutputArray dst, float sigmaColor = 0.03f, float sigmaSpace = 0.1f);

//! @}

} // fastcv::
} // cv::

#endif // OPENCV_FASTCV_SMOOTH_HPP
