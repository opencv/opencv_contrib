/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#ifndef OPENCV_FASTCV_HISTOGRAM_HPP
#define OPENCV_FASTCV_HISTOGRAM_HPP

#include <opencv2/core.hpp>

namespace cv {
namespace fastcv {

//! @addtogroup fastcv
//! @{

/**
 * @brief Calculates histogram of input image. This function implements specific use case of
 *        256-bin histogram calculation for 8u single channel images in an optimized way.
 * @param _src Intput image with type CV_8UC1
 * @param _hist Output histogram of type int of 256 bins
 */
CV_EXPORTS_W void calcHist( InputArray _src, OutputArray _hist );
//! @}

} // fastcv::
} // cv::

#endif // OPENCV_FASTCV_HISTOGRAM_HPP
