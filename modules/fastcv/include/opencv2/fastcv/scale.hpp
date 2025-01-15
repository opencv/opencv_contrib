/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#ifndef OPENCV_FASTCV_SCALE_HPP
#define OPENCV_FASTCV_SCALE_HPP

#include <opencv2/core.hpp>

namespace cv {
namespace fastcv {

//! @addtogroup fastcv
//! @{

/**
 * @brief Down-scale the image by averaging each 2x2 pixel block.
 * 		  This function is not bit-exact with cv::resize but provides faster execution time on Qualcomm's processor.
 * @param _src The first input image data, type CV_8UC1, src height must be a multiple of 2
 * @param _dst The output image data, type CV_8UC1
*/
CV_EXPORTS_W void resizeDownBy2(cv::InputArray _src, cv::OutputArray _dst);

/**
 * @brief Down-scale the image by averaging each 4x4 pixel block.
 * 		  This function is not bit-exact with cv::resize but provides faster execution time on Qualcomm's processor.
 * @param _src The first input image data, type CV_8UC1, src height must be a multiple of 4
 * @param _dst The output image data, type CV_8UC1
*/
CV_EXPORTS_W void resizeDownBy4(cv::InputArray _src, cv::OutputArray _dst);

//! @}

} // fastcv::
} // cv::

#endif // OPENCV_FASTCV_SCALE_HPP
