/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#ifndef OPENCV_FASTCV_THRESH_HPP
#define OPENCV_FASTCV_THRESH_HPP

#include <opencv2/core.hpp>

namespace cv {
namespace fastcv {

//! @addtogroup fastcv
//! @{

/**
 * @brief Binarizes a grayscale image based on a pair of threshold values. The binarized image will be in the two values
 *        selected by user

 * @param src 8-bit grayscale image
 * @param dst Output image of the same size and type as input image, can be the same as input image
 * @param lowThresh The lower threshold value for binarization
 * @param highThresh The higher threshold value for binarization
 * @param trueValue The value assigned to the destination pixel if the source is within the range inclusively defined by the
 *                  pair of threshold values
 * @param falseValue The value assigned to the destination pixel if the source is out of the range inclusively defined by the
 *                   pair of threshold values
 */
CV_EXPORTS_W void thresholdRange(InputArray src, OutputArray dst, uint8_t lowThresh, uint8_t highThresh, uint8_t trueValue, uint8_t falseValue);

//! @}

} // fastcv::
} // cv::

#endif // OPENCV_FASTCV_THRESH_HPP
