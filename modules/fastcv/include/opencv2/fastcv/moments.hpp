/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#ifndef OPENCV_FASTCV_MOMENTS_HPP
#define OPENCV_FASTCV_MOMENTS_HPP

#include <opencv2/core.hpp>

namespace cv {
namespace fastcv {

//! @addtogroup fastcv
//! @{

/**
 * @brief Calculates all of the moments up to the third order of the image pixels' intensities
 *         The results are returned in the structure cv::Moments. This function cv::fastcv::moments()
 *         calculate the moments using floating point calculations whereas cv::moments() calculate moments using double.
 * @param _src      Input image with type CV_8UC1, CV_32SC1, CV_32FC1
 * @param binary    If true, assumes the image to be binary (0x00 for black, 0xff for white), otherwise assumes the image to be
 *                  grayscale.
 */
CV_EXPORTS cv::Moments moments(InputArray _src, bool binary);

//! @}

} // fastcv::
} // cv::

#endif // OPENCV_FASTCV_MOMENTS_HPP
