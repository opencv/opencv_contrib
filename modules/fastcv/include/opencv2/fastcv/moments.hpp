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
          The results are returned in the structure cv::Moments.
 * @param _src Input image with type CV_8UC1, CV_32SC1, CV_32FC1
 * @param binary If 1, binary image (0x00-black, oxff-white); if 0, grayscale image
 */
CV_EXPORTS cv::Moments moments(InputArray _src, bool binary);

//! @}

} // fastcv::
} // cv::

#endif // OPENCV_FASTCV_MOMENTS_HPP
