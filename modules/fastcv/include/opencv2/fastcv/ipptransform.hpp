/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#ifndef OPENCV_FASTCV_IPPTRANSFORM_HPP
#define OPENCV_FASTCV_IPPTRANSFORM_HPP

#include <opencv2/core.hpp>

namespace cv {
namespace fastcv {

//! @addtogroup fastcv
//! @{

/**
 * @brief This function performs 8x8 forward discrete Cosine transform on input image
 * 
 * @param src Input image of type CV_8UC1
 * @param dst Output image of type CV_16SC1
 */
CV_EXPORTS_W void DCT(InputArray src, OutputArray dst);

/**
 * @brief This function performs 8x8 inverse discrete Cosine transform on input image
 *
 * @param src Input image of type CV_16SC1
 * @param dst Output image of type CV_8UC1
 */
CV_EXPORTS_W void IDCT(InputArray src, OutputArray dst);

//! @}

} // fastcv::
} // cv::

#endif // OPENCV_FASTCV_IPPTRANSFORM_HPP
