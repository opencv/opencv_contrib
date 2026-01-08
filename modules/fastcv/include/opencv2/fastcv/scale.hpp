/*
 * Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
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
 * @brief Down-scales the image using specified scaling factors or dimensions.
 *        This function supports both single-channel (CV_8UC1) and two-channel (CV_8UC2) images.
 * 
 * @param _src The input image data, type CV_8UC1 or CV_8UC2.
 * @param _dst The output image data, type CV_8UC1 or CV_8UC2.
 * @param dsize The desired size of the output image. If empty, it is calculated using inv_scale_x and inv_scale_y.
 * @param inv_scale_x The inverse scaling factor for the width. If dsize is provided, this parameter is ignored.
 * @param inv_scale_y The inverse scaling factor for the height. If dsize is provided, this parameter is ignored.
 * 
 * @note If dsize is not specified, inv_scale_x and inv_scale_y must be strictly positive.
 */
CV_EXPORTS_W void resizeDown(cv::InputArray _src, cv::OutputArray _dst, Size dsize, double inv_scale_x, double inv_scale_y);

//! @}

} // fastcv::
} // cv::

#endif // OPENCV_FASTCV_SCALE_HPP
