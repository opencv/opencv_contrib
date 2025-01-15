/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#ifndef OPENCV_FASTCV_BILATERALFILTER_HPP
#define OPENCV_FASTCV_BILATERALFILTER_HPP

#include <opencv2/core.hpp>

namespace cv {
namespace fastcv {

//! @addtogroup fastcv
//! @{

/**
 * @brief Applies Bilateral filter to an image considering d-pixel diameter of each pixel's neighborhood.
          This filter does not work inplace.

 * @param _src Intput image with type CV_8UC1
 * @param _dst Destination image with same type as _src
 * @param d kernel size (can be 5, 7 or 9)
 * @param sigmaColor Filter sigma in the color space.
                     Typical value is 50.0f.
                     Increasing this value means increasing the influence of the neighboring pixels of more different color to the smoothing result.
 * @param sigmaSpace Filter sigma in the coordinate space.
                     Typical value is 1.0f.
                     Increasing this value means increasing the influence of farther neighboring pixels within the kernel size distance to the smoothing result.
 * @param borderType border mode used to extrapolate pixels outside of the image
 */
CV_EXPORTS_W void bilateralFilter( InputArray _src, OutputArray _dst, int d,
                      float sigmaColor, float sigmaSpace,
                      int borderType = BORDER_DEFAULT );

//! @}

} // fastcv::
} // cv::

#endif // OPENCV_FASTCV_BILATERALFILTER_HPP
