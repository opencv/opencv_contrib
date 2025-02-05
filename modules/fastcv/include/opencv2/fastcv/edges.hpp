/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#ifndef OPENCV_EDGES_HPP
#define OPENCV_EDGES_HPP

#include "opencv2/core/mat.hpp"

namespace cv {
namespace fastcv {
/**
 * @defgroup fastcv Module-wrapper for FastCV hardware accelerated functions
 */

//! @addtogroup fastcv
//! @{

/**
 * @brief Creates a 2D gradient image from source luminance data without normalization.
 *        Calculate X direction 1 order derivative or Y direction 1 order derivative or both at the same time, .
 * @param _src          Input image with type CV_8UC1
 * @param _dx           Buffer to store horizontal gradient. Must be (dxyStride)*(height) bytes in size.
 *                      If NULL, the horizontal gradient will not be calculated.
 * @param _dy           Buffer to store vertical gradient. Must be (dxyStride)*(height) bytes in size.
 *                      If NULL, the vertical gradient will not be calculated
 * @param kernel_size   Sobel kernel size, support 3x3, 5x5, 7x7
 * @param borderType    Border type, support BORDER_CONSTANT, BORDER_REPLICATE
 * @param borderValue   Border value for constant border
*/
CV_EXPORTS_W void sobel(InputArray _src, OutputArray _dx, OutputArray _dy, int kernel_size, int borderType, int borderValue);

/**
 * @brief Creates a 2D gradient image from source luminance data without normalization.
 *        This function computes central differences on 3x3 neighborhood and then convolves the result with Sobel kernel,
 *        borders up to half-kernel width are ignored.
 * @param _src          Input image with type CV_8UC1
 * @param _dst          If _dsty is given, buffer to store horizontal gradient, otherwise, output 8-bit image of |dx|+|dy|.
 *                      Size of buffer is (srcwidth)*(srcheight) bytes
 * @param _dsty         (Optional)Buffer to store vertical gradient. Must be (srcwidth)*(srcheight) in size.
 * @param ddepth        The depth of output image CV_8SC1,CV_16SC1,CV_32FC1,
 * @param normalization If do normalization for the result
*/
CV_EXPORTS_W void sobel3x3u8(InputArray _src, OutputArray _dst, OutputArray _dsty = noArray(), int ddepth = CV_8U,
    bool normalization = false);

//! @}

}
}

#endif
