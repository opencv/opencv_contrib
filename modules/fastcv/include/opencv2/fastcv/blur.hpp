/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#ifndef OPENCV_FASTCV_BLUR_HPP
#define OPENCV_FASTCV_BLUR_HPP

#include <opencv2/core.hpp>

namespace cv {
namespace fastcv {

/**
 * @defgroup fastcv Module-wrapper for FastCV hardware accelerated functions
 */

//! @addtogroup fastcv
//! @{

/**
 * @brief Gaussian blur with sigma = 0 and square kernel size. The way of handling borders is different with cv::GaussianBlur,
 *        leading to slight variations in the output.
 * @param _src Intput image with type CV_8UC1
 * @param _dst Output image with type CV_8UC1
 * @param kernel_size Filer kernel size. One of 3, 5, 11
 * @param blur_border If set to true, border is blurred by 0-padding adjacent values.(A variant of the constant border)
 *                    If set to false, borders up to half-kernel width are ignored (e.g. 1 pixel in the 3x3 case).
 *
 * @sa GaussianBlur
 */
CV_EXPORTS_W void gaussianBlur(InputArray _src, OutputArray _dst, int kernel_size = 3, bool blur_border = true);

/**
 * @brief NxN correlation with non-separable kernel. Borders up to half-kernel width are ignored
 * @param _src Intput image with type CV_8UC1
 * @param _dst Output image with type CV_8UC1, CV_16SC1 or CV_32FC1
 * @param ddepth The depth of output image
 * @param _kernel Filer kernel data
 *
 * @sa Filter2D
 */
CV_EXPORTS_W void filter2D(InputArray _src, OutputArray _dst, int ddepth, InputArray _kernel);

/**
 * @brief NxN correlation with separable kernel. If srcImg and dstImg point to the same address and srcStride equals to dstStride,
 *        it will do in-place. Borders up to half-kernel width are ignored.
 *        The way of handling overflow is different with OpenCV, this function will do right shift for
 *        the intermediate results and final result.
 * @param _src Intput image with type CV_8UC1
 * @param _dst Output image with type CV_8UC1, CV_16SC1
 * @param ddepth The depth of output image
 * @param _kernelX Filer kernel data in x direction
 * @param _kernelY Filer kernel data in Y direction (For CV_16SC1, the kernelX and kernelY should be same)
 *
 * @sa sepFilter2D
 */
CV_EXPORTS_W void sepFilter2D(InputArray _src, OutputArray _dst, int ddepth, InputArray _kernelX, InputArray _kernelY);
//! @}

} // fastcv::
} // cv::

#endif // OPENCV_FASTCV_BLUR_HPP
