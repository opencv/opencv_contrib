/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#ifndef OPENCV_WARP_HPP
#define OPENCV_WARP_HPP

#include <opencv2/imgproc.hpp>
namespace cv {
namespace fastcv {

/**
 * @defgroup fastcv Module-wrapper for FastCV hardware accelerated functions
*/

//! @addtogroup fastcv
//! @{

/**
 * @brief Perspective warp two images using the same transformation. Bi-linear interpolation is used where applicable.
 *        For example, to warp a grayscale image and an alpha image at the same time, or warp two color channels.
 * @param _src1     First input 8-bit image. Size of buffer is src1Stride*srcHeight bytes.
 * @param _src2     Second input 8-bit image. Size of buffer is src2Stride*srcHeight bytes.
 * @param _dst1     First warped output image (correspond to src1). Size of buffer is dst1Stride*dstHeight bytes, type CV_8UC1
 * @param _dst2     Second warped output image (correspond to src2). Size of buffer is dst2Stride*dstHeight bytes, type CV_8UC1
 * @param _M0       The 3x3 perspective transformation matrix (inversed map)
 * @param dsize     The output image size
*/
CV_EXPORTS_W void warpPerspective2Plane(InputArray _src1, InputArray _src2, OutputArray _dst1, OutputArray _dst2,
    InputArray _M0, Size dsize);

//! @}

}
}

#endif