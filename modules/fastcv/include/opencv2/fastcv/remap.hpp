/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#ifndef OPENCV_FASTCV_REMAP_HPP
#define OPENCV_FASTCV_REMAP_HPP

#include <opencv2/core.hpp>

namespace cv {
namespace fastcv {

//! @addtogroup fastcv
//! @{

/**
 * @brief Applies a generic geometrical transformation to a greyscale CV_8UC1 image.
 * @param src The first input image data, type CV_8UC1
 * @param dst The output image data, type CV_8UC1
 * @param map1 Floating-point CV_32FC1 matrix with each element as the column coordinate of the mapped location in the source image
 * @param map2 Floating-point CV_32FC1 matrix with each element as the row coordinate of the mapped location in the source image.
 * @param interpolation Only INTER_NEAREST and INTER_LINEAR interpolation is supported
 * @param borderValue constant pixel value
*/
CV_EXPORTS_W void remap( InputArray src, OutputArray dst,
                       InputArray map1, InputArray map2,
                       int interpolation, int borderValue=0);

/**
 * @brief Applies a generic geometrical transformation to a 4-channel CV_8UC4 image with bilinear or nearest neighbor interpolation
 * @param src The first input image data, type CV_8UC4
 * @param dst The output image data, type CV_8UC4
 * @param map1 Floating-point CV_32FC1 matrix with each element as the column coordinate of the mapped location in the source image
 * @param map2 Floating-point CV_32FC1 matrix with each element as the row coordinate of the mapped location in the source image.
 * @param interpolation Only INTER_NEAREST and INTER_LINEAR interpolation is supported
*/
CV_EXPORTS_W void remapRGBA( InputArray src, OutputArray dst,
                             InputArray map1, InputArray map2, int interpolation);

//! @}

} // fastcv::
} // cv::

#endif // OPENCV_FASTCV_REMAP_HPP
