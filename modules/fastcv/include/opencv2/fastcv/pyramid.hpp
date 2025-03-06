/*
 * Copyright (c) 2024-2025 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#ifndef OPENCV_FASTCV_PYRAMID_HPP
#define OPENCV_FASTCV_PYRAMID_HPP

#include <opencv2/core.hpp>

namespace cv {
namespace fastcv {

//! @addtogroup fastcv
//! @{

/**
 * @brief Creates a gradient pyramid from an image pyramid
 *        Note: The borders are ignored during gradient calculation.
 * @param pyr Input pyramid of 1-channel 8-bit images. Only continuous images are supported.
 * @param dx Horizontal Sobel gradient pyramid of the same size as pyr
 * @param dy Verical Sobel gradient pyramid of the same size as pyr
 * @param outType Type of output data, can be CV_8S, CV_16S or CV_32F
 */
CV_EXPORTS_W void sobelPyramid(InputArrayOfArrays pyr, OutputArrayOfArrays dx, OutputArrayOfArrays dy, int outType = CV_8S);

/**
 * @brief Builds an image pyramid of float32 arising from a single
    original image - that are successively downscaled w.r.t. the
    pre-set levels. This API supports both ORB scaling and scale down by half. 
 *
 * @param src Input single-channel image of type 8U or 32F
 * @param pyr Output array containing nLevels downscaled image copies
 * @param nLevels Number of pyramid levels to produce
 * @param scaleBy2 to scale images 2x down or by a factor of 1/(2)^(1/4) which is approximated as 0.8408964 (ORB downscaling),
 *                 ORB scaling is not supported for float point images
 * @param borderType how to process border, the options are BORDER_REFLECT (maps to FASTCV_BORDER_REFLECT),
 *                   BORDER_REFLECT_101 (maps to FASTCV_BORDER_REFLECT_V2) and BORDER_REPLICATE (maps to FASTCV_BORDER_REPLICATE).
 *                   Other border types are mapped to FASTCV_BORDER_UNDEFINED(border pixels are ignored). Currently, borders only
 *                   supported for downscaling by half, ignored for ORB scaling. Also ignored for float point images
 * @param borderValue what value should be used to fill border, ignored for float point images
 */
CV_EXPORTS_W void buildPyramid(InputArray src, OutputArrayOfArrays pyr, int nLevels, bool scaleBy2 = true,
                               int borderType = cv::BORDER_REFLECT, uint8_t borderValue = 0);

//! @}

} // fastcv::
} // cv::

#endif // OPENCV_FASTCV_PYRAMID_HPP
