/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#ifndef OPENCV_FASTCV_DRAW_HPP
#define OPENCV_FASTCV_DRAW_HPP

#include <opencv2/core.hpp>

namespace cv {
namespace fastcv {

//! @addtogroup fastcv
//! @{

/**
 * @brief Draw convex polygon
          This function fills the interior of a convex polygon with the specified color.
          Requires the width and stride to be multple of 8.
 * @param img Image to draw on. Should have up to 4 8-bit channels
 * @param pts Array of polygon points coordinates. Should contain N two-channel or 2*N one-channel 32-bit integer elements
 * @param color Color of drawn polygon stored as B,G,R and A(if supported)
 */
CV_EXPORTS_W void fillConvexPoly(InputOutputArray img, InputArray pts, Scalar color);

//! @}

} // fastcv::
} // cv::

#endif // OPENCV_FASTCV_DRAW_HPP
