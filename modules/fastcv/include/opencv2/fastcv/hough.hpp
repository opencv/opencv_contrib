/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#ifndef OPENCV_FASTCV_HOUGH_HPP
#define OPENCV_FASTCV_HOUGH_HPP

#include <opencv2/core.hpp>

namespace cv {
namespace fastcv {

//! @addtogroup fastcv
//! @{

/**
 * @brief Performs Hough Line detection
 *
 * @param src Input 8-bit image containing binary contour. Width and step should be divisible by 8
 * @param lines Output array containing detected lines in a form of (x1, y1, x2, y2) where all numbers are 32-bit floats
 * @param threshold Controls the minimal length of a detected line. Value must be between 0.0 and 1.0
 *                  Values close to 1.0 reduces the number of detected lines. Values close to 0.0
 *                  detect more lines, but may be noisy. Recommended value is 0.25.
 */
CV_EXPORTS_W void houghLines(InputArray src, OutputArray lines, double threshold = 0.25);

//! @}

} // fastcv::
} // cv::

#endif // OPENCV_FASTCV_HOUGH_HPP
