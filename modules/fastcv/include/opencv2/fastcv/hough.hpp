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


/**
 * @brief Finds circles in a grayscale image using Hough transform.
 *        The radius of circle varies from 0 to max(srcWidth, srcHeight).
 *
 * @param src Input 8-bit image containing binary contour. Step should be divisible by 8, data start should be 128-bit aligned
 * @param circles Output array containing detected circles in a form (x, y, r) where all numbers are 32-bit integers
 * @param minDist Minimum distance between the centers of the detected circles
 * @param cannyThreshold The higher threshold of the two passed to the Canny() edge detector
 *                       (the lower one is twice smaller). Default is 100.
 * @param accThreshold The accumulator threshold for the circle centers at the detection
 *                     stage. The smaller it is, the more false circles may be detected.
 *                     Circles, corresponding to the larger accumulator values, will be
 *                     returned first. Default is 100.
 * @param minRadius Minimum circle radius, default is 0
 * @param maxRadius Maximum circle radius, default is 0
 */
CV_EXPORTS_W void houghCircles(InputArray src, OutputArray circles, uint32_t minDist,
                               uint32_t cannyThreshold = 100, uint32_t accThreshold = 100,
                               uint32_t minRadius = 0, uint32_t maxRadius = 0);

//! @}

} // fastcv::
} // cv::

#endif // OPENCV_FASTCV_HOUGH_HPP
