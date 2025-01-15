/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#ifndef OPENCV_FASTCV_SHIFT_HPP
#define OPENCV_FASTCV_SHIFT_HPP

#include <opencv2/core.hpp>

namespace cv {
namespace fastcv {

//! @addtogroup fastcv
//! @{

/**
 * @brief Applies the meanshift procedure and obtains the final converged position.
          This function applies the meanshift procedure to an original image (usually a probability image)
          and obtains the final converged position. The converged position search will stop either it has reached
          the required accuracy or the maximum number of iterations. Moments used in the algorithm are calculated
          in floating point.
          This function isn't bit-exact with cv::meanShift but provides improved latency on Snapdragon processors.

 * @param src 8-bit, 32-bit int or 32-bit float grayscale image which is usually a probability image
 *            computed based on object histogram
 * @param rect Initial search window position which also returns the final converged window position
 * @param termCrit The criteria used to finish the MeanShift which consists of two termination criteria:
 *                 1) epsilon: required accuracy; 2) max_iter: maximum number of iterations
 * @return Iteration number at which the loop stopped
 */
CV_EXPORTS_W int meanShift(InputArray src, Rect& rect, TermCriteria termCrit);

//! @}

} // fastcv::
} // cv::

#endif // OPENCV_FASTCV_SHIFT_HPP
