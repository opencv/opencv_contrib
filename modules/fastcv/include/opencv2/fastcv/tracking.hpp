/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#ifndef OPENCV_FASTCV_TRACKING_HPP
#define OPENCV_FASTCV_TRACKING_HPP

#include <opencv2/core.hpp>

namespace cv {
namespace fastcv {

//! @addtogroup fastcv
//! @{

/**
 * @brief Calculates sparse optical flow using Lucas-Kanade algorithm
 *		  accepts 8-bit unsigned integer image
 *		  Provides faster execution time on Qualcomm's processor 
 * @param src Input single-channel image of type 8U, initial motion frame
 * @param dst Input single-channel image of type 8U, final motion frame, should have the same size and stride as initial frame
 * @param srcPyr Pyramid built from intial motion frame
 * @param dstPyr Pyramid built from final motion frame
 * @param ptsIn Array of initial subpixel coordinates of starting points, should contain 32F 2D elements
 * @param ptsOut Output array of calculated final points, should contain 32F 2D elements
 * @param ptsEst Input array of estimations for final points, should contain 32F 2D elements, can be empty
 * @param statusVec Output array of int32 values indicating status of each feature, can be empty
 * @param winSize Size of window for optical flow searching. Width and height ust be odd numbers. Suggested values are 5, 7 or 9
 * @param termCriteria Termination criteria containing max number of iterations, max epsilon and stop condition
 */
CV_EXPORTS_W void trackOpticalFlowLK(InputArray src, InputArray dst,
                                     InputArrayOfArrays srcPyr, InputArrayOfArrays dstPyr,
                                     InputArray ptsIn, OutputArray ptsOut, InputArray ptsEst,
                                     OutputArray statusVec, cv::Size winSize = cv::Size(7, 7),
                                     cv::TermCriteria termCriteria = cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS,
                                                                                      /* maxIterations */ 7, /* maxEpsilon */ 0.03f * 0.03f));

/**
 * @brief Overload for v1 of the LK tracking function
 *
 * @param src Input single-channel image of type 8U, initial motion frame
 * @param dst Input single-channel image of type 8U, final motion frame, should have the same size and stride as initial frame
 * @param srcPyr Pyramid built from intial motion frame
 * @param dstPyr Pyramid built from final motion frame
 * @param srcDxPyr Pyramid of Sobel derivative by X of srcPyr
 * @param srcDyPyr Pyramid of Sobel derivative by Y of srcPyr
 * @param ptsIn Array of initial subpixel coordinates of starting points, should contain 32F 2D elements
 * @param ptsOut Output array of calculated final points, should contain 32F 2D elements
 * @param statusVec Output array of int32 values indicating status of each feature, can be empty
 * @param winSize Size of window for optical flow searching. Width and height ust be odd numbers. Suggested values are 5, 7 or 9
 * @param maxIterations Maximum number of iterations to try
 */
CV_EXPORTS_W void trackOpticalFlowLK(InputArray src, InputArray dst,
                                     InputArrayOfArrays srcPyr, InputArrayOfArrays dstPyr,
                                     InputArrayOfArrays srcDxPyr, InputArrayOfArrays srcDyPyr,
                                     InputArray ptsIn, OutputArray ptsOut,
                                     OutputArray statusVec, cv::Size winSize = cv::Size(7, 7), int maxIterations = 7);

//! @}

} // fastcv::
} // cv::

#endif // OPENCV_FASTCV_TRACKING_HPP
