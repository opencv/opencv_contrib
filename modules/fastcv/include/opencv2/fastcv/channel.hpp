/*
 * Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#ifndef OPENCV_FASTCV_CHANNEL_HPP
#define OPENCV_FASTCV_CHANNEL_HPP

#include <opencv2/core.hpp>

namespace cv {
namespace fastcv {

//! @addtogroup fastcv
//! @{

/**
 * @brief Creates one multi-channel mat out of several single-channel CV_8U mats.
 *        Optimized for Qualcomm's processors
 * @param mv input vector of matrices to be merged; all the matrices in mv must be of CV_8UC1 and have the same size
 *           Note: numbers of mats can be 2,3 or 4.
 * @param dst output array of depth CV_8U and same size as mv[0]; The number of channels
 *            will be the total number of matrices in the matrix array
 */
CV_EXPORTS_W void merge(InputArrayOfArrays mv, OutputArray dst);

//! @}

//! @addtogroup fastcv
//! @{

/**
 * @brief Splits an CV_8U multi-channel mat into several CV_8UC1 mats
 *        Optimized for Qualcomm's processors
 * @param src input 2,3 or 4 channel mat of depth CV_8U
 * @param mv  output vector of size src.channels() of CV_8UC1 mats
 */
CV_EXPORTS_W void split(InputArray src, OutputArrayOfArrays mv);

//! @}

} // fastcv::
} // cv::

#endif // OPENCV_FASTCV_CHANNEL_HPP
