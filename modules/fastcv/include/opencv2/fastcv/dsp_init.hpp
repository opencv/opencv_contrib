/*
 * Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#ifndef OPENCV_FASTCV_DSP_INIT_HPP
#define OPENCV_FASTCV_DSP_INIT_HPP

#include <opencv2/core.hpp>

namespace cv {
namespace fastcv {
namespace dsp {

//! @addtogroup fastcv
//! @{

/**
 * @brief Initializes the FastCV DSP environment.
 * 
 * This function sets up the necessary environment and resources for the DSP to operate.
 * It must be called once at the very beginning of the use case or program to ensure that 
 * the DSP is properly initialized before any DSP-related operations are performed.
 *
 * @note This function must be called at the start of the use case or program, before any 
 *       DSP-related operations.
 * 
 * @return int Returns 0 on success, and a non-zero value on failure.
 */
CV_EXPORTS int fcvdspinit();

/**
 * @brief Deinitializes the FastCV DSP environment.
 * 
 * This function releases the resources and environment set up by the 'fcvdspinit' function.
 * It should be called before the use case or program exits to ensure that all DSP resources 
 * are properly cleaned up and no memory leaks occur.
 *
 * @note This function must be called at the end of the use case or program, after all DSP-related 
 *       operations are complete.
 */
CV_EXPORTS void fcvdspdeinit();
//! @}

} // dsp::
} // fastcv::
} // cv::

#endif // OPENCV_FASTCV_DSP_INIT_HPP
