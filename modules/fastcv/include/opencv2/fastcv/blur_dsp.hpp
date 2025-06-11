/*
 * Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#ifndef OPENCV_FASTCV_BLUR_DSP_HPP
#define OPENCV_FASTCV_BLUR_DSP_HPP

#include <opencv2/core.hpp>

namespace cv {
namespace fastcv {
namespace dsp {

//! @addtogroup fastcv
//! @{

/**
 * @brief Filter an image with non-separable kernel
 * @param _src Intput image with type CV_8UC1, src size should be greater than 176*144
 * @param _dst Output image with type CV_8UC1, CV_16SC1 or CV_32FC1
 * @param ddepth The depth of output image
 * @param _kernel Filer kernel data
 */
CV_EXPORTS void filter2D(InputArray _src, OutputArray _dst, int ddepth, InputArray _kernel);

//! @}

} // dsp::
} // fastcv::
} // cv::

#endif // OPENCV_FASTCV_BLUR_DSP_HPP
