/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#ifndef OPENCV_FASTCV_ARITHM_HPP
#define OPENCV_FASTCV_ARITHM_HPP

#include <opencv2/core.hpp>

namespace cv {
namespace fastcv {

//! @addtogroup fastcv
//! @{

/**
 * @brief Matrix multiplication of two int8_t type matrices

 * @param src1 First source matrix of type CV_8S
 * @param src2 Second source matrix of type CV_8S
 * @param dst Resulting matrix of type CV_32S
 */
CV_EXPORTS_W void matmuls8s32(InputArray src1, InputArray src2, OutputArray dst);

//! @}

} // fastcv::
} // cv::

#endif // OPENCV_FASTCV_ARITHM_HPP
