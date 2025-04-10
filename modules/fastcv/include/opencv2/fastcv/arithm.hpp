/*
 * Copyright (c) 2024-2025 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#ifndef OPENCV_FASTCV_ARITHM_HPP
#define OPENCV_FASTCV_ARITHM_HPP

#include <opencv2/core.hpp>

#define FCV_CMP_EQ(val1,val2) (fabs(val1 - val2) < FLT_EPSILON)

#define FCV_OPTYPE(depth,op) ((depth<<3) + op)

namespace cv {
namespace fastcv {

//! @addtogroup fastcv
//! @{

/**
 * @brief Matrix multiplication of two int8_t type matrices
 *		  uses signed integer input/output whereas cv::gemm uses floating point input/output
 *        matmuls8s32 provides enhanced speed on Qualcomm's processors
 * @param src1 First source matrix of type CV_8S
 * @param src2 Second source matrix of type CV_8S
 * @param dst Resulting matrix of type CV_32S
 */
CV_EXPORTS_W void matmuls8s32(InputArray src1, InputArray src2, OutputArray dst);

//! @}

//! @addtogroup fastcv
//! @{

/**
 * @brief Arithmetic add and subtract operations for two matrices
 *        It is optimized for Qualcomm's processors
 * @param src1 First source matrix, can be of type CV_8U, CV_16S, CV_32F.
 *             Note: CV_32F not supported for subtract
 * @param src2 Second source matrix of same type and size as src1
 * @param dst Resulting matrix of type as src mats
 * @param op  type of operation - 0 for add and 1 for subtract
 */
CV_EXPORTS_W void arithmetic_op(InputArray src1, InputArray src2, OutputArray dst, int op);

//! @}

//! @addtogroup fastcv
//! @{

/**
 * @brief Matrix multiplication of two float type matrices
 *        R = a*A*B + b*C where A,B,C,R are matrices and a,b are constants
 *        It is optimized for Qualcomm's processors
 * @param src1 First source matrix of type CV_32F
 * @param src2 Second source matrix of type CV_32F with same rows as src1 cols
 * @param dst Resulting matrix of type CV_32F
 * @param alpha multiplying factor for src1 and src2
 * @param src3 Optional third matrix of type CV_32F to be added to matrix product
 * @param beta multiplying factor for src3
 */
CV_EXPORTS_W void gemm(InputArray src1, InputArray src2, OutputArray dst, float alpha = 1.0,
                           InputArray src3 = noArray(), float beta = 0.0);

//! @}

//! @addtogroup fastcv
//! @{

/**
 * @brief Integral of a YCbCr420 image.
 *        Note: Input height should be multiple of 2. Input width and stride should be multiple of 16.
 *              Output stride should be multiple of 8.
 *              It is optimized for Qualcomm's processors
 * @param Y Input Y component of 8UC1 YCbCr420 image.
 * @param CbCr Input CbCr component(interleaved) of 8UC1 YCbCr420 image.
 * @param IY Output Y integral of CV_32S one channel, size (Y height + 1)*(Y width + 1)
 * @param ICb Output Cb integral of CV_32S one channel, size (Y height/2 + 1)*(Y width/2 + 1)
 * @param ICr Output Cr integral of CV_32S one channel, size (Y height/2 + 1)*(Y width/2 + 1)
 */
CV_EXPORTS_W void integrateYUV(InputArray Y, InputArray CbCr, OutputArray IY, OutputArray ICb, OutputArray ICr);

//! @}

} // fastcv::
} // cv::

#endif // OPENCV_FASTCV_ARITHM_HPP
