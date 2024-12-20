/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#ifndef OPENCV_FASTCV_FFT_HPP
#define OPENCV_FASTCV_FFT_HPP

#include <opencv2/core.hpp>

namespace cv {
namespace fastcv {

//! @addtogroup fastcv
//! @{

/**
 * @brief Computes the 1D or 2D Fast Fourier Transform of a real valued matrix.
          For the 2D case, the width and height of the input and output matrix must be powers of 2.
          For the 1D case, the height of the matrices must be 1, while the width must be a power of 2.
          Accepts 8-bit unsigned integer array, whereas cv::dft accepts floating-point or complex array.
 * @param src Input array of CV_8UC1. The dimensions of the matrix must be powers of 2 for the 2D case,
              and in the 1D case, the height must be 1, while the width must be a power of 2.
 * @param dst The computed FFT matrix of type CV_32FC2. The FFT Re and Im coefficients are stored in different channels.
              Hence the dimensions of the dst are (srcWidth, srcHeight)
 */
CV_EXPORTS_W void FFT(InputArray src, OutputArray dst);

/**
 * @brief Computes the 1D or 2D Inverse Fast Fourier Transform of a complex valued matrix.
          For the 2D case, The width and height of the input and output matrix must be powers of 2.
          For the 1D case, the height of the matrices must be 1, while the width must be a power of 2.

 * @param src Input array of type CV_32FC2 containing FFT Re and Im coefficients stored in separate channels.
              The dimensions of the matrix must be powers of 2 for the 2D case, and in the 1D case, the height must be 1,
              while the width must be a power of 2.
 * @param dst The computed IFFT matrix of type CV_8U. The matrix is real valued and has no imaginary components.
              Hence the dimensions of the dst are (srcWidth , srcHeight)
 */
CV_EXPORTS_W void IFFT(InputArray src, OutputArray dst);

//! @}

} // fastcv::
} // cv::

#endif // OPENCV_FASTCV_FFT_HPP
