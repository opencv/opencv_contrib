/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef __OPENCV_WAVELET_MATRIX_FEATURE_SUPPORT_CHECKS_H__
#define __OPENCV_WAVELET_MATRIX_FEATURE_SUPPORT_CHECKS_H__

#ifdef HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif // HAVE_CUDA


// The CUB library is used for the Median Filter with Wavelet Matrix,
// which has become a standard library since CUDA 11.
#if CUDA_VERSION >= 11000 || CUDART_VERSION >= 11000


// Check `if constexpr` is available.

// GCC has been supported since 7.1
#if defined(__GNUC__) && (__GNUC__ > 7 || (__GNUC__ == 7 && __GNUC_MINOR__ >= 1))
#define __OPENCV_USE_WAVELET_MATRIX_FOR_MEDIAN_FILTER_CUDA__
#endif

// clang has been supported since 5.0
#if defined(__clang__) && (__clang_major__ >= 5)
#define __OPENCV_USE_WAVELET_MATRIX_FOR_MEDIAN_FILTER_CUDA__
#endif


// Visual Studio has been supported since Visual Studio 2019 (16.1.2)
#if defined(_MSC_VER) && _MSC_VER >= 1921
#define __OPENCV_USE_WAVELET_MATRIX_FOR_MEDIAN_FILTER_CUDA__
#endif


// I confirmed that it works with Intel C++ Compiler 2021.1.2. It did not work with icc 19.0.1.
#if defined(__INTEL_COMPILER) && (__INTEL_COMPILER > 202101 || (__INTEL_COMPILER == 202101 && __INTEL_COMPILER_UPDATE >= 2))
#define __OPENCV_USE_WAVELET_MATRIX_FOR_MEDIAN_FILTER_CUDA__
#endif

#endif // CUDA_VERSION
#endif // __OPENCV_WAVELET_MATRIX_FEATURE_SUPPORT_CHECKS_H__
