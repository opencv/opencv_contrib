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


#endif // CUDA_VERSION
#endif // __OPENCV_WAVELET_MATRIX_FEATURE_SUPPORT_CHECKS_H__
