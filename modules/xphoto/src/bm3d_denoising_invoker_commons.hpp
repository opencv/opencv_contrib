/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective icvers.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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

#ifndef __OPENCV_BM3D_DENOISING_INVOKER_COMMONS_HPP__
#define __OPENCV_BM3D_DENOISING_INVOKER_COMMONS_HPP__

#include "bm3d_denoising_invoker_structs.hpp"

// std::isnan is a part of C++11 and it is not supported in MSVS2010/2012
#if defined _MSC_VER && _MSC_VER < 1800 /* MSVC 2013 */
#include <float.h>
namespace std {
    template <typename T> bool isnan(T value) { return _isnan(value) != 0; }
}
#endif

namespace cv
{
namespace xphoto
{

// Returns largest power of 2 smaller than the input value
inline int getLargestPowerOf2SmallerThan(unsigned x)
{
    x = x | (x >> 1);
    x = x | (x >> 2);
    x = x | (x >> 4);
    x = x | (x >> 8);
    x = x | (x >> 16);
    return x - (x >> 1);
}

// Returns true if x is a power of 2. Otherwise false.
inline bool isPowerOf2(int x)
{
    return (x > 0) && !(x & (x - 1));
}


template <typename T>
inline static void shrink(T &val, T &nonZeroCount, const T &threshold)
{
    if (std::abs(val) < threshold)
        val = 0;
    else
        ++nonZeroCount;
}

template <typename T>
inline static void hardThreshold2D(T *dst, T *thrMap, const int &templateWindowSizeSq)
{
    for (int i = 1; i < templateWindowSizeSq; ++i)
    {
        if (std::abs(dst[i] < thrMap[i]))
            dst[i] = 0;
    }
}

template <int N, typename T, typename DT, typename CT>
inline static T HardThreshold(BlockMatch<T, DT, CT> *z, const int &n, T *&thrMap)
{
    T nonZeroCount = 0;

    for (int i = 0; i < N; ++i)
        shrink(z[i][n], nonZeroCount, *thrMap++);

    return nonZeroCount;
}

template <typename T, typename DT, typename CT>
inline static T HardThreshold(BlockMatch<T, DT, CT> *z, const int &n, T *&thrMap, const int &N)
{
    T nonZeroCount = 0;

    for (int i = 0; i < N; ++i)
        shrink(z[i][n], nonZeroCount, *thrMap++);

    return nonZeroCount;
}

template <int N, typename T, typename DT, typename CT>
inline static int WienerFiltering(BlockMatch<T, DT, CT> *zSrc, BlockMatch<T, DT, CT> *zBasic, const int &n, T *&thrMap)
{
    int wienerCoeffs = 0;

    for (int i = 0; i < N; ++i)
    {
        // Possible optimization point here to get rid of floats and casts
        int basicSq = zBasic[i][n] * zBasic[i][n];
        int sigmaSq = *thrMap * *thrMap;
        int denom = basicSq + sigmaSq;
        float wie = (denom == 0) ? 1.0f : ((float)basicSq / (float)denom);

        zBasic[i][n] = (T)(zSrc[i][n] * wie);
        wienerCoeffs += (int)wie;
        ++thrMap;
    }

    return wienerCoeffs;
}

template <typename T, typename DT, typename CT>
inline static int WienerFiltering(BlockMatch<T, DT, CT> *zSrc, BlockMatch<T, DT, CT> *zBasic, const int &n, T *&thrMap, const unsigned &N)
{
    int wienerCoeffs = 0;

    for (unsigned i = 0; i < N; ++i)
    {
        // Possible optimization point here to get rid of floats and casts
        int basicSq = zBasic[i][n] * zBasic[i][n];
        int sigmaSq = *thrMap * *thrMap;
        int denom = basicSq + sigmaSq;
        float wie = (denom == 0) ? 1.0f : ((float)basicSq / (float)denom);

        zBasic[i][n] = (T)(zSrc[i][n] * wie);
        wienerCoeffs += (int)wie;
        ++thrMap;
    }

    return wienerCoeffs;
}


}  // namespace xphoto
}  // namespace cv

#endif
