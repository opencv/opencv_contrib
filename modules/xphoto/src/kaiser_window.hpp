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

#ifndef __OPENCV_BM3D_DENOISING_KAISER_WINDOW_HPP__
#define __OPENCV_BM3D_DENOISING_KAISER_WINDOW_HPP__

#include "opencv2/core.hpp"
#include <cmath>

namespace cv
{
namespace xphoto
{

static int factorial(int n)
{
    if (n == 0)
        return 1;

    int val = 1;
    for (int idx = 1; idx <= n; ++idx)
        val *= idx;

    return val;
}

template <int MAX_ITER>
static float bessel0(const float &x)
{
    float sum = 0.0f;

    for (int m = 0; m < MAX_ITER; ++m)
    {
        float factM = (float)factorial(m);
        float inc = std::pow(1.0f / factM * std::pow(x * 0.5f, (float)m), 2.0f);
        sum += inc;

        if ((inc / sum) < 0.001F)
            break;
    }

    return sum;
}

#define MAX_ITER_BESSEL 100

static void calcKaiserWindow1D(cv::Mat &dst, const int N, const float beta)
{
    if (dst.empty())
        dst.create(cv::Size(1, N), CV_32FC1);

    CV_Assert(dst.total() == (size_t)N);
    CV_Assert(dst.type() == CV_32FC1);
    CV_Assert(N > 0);

    float *p = dst.ptr<float>(0);
    for (int i = 0; i < N; ++i)
    {
        float b = beta * std::sqrt(1.0f - std::pow(2.0f * i / (N - 1.0f) - 1.0f, 2.0f));
        p[i] = bessel0<MAX_ITER_BESSEL>(b) / bessel0<MAX_ITER_BESSEL>(beta);
    }
}

static void calcKaiserWindow2D(float *&kaiser, const int N, const float beta)
{
    if (kaiser == NULL)
        kaiser = new float[N * N];

    if (beta == 0.0f)
    {
        for (int i = 0; i < N * N; ++i)
            kaiser[i] = 1.0f;
        return;
    }

    cv::Mat kaiser1D;
    calcKaiserWindow1D(kaiser1D, N, beta);

    cv::Mat kaiser1Dt;
    cv::transpose(kaiser1D, kaiser1Dt);

    cv::Mat kaiser2D = kaiser1D * kaiser1Dt;
    float *p = kaiser2D.ptr<float>(0);
    for (unsigned i = 0; i < kaiser2D.total(); ++i)
        kaiser[i] = p[i];
}

}  // namespace xphoto
}  // namespace cv

#endif