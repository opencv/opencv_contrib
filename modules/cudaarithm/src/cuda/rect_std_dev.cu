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
// Copyright (c) 2026 Advanced Micro Devices, Inc.
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
// In no event shall the copyright holders or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

// HIP implementation of cv::cuda::rectStdDev. On the NVIDIA build this maps to
// nppiRectStdDev_32s32f_C1R, which has no ROCm analog. It computes, for each
// pixel, the standard deviation over a fixed rectangle read from the sum
// (32S integral) and squared-sum (64F integral) images:
//   stddev = sqrt(max(0, sqsum/area - (sum/area)^2))
// matching the NPP definition (the rectangle corners are sampled from the
// integral images and the result is the population deviation over the window).
//
// \author Jeff Daily <jeff.daily@amd.com>

#include "opencv2/opencv_modules.hpp"

#ifndef HAVE_OPENCV_CUDEV

#error "opencv_cudev is required"

#else

#include "opencv2/cudev.hpp"
#include "opencv2/core/private.cuda.hpp"

using namespace cv;
using namespace cv::cuda;
using namespace cv::cudev;

namespace
{
    __global__ void rectStdDevKernel(const GlobPtrSz<int> sum, const GlobPtr<double> sqr,
                                     GlobPtr<float> dst, int rows, int cols,
                                     int rx, int ry, int rw, int rh)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= cols || y >= rows)
            return;

        const int x1 = x + rx;
        const int y1 = y + ry;
        const int x2 = x1 + rw;
        const int y2 = y1 + rh;

        const int s = sum(y2, x2) - sum(y2, x1) - sum(y1, x2) + sum(y1, x1);
        const double sq = sqr(y2, x2) - sqr(y2, x1) - sqr(y1, x2) + sqr(y1, x1);

        const float area = static_cast<float>(rw) * rh;
        const float mean = s / area;
        float var = static_cast<float>(sq) / area - mean * mean;
        if (var < 0.0f)
            var = 0.0f;

        dst(y, x) = sqrtf(var);
    }
}

namespace cv { namespace cuda { namespace detail {

void rectStdDevHip(const GpuMat& src, const GpuMat& sqr, GpuMat& dst, Rect rect, cudaStream_t stream)
{
    const dim3 block(32, 8);
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));
    rectStdDevKernel<<<grid, block, 0, stream>>>(
        globPtr<int>(src), globPtr<double>(sqr), globPtr<float>(dst),
        dst.rows, dst.cols, rect.x, rect.y, rect.width, rect.height);
    CV_CUDEV_SAFE_CALL(cudaGetLastError());
}

}}}

#endif
