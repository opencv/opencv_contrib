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

// HIP implementation of cv::cuda::flip. On the NVIDIA build this operation is
// served by NPP's nppiMirror family; ROCm has no NPP, so reproduce the same
// per-element mirror as a direct kernel. The kernel moves whole pixels, so it
// is templated on the pixel size in bytes and covers every depth/channel
// combination NPP supports.
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
    template <typename T>
    __global__ void flipKernel(const uchar* src, size_t srcStep, uchar* dst, size_t dstStep,
                               int rows, int cols, bool flipRows, bool flipCols)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= cols || y >= rows)
            return;

        const int sy = flipRows ? (rows - 1 - y) : y;
        const int sx = flipCols ? (cols - 1 - x) : x;

        const T value = *reinterpret_cast<const T*>(src + sy * srcStep + sx * sizeof(T));
        *reinterpret_cast<T*>(dst + y * dstStep + x * sizeof(T)) = value;
    }

    template <typename T>
    void flipImpl(const GpuMat& src, GpuMat& dst, bool flipRows, bool flipCols, cudaStream_t stream)
    {
        const dim3 block(32, 8);
        const dim3 grid(divUp(src.cols, block.x), divUp(src.rows, block.y));
        flipKernel<T><<<grid, block, 0, stream>>>(src.data, src.step, dst.data, dst.step,
                                                  src.rows, src.cols, flipRows, flipCols);
        CV_CUDEV_SAFE_CALL(cudaGetLastError());
        if (stream == 0)
            CV_CUDEV_SAFE_CALL(cudaDeviceSynchronize());
    }
}

namespace cv { namespace cuda { namespace detail {

void flipHip(const GpuMat& src, GpuMat& dst, int flipCode, cudaStream_t stream)
{
    // flipCode == 0: mirror about the horizontal axis (reverse rows)
    // flipCode  > 0: mirror about the vertical axis (reverse columns)
    // flipCode  < 0: mirror about both axes
    const bool flipRows = (flipCode == 0) || (flipCode < 0);
    const bool flipCols = (flipCode > 0) || (flipCode < 0);

    typedef void (*func_t)(const GpuMat& src, GpuMat& dst, bool, bool, cudaStream_t);
    static const func_t funcs[] =
    {
        0,
        flipImpl<uchar>,    // 1 byte
        flipImpl<ushort>,   // 2 bytes
        flipImpl<uchar3>,   // 3 bytes
        flipImpl<uint>,     // 4 bytes
        0,
        flipImpl<ushort3>,  // 6 bytes
        0,
        flipImpl<uint2>,    // 8 bytes
        0, 0, 0,
        flipImpl<uint3>,    // 12 bytes
        0, 0, 0,
        flipImpl<uint4>     // 16 bytes
    };

    const size_t elemSize = src.elemSize();
    CV_Assert(elemSize <= 16 && funcs[elemSize] != 0);
    funcs[elemSize](src, dst, flipRows, flipCols, stream);
}

}}}

#endif
