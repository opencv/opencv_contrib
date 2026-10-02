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

// HIP implementations of the cudaarithm entry points that use NPP on the
// NVIDIA build: the per-channel constant left/right shift (nppiLShiftC /
// nppiRShiftC) and the single-input interleaved-complex magnitude
// (nppiMagnitude / nppiMagnitudeSqr). ROCm has no NPP, so these reproduce the
// same results directly. The right shift follows NPP's signedness: arithmetic
// for signed element types, logical for unsigned.
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
    enum { SHIFT_LEFT, SHIFT_RIGHT };

    template <typename T, int cn>
    __global__ void shiftKernel(const GlobPtrSz<typename MakeVec<T, cn>::type> src,
                                GlobPtr<typename MakeVec<T, cn>::type> dst,
                                int rows, int cols, uint s0, uint s1, uint s2, uint s3, int dir)
    {
        typedef typename MakeVec<T, cn>::type vec_type;
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= cols || y >= rows)
            return;

        vec_type sv = src(y, x);
        vec_type dv;
        T* dp = reinterpret_cast<T*>(&dv);
        const T* sp = reinterpret_cast<const T*>(&sv);
        const uint sh[4] = { s0, s1, s2, s3 };

        #pragma unroll
        for (int c = 0; c < cn; ++c)
            dp[c] = dir == SHIFT_LEFT ? static_cast<T>(sp[c] << sh[c]) : static_cast<T>(sp[c] >> sh[c]);

        dst(y, x) = dv;
    }

    template <typename T, int cn>
    void shiftImpl(const GpuMat& src, const uint sh[4], GpuMat& dst, int dir, cudaStream_t stream)
    {
        typedef typename MakeVec<T, cn>::type vec_type;
        const dim3 block(32, 8);
        const dim3 grid(divUp(src.cols, block.x), divUp(src.rows, block.y));
        shiftKernel<T, cn><<<grid, block, 0, stream>>>(
            globPtr<vec_type>(src), globPtr<vec_type>(dst), src.rows, src.cols,
            sh[0], sh[1], sh[2], sh[3], dir);
        CV_CUDEV_SAFE_CALL(cudaGetLastError());
        if (stream == 0)
            CV_CUDEV_SAFE_CALL(cudaDeviceSynchronize());
    }

    void dispatchShift(const GpuMat& src, const Scalar_<int>& val, GpuMat& dst, int dir, cudaStream_t stream)
    {
        const uint sh[4] = { (uint)val[0], (uint)val[1], (uint)val[2], (uint)val[3] };
        const int depth = src.depth();
        const int cn = src.channels();

        typedef void (*func_t)(const GpuMat&, const uint[4], GpuMat&, int, cudaStream_t);
        static const func_t funcs[5][4] =
        {
            { shiftImpl<uchar, 1> , 0, shiftImpl<uchar, 3> , shiftImpl<uchar, 4>  },
            { shiftImpl<schar, 1> , 0, shiftImpl<schar, 3> , shiftImpl<schar, 4>  },
            { shiftImpl<ushort, 1>, 0, shiftImpl<ushort, 3>, shiftImpl<ushort, 4> },
            { shiftImpl<short, 1> , 0, shiftImpl<short, 3> , shiftImpl<short, 4>  },
            { shiftImpl<int, 1>   , 0, shiftImpl<int, 3>   , shiftImpl<int, 4>    }
        };
        funcs[depth][cn - 1](src, sh, dst, dir, stream);
    }
}

namespace cv { namespace cuda { namespace detail {

void lshiftHip(const GpuMat& src, Scalar_<int> val, GpuMat& dst, cudaStream_t stream)
{
    dispatchShift(src, val, dst, SHIFT_LEFT, stream);
}

void rshiftHip(const GpuMat& src, Scalar_<int> val, GpuMat& dst, cudaStream_t stream)
{
    dispatchShift(src, val, dst, SHIFT_RIGHT, stream);
}

void magnitudeHip(const GpuMat& src, GpuMat& dst, Stream& stream)
{
    gridTransformUnary(globPtr<float2>(src), globPtr<float>(dst),
                       magnitude_interleaved_func<float2>(), stream);
}

void magnitudeSqrHip(const GpuMat& src, GpuMat& dst, Stream& stream)
{
    gridTransformUnary(globPtr<float2>(src), globPtr<float>(dst),
                       magnitude_sqr_interleaved_func<float2>(), stream);
}

}}}

#endif
