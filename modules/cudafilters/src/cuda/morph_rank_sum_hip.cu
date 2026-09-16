/*M///////////////////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.
//
//M*/

// Direct HIP kernels that replace the NPP entry points cudafilters relies on
// (NPP has no ROCm analogue): morphological erode/dilate with an arbitrary
// structuring element, the box max/min rank filters, and the row/column sliding
// window sums. Every kernel reads from a source view whose surrounding border
// pixels have already been materialised by cuda::copyMakeBorder (matching the
// NPP call sites), so the gather window never needs its own border handling.
//
// \author Jeff Daily <jeff.daily@amd.com>

#if defined(__HIP_PLATFORM_AMD__) && !defined(CUDA_DISABLER)

#include <hip/hip_runtime.h>
#include "opencv2/core/cuda/common.hpp"

namespace cv { namespace cuda { namespace device { namespace filter_hip
{
    // ---- morphology (erode = min, dilate = max over the structuring element) ----

    template <typename T, int CN, bool IS_MAX>
    __global__ void morphKernel(const PtrStepSz<T> src, PtrStep<T> dst,
                                const uchar* __restrict__ se, int seW, int seH,
                                int anchorX, int anchorY)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= src.cols || y >= src.rows)
            return;

        T acc[CN];
        bool hasVal = false;
        for (int j = 0; j < seH; ++j)
        {
            const T* srow = src.ptr(y - anchorY + j);
            for (int i = 0; i < seW; ++i)
            {
                if (se[j * seW + i] == 0)
                    continue;
                const T* sval = srow + (x - anchorX + i) * CN;
                if (!hasVal)
                {
                    for (int c = 0; c < CN; ++c)
                        acc[c] = sval[c];
                    hasVal = true;
                }
                else
                {
                    for (int c = 0; c < CN; ++c)
                        acc[c] = IS_MAX ? max(acc[c], sval[c]) : min(acc[c], sval[c]);
                }
            }
        }

        T* drow = dst.ptr(y) + x * CN;
        for (int c = 0; c < CN; ++c)
            drow[c] = acc[c];
    }

    template <typename T, int CN, bool IS_MAX>
    void morphImpl(PtrStepSz<T> src, PtrStep<T> dst, const uchar* se, int seW, int seH,
                   int anchorX, int anchorY, hipStream_t stream)
    {
        const dim3 block(32, 8);
        const dim3 grid(divUp(src.cols, block.x), divUp(src.rows, block.y));
        morphKernel<T, CN, IS_MAX><<<grid, block, 0, stream>>>(src, dst, se, seW, seH, anchorX, anchorY);
        cudaSafeCall(hipGetLastError());
        if (stream == 0)
            cudaSafeCall(hipDeviceSynchronize());
    }

    void morph8u(const PtrStepSzb src, PtrStepSzb dst, int cn, const uchar* se, int seW, int seH,
                 int anchorX, int anchorY, bool isMax, hipStream_t stream)
    {
        // GpuMat::cols is the pixel column count regardless of channels; the
        // kernels iterate over pixels and index channels via CN.
        PtrStepSz<uchar> s(src.rows, src.cols, (uchar*)src.data, src.step);
        PtrStep<uchar> d((uchar*)dst.data, dst.step);
        if (cn == 1)
            isMax ? morphImpl<uchar, 1, true>(s, d, se, seW, seH, anchorX, anchorY, stream)
                  : morphImpl<uchar, 1, false>(s, d, se, seW, seH, anchorX, anchorY, stream);
        else
            isMax ? morphImpl<uchar, 4, true>(s, d, se, seW, seH, anchorX, anchorY, stream)
                  : morphImpl<uchar, 4, false>(s, d, se, seW, seH, anchorX, anchorY, stream);
    }

    void morph32f(const PtrStepSzf src, PtrStepSzf dst, int cn, const uchar* se, int seW, int seH,
                  int anchorX, int anchorY, bool isMax, hipStream_t stream)
    {
        PtrStepSz<float> s(src.rows, src.cols, (float*)src.data, src.step);
        PtrStep<float> d((float*)dst.data, dst.step);
        if (cn == 1)
            isMax ? morphImpl<float, 1, true>(s, d, se, seW, seH, anchorX, anchorY, stream)
                  : morphImpl<float, 1, false>(s, d, se, seW, seH, anchorX, anchorY, stream);
        else
            isMax ? morphImpl<float, 4, true>(s, d, se, seW, seH, anchorX, anchorY, stream)
                  : morphImpl<float, 4, false>(s, d, se, seW, seH, anchorX, anchorY, stream);
    }

    // ---- box rank filter (full rectangular window, max or min) ----

    template <int CN, bool IS_MAX>
    __global__ void rankKernel(const PtrStepSz<uchar> src, PtrStep<uchar> dst,
                               int kW, int kH, int anchorX, int anchorY)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= src.cols || y >= src.rows)
            return;

        uchar acc[CN];
        for (int c = 0; c < CN; ++c)
            acc[c] = IS_MAX ? 0 : 255;
        for (int j = 0; j < kH; ++j)
        {
            const uchar* srow = src.ptr(y - anchorY + j);
            for (int i = 0; i < kW; ++i)
            {
                const uchar* sval = srow + (x - anchorX + i) * CN;
                for (int c = 0; c < CN; ++c)
                    acc[c] = IS_MAX ? max(acc[c], sval[c]) : min(acc[c], sval[c]);
            }
        }
        uchar* drow = dst.ptr(y) + x * CN;
        for (int c = 0; c < CN; ++c)
            drow[c] = acc[c];
    }

    template <int CN, bool IS_MAX>
    void rankImpl(PtrStepSz<uchar> src, PtrStep<uchar> dst, int kW, int kH,
                  int anchorX, int anchorY, hipStream_t stream)
    {
        const dim3 block(32, 8);
        const dim3 grid(divUp(src.cols, block.x), divUp(src.rows, block.y));
        rankKernel<CN, IS_MAX><<<grid, block, 0, stream>>>(src, dst, kW, kH, anchorX, anchorY);
        cudaSafeCall(hipGetLastError());
        if (stream == 0)
            cudaSafeCall(hipDeviceSynchronize());
    }

    void rank8u(const PtrStepSzb src, PtrStepSzb dst, int cn, int kW, int kH,
                int anchorX, int anchorY, bool isMax, hipStream_t stream)
    {
        PtrStepSz<uchar> s(src.rows, src.cols, (uchar*)src.data, src.step);
        PtrStep<uchar> d((uchar*)dst.data, dst.step);
        if (cn == 1)
            isMax ? rankImpl<1, true>(s, d, kW, kH, anchorX, anchorY, stream)
                  : rankImpl<1, false>(s, d, kW, kH, anchorX, anchorY, stream);
        else
            isMax ? rankImpl<4, true>(s, d, kW, kH, anchorX, anchorY, stream)
                  : rankImpl<4, false>(s, d, kW, kH, anchorX, anchorY, stream);
    }

    // ---- sliding window sums (8u -> 32f), row and column ----

    __global__ void rowSumKernel(const PtrStepSz<uchar> src, PtrStepf dst, int ksize, int anchor)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= src.cols || y >= src.rows)
            return;
        const uchar* srow = src.ptr(y);
        float s = 0.f;
        for (int i = 0; i < ksize; ++i)
            s += srow[x - anchor + i];
        dst.ptr(y)[x] = s;
    }

    __global__ void columnSumKernel(const PtrStepSz<uchar> src, PtrStepf dst, int ksize, int anchor)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= src.cols || y >= src.rows)
            return;
        float s = 0.f;
        for (int j = 0; j < ksize; ++j)
            s += src.ptr(y - anchor + j)[x];
        dst.ptr(y)[x] = s;
    }

    void rowSum8u32f(const PtrStepSzb src, PtrStepSzf dst, int ksize, int anchor, hipStream_t stream)
    {
        const dim3 block(32, 8);
        const dim3 grid(divUp(src.cols, block.x), divUp(src.rows, block.y));
        rowSumKernel<<<grid, block, 0, stream>>>(src, dst, ksize, anchor);
        cudaSafeCall(hipGetLastError());
        if (stream == 0)
            cudaSafeCall(hipDeviceSynchronize());
    }

    void columnSum8u32f(const PtrStepSzb src, PtrStepSzf dst, int ksize, int anchor, hipStream_t stream)
    {
        const dim3 block(32, 8);
        const dim3 grid(divUp(src.cols, block.x), divUp(src.rows, block.y));
        columnSumKernel<<<grid, block, 0, stream>>>(src, dst, ksize, anchor);
        cudaSafeCall(hipGetLastError());
        if (stream == 0)
            cudaSafeCall(hipDeviceSynchronize());
    }
}}}}

#endif // __HIP_PLATFORM_AMD__ && !CUDA_DISABLER
