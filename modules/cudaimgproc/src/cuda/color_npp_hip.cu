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

// Direct HIP kernels replacing the NPP entry points cudaimgproc's color.cpp
// relies on (NPP has no ROCm analogue): the 4-channel channel swap, sRGB-style
// gamma forward/inverse, and the alpha-premultiply used by COLOR_*2mRGBA.
//
// \author Jeff Daily <jeff.daily@amd.com>

#if defined(__HIP_PLATFORM_AMD__) && !defined(CUDA_DISABLER)

#include <hip/hip_runtime.h>
#include "opencv2/core/cuda/common.hpp"
#include "opencv2/core/cuda/saturate_cast.hpp"

namespace cv { namespace cuda { namespace device { namespace color_hip
{
    __global__ void swapChannels8uC4(PtrStepSz<uchar4> img, int4 order)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= img.cols || y >= img.rows)
            return;
        uchar4 p = img(y, x);
        const uchar c[4] = { p.x, p.y, p.z, p.w };
        img(y, x) = make_uchar4(c[order.x], c[order.y], c[order.z], c[order.w]);
    }

    void swapChannels_gpu(PtrStepSzb image, const int dstOrder[4], hipStream_t stream)
    {
        PtrStepSz<uchar4> img(image.rows, image.cols, (uchar4*)image.data, image.step);
        const dim3 block(32, 8);
        const dim3 grid(divUp(img.cols, block.x), divUp(img.rows, block.y));
        int4 order = make_int4(dstOrder[0], dstOrder[1], dstOrder[2], dstOrder[3]);
        swapChannels8uC4<<<grid, block, 0, stream>>>(img, order);
        cudaSafeCall(hipGetLastError());
        if (stream == 0)
            cudaSafeCall(hipDeviceSynchronize());
    }

    // NPP's gamma uses the standard sRGB transfer with the 0.055 offset and a
    // small linear segment near zero; forward = encode (linear->gamma), inverse
    // = decode. Applied per colour channel, leaving any 4th (alpha) untouched.
    __device__ __forceinline__ float gammaFwd(float c)
    {
        return c <= 0.018f ? 4.5f * c : 1.099f * powf(c, 1.0f / 2.4f) - 0.099f;
    }
    __device__ __forceinline__ float gammaInv(float c)
    {
        return c <= 0.081f ? c / 4.5f : powf((c + 0.099f) / 1.099f, 2.4f);
    }

    template <int CN, bool FWD>
    __global__ void gammaKernel(PtrStepSz<uchar> img, int width)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= img.rows)
            return;
        uchar* p = img.ptr(y) + x * CN;
        for (int c = 0; c < 3; ++c)
        {
            float v = p[c] / 255.0f;
            v = FWD ? gammaFwd(v) : gammaInv(v);
            p[c] = saturate_cast<uchar>(v * 255.0f);
        }
    }

    void gamma_gpu(PtrStepSzb img, int cn, bool forward, hipStream_t stream)
    {
        const dim3 block(32, 8);
        const dim3 grid(divUp(img.cols, block.x), divUp(img.rows, block.y));
        if (cn == 3)
            forward ? gammaKernel<3, true><<<grid, block, 0, stream>>>(img, img.cols)
                    : gammaKernel<3, false><<<grid, block, 0, stream>>>(img, img.cols);
        else
            forward ? gammaKernel<4, true><<<grid, block, 0, stream>>>(img, img.cols)
                    : gammaKernel<4, false><<<grid, block, 0, stream>>>(img, img.cols);
        cudaSafeCall(hipGetLastError());
        if (stream == 0)
            cudaSafeCall(hipDeviceSynchronize());
    }

    template <typename T>
    __global__ void premulKernel(PtrStepSz<T> src, PtrStep<T> dst, float maxv)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= src.cols || y >= src.rows)
            return;
        const T* s = src.ptr(y) + x * 4;
        T* d = dst.ptr(y) + x * 4;
        const float a = s[3] / maxv;
        d[0] = saturate_cast<T>(s[0] * a);
        d[1] = saturate_cast<T>(s[1] * a);
        d[2] = saturate_cast<T>(s[2] * a);
        d[3] = s[3];
    }

    template <typename T>
    void premulImpl(PtrStepSz<T> src, PtrStep<T> dst, float maxv, hipStream_t stream)
    {
        const dim3 block(32, 8);
        const dim3 grid(divUp(src.cols, block.x), divUp(src.rows, block.y));
        premulKernel<T><<<grid, block, 0, stream>>>(src, dst, maxv);
        cudaSafeCall(hipGetLastError());
        if (stream == 0)
            cudaSafeCall(hipDeviceSynchronize());
    }

    void alphaPremul_gpu(PtrStepSzb src, PtrStepSzb dst, int depth, hipStream_t stream)
    {
        if (depth == 0) // CV_8U
        {
            PtrStepSz<uchar> s(src.rows, src.cols, (uchar*)src.data, src.step);
            PtrStep<uchar> d((uchar*)dst.data, dst.step);
            premulImpl<uchar>(s, d, 255.0f, stream);
        }
        else // CV_16U
        {
            PtrStepSz<ushort> s(src.rows, src.cols, (ushort*)src.data, src.step);
            PtrStep<ushort> d((ushort*)dst.data, dst.step);
            premulImpl<ushort>(s, d, 65535.0f, stream);
        }
    }
}}}}

#endif // __HIP_PLATFORM_AMD__ && !CUDA_DISABLER
