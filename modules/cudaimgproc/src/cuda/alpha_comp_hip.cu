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

// Direct HIP kernels replacing NPP's nppiAlphaComp_*_AC4R, which has no ROCm
// analogue. cv::cuda::alphaComp composites two 4-channel images with their
// alpha channels using the 13 Porter-Duff operators of AlphaCompTypes.
//
// Semantics reproduce NPP's nppiAlphaComp. With per-pixel source alphas
// as = a1/maxv and ad = a2/maxv (the 4th channel of each input), each operator
// selects Porter-Duff coverage factors (Fa, Fb):
//   OVER  Fa=1     Fb=1-as     IN   Fa=ad    Fb=0
//   OUT   Fa=1-ad  Fb=0        ATOP Fa=ad    Fb=1-as
//   XOR   Fa=1-ad  Fb=1-as     PLUS Fa=1     Fb=1
// The PREMUL operators reuse the same factor table; PREMUL by itself is the
// degenerate Fa=1, Fb=0 (return src1 premultiplied by its own alpha).
//
// For the non-premultiplied operators the inputs are straight-alpha, so the
// colour contribution of each source is weighted by its own alpha:
//   Cout = clamp(C1*as*Fa + C2*ad*Fb),  Aout = clamp(maxv*(as*Fa + ad*Fb))
// For the *_PREMUL operators the colours are treated as already premultiplied,
// so the own-alpha weighting is dropped:
//   Cout = clamp(C1*Fa + C2*Fb),        Aout = clamp(maxv*(as*Fa + ad*Fb))
// This matches nppiAlphaComp's published behaviour and the Porter-Duff algebra.
//
// \author Jeff Daily <jeff.daily@amd.com>

#if defined(__HIP_PLATFORM_AMD__) && !defined(CUDA_DISABLER)

#include <hip/hip_runtime.h>
#include "opencv2/core/cuda/common.hpp"
#include "opencv2/core/cuda/saturate_cast.hpp"

namespace cv { namespace cuda { namespace device { namespace alpha_hip
{
    // Mirrors AlphaCompTypes in opencv2/cudaimgproc.hpp.
    enum
    {
        ALPHA_OVER, ALPHA_IN, ALPHA_OUT, ALPHA_ATOP, ALPHA_XOR, ALPHA_PLUS,
        ALPHA_OVER_PREMUL, ALPHA_IN_PREMUL, ALPHA_OUT_PREMUL, ALPHA_ATOP_PREMUL,
        ALPHA_XOR_PREMUL, ALPHA_PLUS_PREMUL, ALPHA_PREMUL
    };

    __device__ __forceinline__ bool isPremulOp(int op)
    {
        return op >= ALPHA_OVER_PREMUL;
    }

    __device__ __forceinline__ void porterDuffFactors(int op, float as, float ad, float& Fa, float& Fb)
    {
        switch (op)
        {
            case ALPHA_OVER:  case ALPHA_OVER_PREMUL: Fa = 1.0f;      Fb = 1.0f - as; break;
            case ALPHA_IN:    case ALPHA_IN_PREMUL:   Fa = ad;        Fb = 0.0f;      break;
            case ALPHA_OUT:   case ALPHA_OUT_PREMUL:  Fa = 1.0f - ad; Fb = 0.0f;      break;
            case ALPHA_ATOP:  case ALPHA_ATOP_PREMUL: Fa = ad;        Fb = 1.0f - as; break;
            case ALPHA_XOR:   case ALPHA_XOR_PREMUL:  Fa = 1.0f - ad; Fb = 1.0f - as; break;
            case ALPHA_PLUS:  case ALPHA_PLUS_PREMUL: Fa = 1.0f;      Fb = 1.0f;      break;
            default: /* ALPHA_PREMUL */               Fa = 1.0f;      Fb = 0.0f;      break;
        }
    }

    template <typename T>
    __global__ void alphaCompKernel(PtrStepSz<T> img1, PtrStep<T> img2, PtrStep<T> dst, float maxv, int op)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= img1.cols || y >= img1.rows)
            return;

        const T* p1 = img1.ptr(y) + x * 4;
        const T* p2 = img2.ptr(y) + x * 4;
        T* d = dst.ptr(y) + x * 4;

        const float as = (float)p1[3] / maxv;
        const float ad = (float)p2[3] / maxv;

        float Fa, Fb;
        porterDuffFactors(op, as, ad, Fa, Fb);

        const bool premul = isPremulOp(op);
        const float w1 = premul ? Fa : as * Fa;
        const float w2 = premul ? Fb : ad * Fb;

        d[0] = saturate_cast<T>((float)p1[0] * w1 + (float)p2[0] * w2);
        d[1] = saturate_cast<T>((float)p1[1] * w1 + (float)p2[1] * w2);
        d[2] = saturate_cast<T>((float)p1[2] * w1 + (float)p2[2] * w2);
        d[3] = saturate_cast<T>(maxv * (as * Fa + ad * Fb));
    }

    template <typename T>
    void alphaCompImpl(PtrStepSz<T> img1, PtrStep<T> img2, PtrStep<T> dst, float maxv, int op, hipStream_t stream)
    {
        const dim3 block(32, 8);
        const dim3 grid(divUp(img1.cols, block.x), divUp(img1.rows, block.y));
        alphaCompKernel<T><<<grid, block, 0, stream>>>(img1, img2, dst, maxv, op);
        cudaSafeCall(hipGetLastError());
        if (stream == 0)
            cudaSafeCall(hipDeviceSynchronize());
    }

    // depth: CV_8U=0, CV_16U=2, CV_32S=4, CV_32F=5 (OpenCV depth codes).
    void alphaComp_gpu(PtrStepSzb img1, PtrStepSzb img2, PtrStepSzb dst, int depth, int op, hipStream_t stream)
    {
        switch (depth)
        {
            case 0: // CV_8U
            {
                PtrStepSz<uchar> a(img1.rows, img1.cols, (uchar*)img1.data, img1.step);
                PtrStep<uchar> b((uchar*)img2.data, img2.step);
                PtrStep<uchar> d((uchar*)dst.data, dst.step);
                alphaCompImpl<uchar>(a, b, d, 255.0f, op, stream);
                break;
            }
            case 2: // CV_16U
            {
                PtrStepSz<ushort> a(img1.rows, img1.cols, (ushort*)img1.data, img1.step);
                PtrStep<ushort> b((ushort*)img2.data, img2.step);
                PtrStep<ushort> d((ushort*)dst.data, dst.step);
                alphaCompImpl<ushort>(a, b, d, 65535.0f, op, stream);
                break;
            }
            case 4: // CV_32S
            {
                PtrStepSz<int> a(img1.rows, img1.cols, (int*)img1.data, img1.step);
                PtrStep<int> b((int*)img2.data, img2.step);
                PtrStep<int> d((int*)dst.data, dst.step);
                alphaCompImpl<int>(a, b, d, 2147483647.0f, op, stream);
                break;
            }
            default: // CV_32F
            {
                PtrStepSz<float> a(img1.rows, img1.cols, (float*)img1.data, img1.step);
                PtrStep<float> b((float*)img2.data, img2.step);
                PtrStep<float> d((float*)dst.data, dst.step);
                alphaCompImpl<float>(a, b, d, 1.0f, op, stream);
                break;
            }
        }
    }
}}}}

#endif // __HIP_PLATFORM_AMD__ && !CUDA_DISABLER
