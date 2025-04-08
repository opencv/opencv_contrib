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

#if !defined CUDA_DISABLER

#include "opencv2/core/cuda/common.hpp"
#include "opencv2/core/cuda/vec_traits.hpp"
#include "opencv2/core/cuda/vec_math.hpp"
#include "opencv2/core/cuda/limits.hpp"
#include "opencv2/core/cuda/color.hpp"
#include "opencv2/core/cuda/saturate_cast.hpp"
#include "opencv2/cudev/ptr2d/texture.hpp"

namespace cv { namespace cuda { namespace device
{
    template <typename T> struct Bayer2BGR;

    template <> struct Bayer2BGR<uchar>
    {
        uchar3 res0;
        uchar3 res1;
        uchar3 res2;
        uchar3 res3;

        __device__ void apply(const PtrStepSzb& src, int s_x, int s_y, bool blue_last, bool start_with_green)
        {
            uchar4 patch[3][3];
            patch[0][1] = ((const uchar4*) src.ptr(s_y - 1))[s_x];
            patch[0][0] = ((const uchar4*) src.ptr(s_y - 1))[::max(s_x - 1, 0)];
            patch[0][2] = ((const uchar4*) src.ptr(s_y - 1))[::min(s_x + 1, ((src.cols + 3) >> 2) - 1)];

            patch[1][1] = ((const uchar4*) src.ptr(s_y))[s_x];
            patch[1][0] = ((const uchar4*) src.ptr(s_y))[::max(s_x - 1, 0)];
            patch[1][2] = ((const uchar4*) src.ptr(s_y))[::min(s_x + 1, ((src.cols + 3) >> 2) - 1)];

            patch[2][1] = ((const uchar4*) src.ptr(s_y + 1))[s_x];
            patch[2][0] = ((const uchar4*) src.ptr(s_y + 1))[::max(s_x - 1, 0)];
            patch[2][2] = ((const uchar4*) src.ptr(s_y + 1))[::min(s_x + 1, ((src.cols + 3) >> 2) - 1)];

            if ((s_y & 1) ^ start_with_green)
            {
                const int t0 = (patch[0][1].x + patch[2][1].x + 1) >> 1;
                const int t1 = (patch[1][0].w + patch[1][1].y + 1) >> 1;

                const int t2 = (patch[0][1].x + patch[0][1].z + patch[2][1].x + patch[2][1].z + 2) >> 2;
                const int t3 = (patch[0][1].y + patch[1][1].x + patch[1][1].z + patch[2][1].y + 2) >> 2;

                const int t4 = (patch[0][1].z + patch[2][1].z + 1) >> 1;
                const int t5 = (patch[1][1].y + patch[1][1].w + 1) >> 1;

                const int t6 = (patch[0][1].z + patch[0][2].x + patch[2][1].z + patch[2][2].x + 2) >> 2;
                const int t7 = (patch[0][1].w + patch[1][1].z + patch[1][2].x + patch[2][1].w + 2) >> 2;

                if ((s_y & 1) ^ blue_last)
                {
                    res0.x = t1;
                    res0.y = patch[1][1].x;
                    res0.z = t0;

                    res1.x = patch[1][1].y;
                    res1.y = t3;
                    res1.z = t2;

                    res2.x = t5;
                    res2.y = patch[1][1].z;
                    res2.z = t4;

                    res3.x = patch[1][1].w;
                    res3.y = t7;
                    res3.z = t6;
                }
                else
                {
                    res0.x = t0;
                    res0.y = patch[1][1].x;
                    res0.z = t1;

                    res1.x = t2;
                    res1.y = t3;
                    res1.z = patch[1][1].y;

                    res2.x = t4;
                    res2.y = patch[1][1].z;
                    res2.z = t5;

                    res3.x = t6;
                    res3.y = t7;
                    res3.z = patch[1][1].w;
                }
            }
            else
            {
                const int t0 = (patch[0][0].w + patch[0][1].y + patch[2][0].w + patch[2][1].y + 2) >> 2;
                const int t1 = (patch[0][1].x + patch[1][0].w + patch[1][1].y + patch[2][1].x + 2) >> 2;

                const int t2 = (patch[0][1].y + patch[2][1].y + 1) >> 1;
                const int t3 = (patch[1][1].x + patch[1][1].z + 1) >> 1;

                const int t4 = (patch[0][1].y + patch[0][1].w + patch[2][1].y + patch[2][1].w + 2) >> 2;
                const int t5 = (patch[0][1].z + patch[1][1].y + patch[1][1].w + patch[2][1].z + 2) >> 2;

                const int t6 = (patch[0][1].w + patch[2][1].w + 1) >> 1;
                const int t7 = (patch[1][1].z + patch[1][2].x + 1) >> 1;

                if ((s_y & 1) ^ blue_last)
                {
                    res0.x = patch[1][1].x;
                    res0.y = t1;
                    res0.z = t0;

                    res1.x = t3;
                    res1.y = patch[1][1].y;
                    res1.z = t2;

                    res2.x = patch[1][1].z;
                    res2.y = t5;
                    res2.z = t4;

                    res3.x = t7;
                    res3.y = patch[1][1].w;
                    res3.z = t6;
                }
                else
                {
                    res0.x = t0;
                    res0.y = t1;
                    res0.z = patch[1][1].x;

                    res1.x = t2;
                    res1.y = patch[1][1].y;
                    res1.z = t3;

                    res2.x = t4;
                    res2.y = t5;
                    res2.z = patch[1][1].z;

                    res3.x = t6;
                    res3.y = patch[1][1].w;
                    res3.z = t7;
                }
            }
        }
    };

    template <typename D> __device__ __forceinline__ D toDst(const uchar3& pix);
    template <> __device__ __forceinline__ uchar toDst<uchar>(const uchar3& pix)
    {
        typename bgr_to_gray_traits<uchar>::functor_type f = bgr_to_gray_traits<uchar>::create_functor();
        return f(pix);
    }
    template <> __device__ __forceinline__ uchar3 toDst<uchar3>(const uchar3& pix)
    {
        return pix;
    }
    template <> __device__ __forceinline__ uchar4 toDst<uchar4>(const uchar3& pix)
    {
        return make_uchar4(pix.x, pix.y, pix.z, 255);
    }

    template <typename D>
    __global__ void Bayer2BGR_8u(const PtrStepSzb src, PtrStep<D> dst, const bool blue_last, const bool start_with_green)
    {
        const int s_x = blockIdx.x * blockDim.x + threadIdx.x;
        int s_y = blockIdx.y * blockDim.y + threadIdx.y;

        if (s_y >= src.rows || (s_x << 2) >= src.cols)
            return;

        s_y = ::min(::max(s_y, 1), src.rows - 2);

        Bayer2BGR<uchar> bayer;
        bayer.apply(src, s_x, s_y, blue_last, start_with_green);

        const int d_x = (blockIdx.x * blockDim.x + threadIdx.x) << 2;
        const int d_y = blockIdx.y * blockDim.y + threadIdx.y;

        dst(d_y, d_x) = toDst<D>(bayer.res0);
        if (d_x + 1 < src.cols)
            dst(d_y, d_x + 1) = toDst<D>(bayer.res1);
        if (d_x + 2 < src.cols)
            dst(d_y, d_x + 2) = toDst<D>(bayer.res2);
        if (d_x + 3 < src.cols)
            dst(d_y, d_x + 3) = toDst<D>(bayer.res3);
    }

    template <> struct Bayer2BGR<ushort>
    {
        ushort3 res0;
        ushort3 res1;

        __device__ void apply(const PtrStepSzb& src, int s_x, int s_y, bool blue_last, bool start_with_green)
        {
            ushort2 patch[3][3];
            patch[0][1] = ((const ushort2*) src.ptr(s_y - 1))[s_x];
            patch[0][0] = ((const ushort2*) src.ptr(s_y - 1))[::max(s_x - 1, 0)];
            patch[0][2] = ((const ushort2*) src.ptr(s_y - 1))[::min(s_x + 1, ((src.cols + 1) >> 1) - 1)];

            patch[1][1] = ((const ushort2*) src.ptr(s_y))[s_x];
            patch[1][0] = ((const ushort2*) src.ptr(s_y))[::max(s_x - 1, 0)];
            patch[1][2] = ((const ushort2*) src.ptr(s_y))[::min(s_x + 1, ((src.cols + 1) >> 1) - 1)];

            patch[2][1] = ((const ushort2*) src.ptr(s_y + 1))[s_x];
            patch[2][0] = ((const ushort2*) src.ptr(s_y + 1))[::max(s_x - 1, 0)];
            patch[2][2] = ((const ushort2*) src.ptr(s_y + 1))[::min(s_x + 1, ((src.cols + 1) >> 1) - 1)];

            if ((s_y & 1) ^ start_with_green)
            {
                const int t0 = (patch[0][1].x + patch[2][1].x + 1) >> 1;
                const int t1 = (patch[1][0].y + patch[1][1].y + 1) >> 1;

                const int t2 = (patch[0][1].x + patch[0][2].x + patch[2][1].x + patch[2][2].x + 2) >> 2;
                const int t3 = (patch[0][1].y + patch[1][1].x + patch[1][2].x + patch[2][1].y + 2) >> 2;

                if ((s_y & 1) ^ blue_last)
                {
                    res0.x = t1;
                    res0.y = patch[1][1].x;
                    res0.z = t0;

                    res1.x = patch[1][1].y;
                    res1.y = t3;
                    res1.z = t2;
                }
                else
                {
                    res0.x = t0;
                    res0.y = patch[1][1].x;
                    res0.z = t1;

                    res1.x = t2;
                    res1.y = t3;
                    res1.z = patch[1][1].y;
                }
            }
            else
            {
                const int t0 = (patch[0][0].y + patch[0][1].y + patch[2][0].y + patch[2][1].y + 2) >> 2;
                const int t1 = (patch[0][1].x + patch[1][0].y + patch[1][1].y + patch[2][1].x + 2) >> 2;

                const int t2 = (patch[0][1].y + patch[2][1].y + 1) >> 1;
                const int t3 = (patch[1][1].x + patch[1][2].x + 1) >> 1;

                if ((s_y & 1) ^ blue_last)
                {
                    res0.x = patch[1][1].x;
                    res0.y = t1;
                    res0.z = t0;

                    res1.x = t3;
                    res1.y = patch[1][1].y;
                    res1.z = t2;
                }
                else
                {
                    res0.x = t0;
                    res0.y = t1;
                    res0.z = patch[1][1].x;

                    res1.x = t2;
                    res1.y = patch[1][1].y;
                    res1.z = t3;
                }
            }
        }
    };

    template <typename D> __device__ __forceinline__ D toDst(const ushort3& pix);
    template <> __device__ __forceinline__ ushort toDst<ushort>(const ushort3& pix)
    {
        typename bgr_to_gray_traits<ushort>::functor_type f = bgr_to_gray_traits<ushort>::create_functor();
        return f(pix);
    }
    template <> __device__ __forceinline__ ushort3 toDst<ushort3>(const ushort3& pix)
    {
        return pix;
    }
    template <> __device__ __forceinline__ ushort4 toDst<ushort4>(const ushort3& pix)
    {
        return make_ushort4(pix.x, pix.y, pix.z, numeric_limits<ushort>::max());
    }

    template <typename D>
    __global__ void Bayer2BGR_16u(const PtrStepSzb src, PtrStep<D> dst, const bool blue_last, const bool start_with_green)
    {
        const int s_x = blockIdx.x * blockDim.x + threadIdx.x;
        int s_y = blockIdx.y * blockDim.y + threadIdx.y;

        if (s_y >= src.rows || (s_x << 1) >= src.cols)
            return;

        s_y = ::min(::max(s_y, 1), src.rows - 2);

        Bayer2BGR<ushort> bayer;
        bayer.apply(src, s_x, s_y, blue_last, start_with_green);

        const int d_x = (blockIdx.x * blockDim.x + threadIdx.x) << 1;
        const int d_y = blockIdx.y * blockDim.y + threadIdx.y;

        dst(d_y, d_x) = toDst<D>(bayer.res0);
        if (d_x + 1 < src.cols)
            dst(d_y, d_x + 1) = toDst<D>(bayer.res1);
    }

    template <int cn>
    void Bayer2BGR_8u_gpu(PtrStepSzb src, PtrStepSzb dst, bool blue_last, bool start_with_green, cudaStream_t stream)
    {
        typedef typename TypeVec<uchar, cn>::vec_type dst_t;

        const dim3 block(32, 8);
        const dim3 grid(divUp(src.cols, 4 * block.x), divUp(src.rows, block.y));

        cudaSafeCall( cudaFuncSetCacheConfig(Bayer2BGR_8u<dst_t>, cudaFuncCachePreferL1) );

        Bayer2BGR_8u<dst_t><<<grid, block, 0, stream>>>(src, (PtrStepSz<dst_t>)dst, blue_last, start_with_green);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }

    template <int cn>
    void Bayer2BGR_16u_gpu(PtrStepSzb src, PtrStepSzb dst, bool blue_last, bool start_with_green, cudaStream_t stream)
    {
        typedef typename TypeVec<ushort, cn>::vec_type dst_t;

        const dim3 block(32, 8);
        const dim3 grid(divUp(src.cols, 2 * block.x), divUp(src.rows, block.y));

        cudaSafeCall( cudaFuncSetCacheConfig(Bayer2BGR_16u<dst_t>, cudaFuncCachePreferL1) );

        Bayer2BGR_16u<dst_t><<<grid, block, 0, stream>>>(src, (PtrStepSz<dst_t>)dst, blue_last, start_with_green);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }

    template void Bayer2BGR_8u_gpu<1>(PtrStepSzb src, PtrStepSzb dst, bool blue_last, bool start_with_green, cudaStream_t stream);
    template void Bayer2BGR_8u_gpu<3>(PtrStepSzb src, PtrStepSzb dst, bool blue_last, bool start_with_green, cudaStream_t stream);
    template void Bayer2BGR_8u_gpu<4>(PtrStepSzb src, PtrStepSzb dst, bool blue_last, bool start_with_green, cudaStream_t stream);

    template void Bayer2BGR_16u_gpu<1>(PtrStepSzb src, PtrStepSzb dst, bool blue_last, bool start_with_green, cudaStream_t stream);
    template void Bayer2BGR_16u_gpu<3>(PtrStepSzb src, PtrStepSzb dst, bool blue_last, bool start_with_green, cudaStream_t stream);
    template void Bayer2BGR_16u_gpu<4>(PtrStepSzb src, PtrStepSzb dst, bool blue_last, bool start_with_green, cudaStream_t stream);

    //////////////////////////////////////////////////////////////
    // Bayer Demosaicing (Malvar, He, and Cutler)
    //
    // by Morgan McGuire, Williams College
    // http://graphics.cs.williams.edu/papers/BayerJGT09/#shaders
    //
    // ported to CUDA

    template<typename Depth> __device__
    typename TypeVec<Depth, 3>::vec_type make_3(Depth x, Depth y, Depth z);

    template<> __device__ TypeVec<uchar, 3>::vec_type make_3<uchar>(uchar x, uchar y, uchar z) {
        return make_uchar3(x, y, z);
    }

    template<> __device__ TypeVec<ushort, 3>::vec_type make_3<ushort>(ushort x, ushort y, ushort z) {
        return make_ushort3(x, y, z);
    }

    template <typename DstType, class Ptr2D>
    __global__ void MHCdemosaic(PtrStepSz<DstType> dst, Ptr2D src, const int2 firstRed)
    {
        const float   kAx = -1.0f / 8.0f,     kAy = -1.5f / 8.0f,     kAz =  0.5f / 8.0f    /*kAw = -1.0f / 8.0f*/;
        const float   kBx =  2.0f / 8.0f,   /*kBy =  0.0f / 8.0f,*/ /*kBz =  0.0f / 8.0f,*/   kBw =  4.0f / 8.0f  ;
        const float   kCx =  4.0f / 8.0f,     kCy =  6.0f / 8.0f,     kCz =  5.0f / 8.0f    /*kCw =  5.0f / 8.0f*/;
        const float /*kDx =  0.0f / 8.0f,*/   kDy =  2.0f / 8.0f,     kDz = -1.0f / 8.0f    /*kDw = -1.0f / 8.0f*/;
        const float   kEx = -1.0f / 8.0f,     kEy = -1.5f / 8.0f,   /*kEz = -1.0f / 8.0f,*/   kEw =  0.5f / 8.0f  ;
        const float   kFx =  2.0f / 8.0f,   /*kFy =  0.0f / 8.0f,*/   kFz =  4.0f / 8.0f    /*kFw =  0.0f / 8.0f*/;

        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x == 0 || x >= dst.cols - 1 || y == 0 || y >= dst.rows - 1)
            return;

        int2 center;
        center.x = x;
        center.y = y;

        int4 xCoord;
        xCoord.x = center.x - 2;
        xCoord.y = center.x - 1;
        xCoord.z = center.x + 1;
        xCoord.w = center.x + 2;

        int4 yCoord;
        yCoord.x = center.y - 2;
        yCoord.y = center.y - 1;
        yCoord.z = center.y + 1;
        yCoord.w = center.y + 2;

        float C = src(center.y, center.x); // ( 0, 0)

        float4 Dvec;
        Dvec.x = src(yCoord.y, xCoord.y); // (-1,-1)
        Dvec.y = src(yCoord.z, xCoord.y); // (-1, 1)
        Dvec.z = src(yCoord.y, xCoord.z); // ( 1,-1)
        Dvec.w = src(yCoord.z, xCoord.z); // ( 1, 1)


        float4 value;
        value.x = src(yCoord.x, center.x); // ( 0,-2) A0
        value.y = src(yCoord.y, center.x); // ( 0,-1) B0
        value.z = src(center.y, xCoord.x); // (-2, 0) E0
        value.w = src(center.y, xCoord.y); // (-1, 0) F0

        // (A0 + A1), (B0 + B1), (E0 + E1), (F0 + F1)
        value.x += src(yCoord.w, center.x); // ( 0, 2) A1
        value.y += src(yCoord.z, center.x); // ( 0, 1) B1
        value.z += src(center.y, xCoord.w); // ( 2, 0) E1
        value.w += src(center.y, xCoord.z); // ( 1, 0) F1

        float4 PATTERN;
        PATTERN.x = kCx * C;
        PATTERN.y = kCy * C;
        PATTERN.z = kCz * C;
        PATTERN.w = PATTERN.z;

        float D = Dvec.x + Dvec.y + Dvec.z + Dvec.w;

        // There are five filter patterns (identity, cross, checker,
        // theta, phi). Precompute the terms from all of them and then
        // use swizzles to assign to color channels.
        //
        // Channel Matches
        // x cross (e.g., EE G)
        // y checker (e.g., EE B)
        // z theta (e.g., EO R)
        // w phi (e.g., EO B)

        #define A value.x  // A0 + A1
        #define B value.y  // B0 + B1
        #define E value.z  // E0 + E1
        #define F value.w  // F0 + F1

        float3 temp;

        // PATTERN.yzw += (kD.yz * D).xyy;
        temp.x = kDy * D;
        temp.y = kDz * D;
        PATTERN.y += temp.x;
        PATTERN.z += temp.y;
        PATTERN.w += temp.y;

        // PATTERN += (kA.xyz * A).xyzx;
        temp.x = kAx * A;
        temp.y = kAy * A;
        temp.z = kAz * A;
        PATTERN.x += temp.x;
        PATTERN.y += temp.y;
        PATTERN.z += temp.z;
        PATTERN.w += temp.x;

        // PATTERN += (kE.xyw * E).xyxz;
        temp.x = kEx * E;
        temp.y = kEy * E;
        temp.z = kEw * E;
        PATTERN.x += temp.x;
        PATTERN.y += temp.y;
        PATTERN.z += temp.x;
        PATTERN.w += temp.z;

        // PATTERN.xw += kB.xw * B;
        PATTERN.x += kBx * B;
        PATTERN.w += kBw * B;

        // PATTERN.xz += kF.xz * F;
        PATTERN.x += kFx * F;
        PATTERN.z += kFz * F;

        // Determine which of four types of pixels we are on.
        int2 alternate;
        alternate.x = (x + firstRed.x) % 2;
        alternate.y = (y + firstRed.y) % 2;

        typedef typename VecTraits<DstType>::elem_type SrcElemType;
        typedef typename TypeVec<SrcElemType, 3>::vec_type SrcType;

        SrcType pixelColor =
            (alternate.y == 0) ?
                ((alternate.x == 0) ?
                    make_3<SrcElemType>(saturate_cast<SrcElemType>(PATTERN.y), saturate_cast<SrcElemType>(PATTERN.x), saturate_cast<SrcElemType>(C)) :
                    make_3<SrcElemType>(saturate_cast<SrcElemType>(PATTERN.w), saturate_cast<SrcElemType>(C), saturate_cast<SrcElemType>(PATTERN.z))) :
                ((alternate.x == 0) ?
                    make_3<SrcElemType>(saturate_cast<SrcElemType>(PATTERN.z), saturate_cast<SrcElemType>(C), saturate_cast<SrcElemType>(PATTERN.w)) :
                    make_3<SrcElemType>(saturate_cast<SrcElemType>(C), saturate_cast<SrcElemType>(PATTERN.x), saturate_cast<SrcElemType>(PATTERN.y)));

        dst(y, x) = toDst<DstType>(pixelColor);
    }

    template <int cn, typename Depth>
    void MHCdemosaic(PtrStepSz<Depth> src, int2 sourceOffset, PtrStepSz<Depth> dst, int2 firstRed, cudaStream_t stream)
    {
        typedef typename TypeVec<Depth, cn>::vec_type dst_t;

        const dim3 block(32, 8);
        const dim3 grid(divUp(src.cols, block.x), divUp(src.rows, block.y));

        if (sourceOffset.x || sourceOffset.y) {
            cv::cudev::TextureOff<Depth> texSrc(src, sourceOffset.y, sourceOffset.x);
            MHCdemosaic<dst_t, cv::cudev::TextureOffPtr<Depth>><<<grid, block, 0, stream>>>((PtrStepSz<dst_t>)dst, texSrc, firstRed);
        }
        else {
            cv::cudev::Texture<Depth> texSrc(src);
            MHCdemosaic<dst_t, cv::cudev::TexturePtr<Depth>><<<grid, block, 0, stream>>>((PtrStepSz<dst_t>)dst, texSrc, firstRed);
        }

        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }

    template void MHCdemosaic<1, uchar>(PtrStepSzb src, int2 sourceOffset, PtrStepSzb dst, int2 firstRed, cudaStream_t stream);
    template void MHCdemosaic<3, uchar>(PtrStepSzb src, int2 sourceOffset, PtrStepSzb dst, int2 firstRed, cudaStream_t stream);
    template void MHCdemosaic<4, uchar>(PtrStepSzb src, int2 sourceOffset, PtrStepSzb dst, int2 firstRed, cudaStream_t stream);
    template void MHCdemosaic<1, ushort>(PtrStepSz<ushort> src, int2 sourceOffset, PtrStepSz<ushort> dst, int2 firstRed, cudaStream_t stream);
    template void MHCdemosaic<3, ushort>(PtrStepSz<ushort> src, int2 sourceOffset, PtrStepSz<ushort> dst, int2 firstRed, cudaStream_t stream);
    template void MHCdemosaic<4, ushort>(PtrStepSz<ushort> src, int2 sourceOffset, PtrStepSz<ushort> dst, int2 firstRed, cudaStream_t stream);
}}}

#endif /* CUDA_DISABLER */
