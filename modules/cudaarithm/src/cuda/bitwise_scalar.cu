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

#include "opencv2/opencv_modules.hpp"

#ifndef HAVE_OPENCV_CUDEV

#error "opencv_cudev is required"

#else

#include "opencv2/cudev.hpp"
#include "opencv2/core/private.cuda.hpp"

using namespace cv::cudev;

void bitScalar(const GpuMat& src, cv::Scalar value, bool, GpuMat& dst, const GpuMat& mask, double, Stream& stream, int op);

namespace
{
    template <template <typename> class Op, typename T>
    void bitScalarOp(const GpuMat& src, uint value, GpuMat& dst, Stream& stream)
    {
        gridTransformUnary(globPtr<T>(src), globPtr<T>(dst), bind2nd(Op<T>(), value), stream);
    }

    typedef void (*bit_scalar_func_t)(const GpuMat& src, uint value, GpuMat& dst, Stream& stream);

    template <typename T, bit_scalar_func_t func> struct BitScalar
    {
        static void call(const GpuMat& src, cv::Scalar value, GpuMat& dst, Stream& stream)
        {
            func(src, cv::saturate_cast<T>(value[0]), dst, stream);
        }
    };

    template <bit_scalar_func_t func> struct BitScalar4
    {
        static void call(const GpuMat& src, cv::Scalar value, GpuMat& dst, Stream& stream)
        {
            uint packedVal = 0;

            packedVal |= cv::saturate_cast<uchar>(value[0]);
            packedVal |= cv::saturate_cast<uchar>(value[1]) << 8;
            packedVal |= cv::saturate_cast<uchar>(value[2]) << 16;
            packedVal |= cv::saturate_cast<uchar>(value[3]) << 24;

            func(src, packedVal, dst, stream);
        }
    };

#ifdef __HIP_PLATFORM_AMD__
    // NPP supplies the per-channel constant bitwise operators (nppiAndC/OrC/XorC)
    // on the C3/C4 inputs the dispatch table routes here. They apply a separate
    // constant to each channel, so reproduce that directly as a HIP kernel: read
    // the cn-channel pixel, combine each lane with its own constant, write back.
    enum { HIP_BIT_AND, HIP_BIT_OR, HIP_BIT_XOR };

    template <typename T>
    __device__ __forceinline__ T hipBitOp(int op, T a, T b)
    {
        return op == HIP_BIT_AND ? (a & b) : (op == HIP_BIT_OR ? (a | b) : (a ^ b));
    }

    template <typename T, int cn>
    __global__ void hipBitwiseCKernel(const GlobPtrSz<typename MakeVec<T, cn>::type> src,
                                      GlobPtr<typename MakeVec<T, cn>::type> dst,
                                      int rows, int cols, T c0, T c1, T c2, T c3, int op)
    {
        typedef typename MakeVec<T, cn>::type vec_type;
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= cols || y >= rows)
            return;
        vec_type s = src(y, x);
        vec_type d;
        T* dp = reinterpret_cast<T*>(&d);
        const T* sp = reinterpret_cast<const T*>(&s);
        dp[0] = hipBitOp<T>(op, sp[0], c0);
        if (cn > 1) dp[1] = hipBitOp<T>(op, sp[1], c1);
        if (cn > 2) dp[2] = hipBitOp<T>(op, sp[2], c2);
        if (cn > 3) dp[3] = hipBitOp<T>(op, sp[3], c3);
        dst(y, x) = d;
    }

    template <typename T, int cn>
    void hipBitwiseC(const GpuMat& src, cv::Scalar value, GpuMat& dst, int op, Stream& _stream)
    {
        typedef typename MakeVec<T, cn>::type vec_type;
        cudaStream_t stream = StreamAccessor::getStream(_stream);
        const dim3 block(32, 8);
        const dim3 grid(divUp(src.cols, block.x), divUp(src.rows, block.y));
        hipBitwiseCKernel<T, cn><<<grid, block, 0, stream>>>(
            globPtr<vec_type>(src), globPtr<vec_type>(dst), src.rows, src.cols,
            cv::saturate_cast<T>(value[0]), cv::saturate_cast<T>(value[1]),
            cv::saturate_cast<T>(value[2]), cv::saturate_cast<T>(value[3]), op);
        CV_CUDEV_SAFE_CALL(cudaGetLastError());
        if (stream == 0)
            CV_CUDEV_SAFE_CALL(cudaDeviceSynchronize());
    }

    template <typename T, int cn, int op> struct HipBitwiseC
    {
        static void call(const GpuMat& src, cv::Scalar value, GpuMat& dst, Stream& stream)
        {
            hipBitwiseC<T, cn>(src, value, dst, op, stream);
        }
    };
#else
    template <int DEPTH, int cn> struct NppBitwiseCFunc
    {
        typedef typename NPPTypeTraits<DEPTH>::npp_type npp_type;

#if USE_NPP_STREAM_CTX
        typedef NppStatus(*func_t)(const npp_type* pSrc1, int nSrc1Step, const npp_type* pConstants, npp_type* pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext ctx);
#else
        typedef NppStatus (*func_t)(const npp_type* pSrc1, int nSrc1Step, const npp_type* pConstants, npp_type* pDst, int nDstStep, NppiSize oSizeROI);
#endif
    };

    template <int DEPTH, int cn, typename NppBitwiseCFunc<DEPTH, cn>::func_t func> struct NppBitwiseC
    {
        typedef typename NppBitwiseCFunc<DEPTH, cn>::npp_type npp_type;

        static void call(const GpuMat& src, cv::Scalar value, GpuMat& dst, Stream& _stream)
        {
            cudaStream_t stream = StreamAccessor::getStream(_stream);
            NppStreamHandler h(stream);

            NppiSize oSizeROI;
            oSizeROI.width = src.cols;
            oSizeROI.height = src.rows;

            const npp_type pConstants[] =
            {
                cv::saturate_cast<npp_type>(value[0]),
                cv::saturate_cast<npp_type>(value[1]),
                cv::saturate_cast<npp_type>(value[2]),
                cv::saturate_cast<npp_type>(value[3])
            };

#if USE_NPP_STREAM_CTX
            nppSafeCall(func(src.ptr<npp_type>(), static_cast<int>(src.step), pConstants, dst.ptr<npp_type>(), static_cast<int>(dst.step), oSizeROI, h));
#else
            nppSafeCall( func(src.ptr<npp_type>(), static_cast<int>(src.step), pConstants, dst.ptr<npp_type>(), static_cast<int>(dst.step), oSizeROI) );
#endif

            if (stream == 0)
                CV_CUDEV_SAFE_CALL( cudaDeviceSynchronize() );
        }
    };
#endif // __HIP_PLATFORM_AMD__
}

void bitScalar(const GpuMat& src, cv::Scalar value, bool, GpuMat& dst, const GpuMat& mask, double, Stream& stream, int op)
{
    CV_UNUSED(mask);

    typedef void (*func_t)(const GpuMat& src, cv::Scalar value, GpuMat& dst, Stream& stream);
    static const func_t funcs[3][6][4] =
    {
#ifdef __HIP_PLATFORM_AMD__
        {
            {BitScalar<uchar, bitScalarOp<bit_and, uchar> >::call  , 0, HipBitwiseC<uchar , 3, HIP_BIT_AND>::call, BitScalar4< bitScalarOp<bit_and, uint> >::call},
            {BitScalar<uchar, bitScalarOp<bit_and, uchar> >::call  , 0, HipBitwiseC<uchar , 3, HIP_BIT_AND>::call, BitScalar4< bitScalarOp<bit_and, uint> >::call},
            {BitScalar<ushort, bitScalarOp<bit_and, ushort> >::call, 0, HipBitwiseC<ushort, 3, HIP_BIT_AND>::call, HipBitwiseC<ushort, 4, HIP_BIT_AND>::call},
            {BitScalar<ushort, bitScalarOp<bit_and, ushort> >::call, 0, HipBitwiseC<ushort, 3, HIP_BIT_AND>::call, HipBitwiseC<ushort, 4, HIP_BIT_AND>::call},
            {BitScalar<uint, bitScalarOp<bit_and, uint> >::call    , 0, HipBitwiseC<uint  , 3, HIP_BIT_AND>::call, HipBitwiseC<uint  , 4, HIP_BIT_AND>::call},
            {BitScalar<uint, bitScalarOp<bit_and, uint> >::call    , 0, HipBitwiseC<uint  , 3, HIP_BIT_AND>::call, HipBitwiseC<uint  , 4, HIP_BIT_AND>::call}
        },
        {
            {BitScalar<uchar, bitScalarOp<bit_or, uchar> >::call  , 0, HipBitwiseC<uchar , 3, HIP_BIT_OR>::call, BitScalar4< bitScalarOp<bit_or, uint> >::call},
            {BitScalar<uchar, bitScalarOp<bit_or, uchar> >::call  , 0, HipBitwiseC<uchar , 3, HIP_BIT_OR>::call, BitScalar4< bitScalarOp<bit_or, uint> >::call},
            {BitScalar<ushort, bitScalarOp<bit_or, ushort> >::call, 0, HipBitwiseC<ushort, 3, HIP_BIT_OR>::call, HipBitwiseC<ushort, 4, HIP_BIT_OR>::call},
            {BitScalar<ushort, bitScalarOp<bit_or, ushort> >::call, 0, HipBitwiseC<ushort, 3, HIP_BIT_OR>::call, HipBitwiseC<ushort, 4, HIP_BIT_OR>::call},
            {BitScalar<uint, bitScalarOp<bit_or, uint> >::call    , 0, HipBitwiseC<uint  , 3, HIP_BIT_OR>::call, HipBitwiseC<uint  , 4, HIP_BIT_OR>::call},
            {BitScalar<uint, bitScalarOp<bit_or, uint> >::call    , 0, HipBitwiseC<uint  , 3, HIP_BIT_OR>::call, HipBitwiseC<uint  , 4, HIP_BIT_OR>::call}
        },
        {
            {BitScalar<uchar, bitScalarOp<bit_xor, uchar> >::call  , 0, HipBitwiseC<uchar , 3, HIP_BIT_XOR>::call, BitScalar4< bitScalarOp<bit_xor, uint> >::call},
            {BitScalar<uchar, bitScalarOp<bit_xor, uchar> >::call  , 0, HipBitwiseC<uchar , 3, HIP_BIT_XOR>::call, BitScalar4< bitScalarOp<bit_xor, uint> >::call},
            {BitScalar<ushort, bitScalarOp<bit_xor, ushort> >::call, 0, HipBitwiseC<ushort, 3, HIP_BIT_XOR>::call, HipBitwiseC<ushort, 4, HIP_BIT_XOR>::call},
            {BitScalar<ushort, bitScalarOp<bit_xor, ushort> >::call, 0, HipBitwiseC<ushort, 3, HIP_BIT_XOR>::call, HipBitwiseC<ushort, 4, HIP_BIT_XOR>::call},
            {BitScalar<uint, bitScalarOp<bit_xor, uint> >::call    , 0, HipBitwiseC<uint  , 3, HIP_BIT_XOR>::call, HipBitwiseC<uint  , 4, HIP_BIT_XOR>::call},
            {BitScalar<uint, bitScalarOp<bit_xor, uint> >::call    , 0, HipBitwiseC<uint  , 3, HIP_BIT_XOR>::call, HipBitwiseC<uint  , 4, HIP_BIT_XOR>::call}
        }
#elif USE_NPP_STREAM_CTX
        {
            {BitScalar<uchar, bitScalarOp<bit_and, uchar> >::call  , 0, NppBitwiseC<CV_8U , 3, nppiAndC_8u_C3R_Ctx >::call, BitScalar4< bitScalarOp<bit_and, uint> >::call},
            {BitScalar<uchar, bitScalarOp<bit_and, uchar> >::call  , 0, NppBitwiseC<CV_8U , 3, nppiAndC_8u_C3R_Ctx >::call, BitScalar4< bitScalarOp<bit_and, uint> >::call},
            {BitScalar<ushort, bitScalarOp<bit_and, ushort> >::call, 0, NppBitwiseC<CV_16U, 3, nppiAndC_16u_C3R_Ctx>::call, NppBitwiseC<CV_16U, 4, nppiAndC_16u_C4R_Ctx>::call},
            {BitScalar<ushort, bitScalarOp<bit_and, ushort> >::call, 0, NppBitwiseC<CV_16U, 3, nppiAndC_16u_C3R_Ctx>::call, NppBitwiseC<CV_16U, 4, nppiAndC_16u_C4R_Ctx>::call},
            {BitScalar<uint, bitScalarOp<bit_and, uint> >::call    , 0, NppBitwiseC<CV_32S, 3, nppiAndC_32s_C3R_Ctx>::call, NppBitwiseC<CV_32S, 4, nppiAndC_32s_C4R_Ctx>::call},
            {BitScalar<uint, bitScalarOp<bit_and, uint> >::call    , 0, NppBitwiseC<CV_32S, 3, nppiAndC_32s_C3R_Ctx>::call, NppBitwiseC<CV_32S, 4, nppiAndC_32s_C4R_Ctx>::call}
        },
        {
            {BitScalar<uchar, bitScalarOp<bit_or, uchar> >::call  , 0, NppBitwiseC<CV_8U , 3, nppiOrC_8u_C3R_Ctx >::call, BitScalar4< bitScalarOp<bit_or, uint> >::call},
            {BitScalar<uchar, bitScalarOp<bit_or, uchar> >::call  , 0, NppBitwiseC<CV_8U , 3, nppiOrC_8u_C3R_Ctx >::call, BitScalar4< bitScalarOp<bit_or, uint> >::call},
            {BitScalar<ushort, bitScalarOp<bit_or, ushort> >::call, 0, NppBitwiseC<CV_16U, 3, nppiOrC_16u_C3R_Ctx>::call, NppBitwiseC<CV_16U, 4, nppiOrC_16u_C4R_Ctx>::call},
            {BitScalar<ushort, bitScalarOp<bit_or, ushort> >::call, 0, NppBitwiseC<CV_16U, 3, nppiOrC_16u_C3R_Ctx>::call, NppBitwiseC<CV_16U, 4, nppiOrC_16u_C4R_Ctx>::call},
            {BitScalar<uint, bitScalarOp<bit_or, uint> >::call    , 0, NppBitwiseC<CV_32S, 3, nppiOrC_32s_C3R_Ctx>::call, NppBitwiseC<CV_32S, 4, nppiOrC_32s_C4R_Ctx>::call},
            {BitScalar<uint, bitScalarOp<bit_or, uint> >::call    , 0, NppBitwiseC<CV_32S, 3, nppiOrC_32s_C3R_Ctx>::call, NppBitwiseC<CV_32S, 4, nppiOrC_32s_C4R_Ctx>::call}
        },
        {
            {BitScalar<uchar, bitScalarOp<bit_xor, uchar> >::call  , 0, NppBitwiseC<CV_8U , 3, nppiXorC_8u_C3R_Ctx >::call, BitScalar4< bitScalarOp<bit_xor, uint> >::call},
            {BitScalar<uchar, bitScalarOp<bit_xor, uchar> >::call  , 0, NppBitwiseC<CV_8U , 3, nppiXorC_8u_C3R_Ctx >::call, BitScalar4< bitScalarOp<bit_xor, uint> >::call},
            {BitScalar<ushort, bitScalarOp<bit_xor, ushort> >::call, 0, NppBitwiseC<CV_16U, 3, nppiXorC_16u_C3R_Ctx>::call, NppBitwiseC<CV_16U, 4, nppiXorC_16u_C4R_Ctx>::call},
            {BitScalar<ushort, bitScalarOp<bit_xor, ushort> >::call, 0, NppBitwiseC<CV_16U, 3, nppiXorC_16u_C3R_Ctx>::call, NppBitwiseC<CV_16U, 4, nppiXorC_16u_C4R_Ctx>::call},
            {BitScalar<uint, bitScalarOp<bit_xor, uint> >::call    , 0, NppBitwiseC<CV_32S, 3, nppiXorC_32s_C3R_Ctx>::call, NppBitwiseC<CV_32S, 4, nppiXorC_32s_C4R_Ctx>::call},
            {BitScalar<uint, bitScalarOp<bit_xor, uint> >::call    , 0, NppBitwiseC<CV_32S, 3, nppiXorC_32s_C3R_Ctx>::call, NppBitwiseC<CV_32S, 4, nppiXorC_32s_C4R_Ctx>::call}
        }
#else
        {
            { BitScalar<uchar, bitScalarOp<bit_and, uchar> >::call, 0, NppBitwiseC<CV_8U, 3, nppiAndC_8u_C3R >::call, BitScalar4< bitScalarOp<bit_and, uint> >::call },
            { BitScalar<uchar, bitScalarOp<bit_and, uchar> >::call  , 0, NppBitwiseC<CV_8U , 3, nppiAndC_8u_C3R >::call, BitScalar4< bitScalarOp<bit_and, uint> >::call },
            { BitScalar<ushort, bitScalarOp<bit_and, ushort> >::call, 0, NppBitwiseC<CV_16U, 3, nppiAndC_16u_C3R>::call, NppBitwiseC<CV_16U, 4, nppiAndC_16u_C4R>::call },
            { BitScalar<ushort, bitScalarOp<bit_and, ushort> >::call, 0, NppBitwiseC<CV_16U, 3, nppiAndC_16u_C3R>::call, NppBitwiseC<CV_16U, 4, nppiAndC_16u_C4R>::call },
            { BitScalar<uint, bitScalarOp<bit_and, uint> >::call    , 0, NppBitwiseC<CV_32S, 3, nppiAndC_32s_C3R>::call, NppBitwiseC<CV_32S, 4, nppiAndC_32s_C4R>::call },
            { BitScalar<uint, bitScalarOp<bit_and, uint> >::call    , 0, NppBitwiseC<CV_32S, 3, nppiAndC_32s_C3R>::call, NppBitwiseC<CV_32S, 4, nppiAndC_32s_C4R>::call }
        },
        {
            {BitScalar<uchar, bitScalarOp<bit_or, uchar> >::call  , 0, NppBitwiseC<CV_8U , 3, nppiOrC_8u_C3R >::call, BitScalar4< bitScalarOp<bit_or, uint> >::call},
            {BitScalar<uchar, bitScalarOp<bit_or, uchar> >::call  , 0, NppBitwiseC<CV_8U , 3, nppiOrC_8u_C3R >::call, BitScalar4< bitScalarOp<bit_or, uint> >::call},
            {BitScalar<ushort, bitScalarOp<bit_or, ushort> >::call, 0, NppBitwiseC<CV_16U, 3, nppiOrC_16u_C3R>::call, NppBitwiseC<CV_16U, 4, nppiOrC_16u_C4R>::call},
            {BitScalar<ushort, bitScalarOp<bit_or, ushort> >::call, 0, NppBitwiseC<CV_16U, 3, nppiOrC_16u_C3R>::call, NppBitwiseC<CV_16U, 4, nppiOrC_16u_C4R>::call},
            {BitScalar<uint, bitScalarOp<bit_or, uint> >::call    , 0, NppBitwiseC<CV_32S, 3, nppiOrC_32s_C3R>::call, NppBitwiseC<CV_32S, 4, nppiOrC_32s_C4R>::call},
            {BitScalar<uint, bitScalarOp<bit_or, uint> >::call    , 0, NppBitwiseC<CV_32S, 3, nppiOrC_32s_C3R>::call, NppBitwiseC<CV_32S, 4, nppiOrC_32s_C4R>::call}
        },
        {
            {BitScalar<uchar, bitScalarOp<bit_xor, uchar> >::call  , 0, NppBitwiseC<CV_8U , 3, nppiXorC_8u_C3R >::call, BitScalar4< bitScalarOp<bit_xor, uint> >::call},
            {BitScalar<uchar, bitScalarOp<bit_xor, uchar> >::call  , 0, NppBitwiseC<CV_8U , 3, nppiXorC_8u_C3R >::call, BitScalar4< bitScalarOp<bit_xor, uint> >::call},
            {BitScalar<ushort, bitScalarOp<bit_xor, ushort> >::call, 0, NppBitwiseC<CV_16U, 3, nppiXorC_16u_C3R>::call, NppBitwiseC<CV_16U, 4, nppiXorC_16u_C4R>::call},
            {BitScalar<ushort, bitScalarOp<bit_xor, ushort> >::call, 0, NppBitwiseC<CV_16U, 3, nppiXorC_16u_C3R>::call, NppBitwiseC<CV_16U, 4, nppiXorC_16u_C4R>::call},
            {BitScalar<uint, bitScalarOp<bit_xor, uint> >::call    , 0, NppBitwiseC<CV_32S, 3, nppiXorC_32s_C3R>::call, NppBitwiseC<CV_32S, 4, nppiXorC_32s_C4R>::call},
            {BitScalar<uint, bitScalarOp<bit_xor, uint> >::call    , 0, NppBitwiseC<CV_32S, 3, nppiXorC_32s_C3R>::call, NppBitwiseC<CV_32S, 4, nppiXorC_32s_C4R>::call}
        }
#endif
    };

    const int depth = src.depth();
    const int cn = src.channels();

    CV_DbgAssert( depth <= CV_32F );
    CV_DbgAssert( cn == 1 || cn == 3 || cn == 4 );
    CV_DbgAssert( mask.empty() );
    CV_DbgAssert( op >= 0 && op < 3 );

    funcs[op][depth][cn - 1](src, value, dst, stream);
}

#endif
