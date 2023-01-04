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

#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudev.hpp"
#include "opencv2/core/private.cuda.hpp"

using namespace cv;
using namespace cv::cuda;
using namespace cv::cudev;

#define USE_NPP_STREAM_CONTEXT (NPP_VERSION >= (10 * 1000 + 1 * 100 + 0))

namespace
{
  template <int DEPTH> struct NppTransposeFunc
  {
    typedef typename NPPTypeTraits<DEPTH>::npp_type npp_type;

    typedef NppStatus (*func_t)(const npp_type* pSrc, int srcStep, npp_type* pDst, int dstStep, NppiSize srcSize);
    #if USE_NPP_STREAM_CONTEXT
    typedef NppStatus (*func_ctx_t)(const npp_type* pSrc, int srcStep, npp_type* pDst, int dstStep, NppiSize srcSize, NppStreamContext stream);
    #endif
  };

  template <int DEPTH, typename NppTransposeFunc<DEPTH>::func_t func> struct NppTranspose
  {
    typedef typename NppTransposeFunc<DEPTH>::npp_type npp_type;

    static void call(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cudaStream_t stream)
    {
      NppiSize srcsz;
      srcsz.height = src.rows;
      srcsz.width = src.cols;

      cv::cuda::NppStreamHandler h(stream);

      nppSafeCall( func(src.ptr<npp_type>(), static_cast<int>(src.step), dst.ptr<npp_type>(), static_cast<int>(dst.step), srcsz) );

      if (stream == 0)
        cudaSafeCall( cudaDeviceSynchronize() );
    }
  };

  #if USE_NPP_STREAM_CONTEXT
  template <int DEPTH, typename NppTransposeFunc<DEPTH>::func_ctx_t func> struct NppTransposeCtx
  {
    typedef typename NppTransposeFunc<DEPTH>::npp_type npp_type;

    static void call(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cudaStream_t stream)
    {
      NppiSize srcsz;
      srcsz.height = src.rows;
      srcsz.width = src.cols;

      NppStreamContext ctx;
      nppSafeCall( nppGetStreamContext(&ctx) );
      ctx.hStream = stream;

      nppSafeCall( func(src.ptr<npp_type>(), static_cast<int>(src.step), dst.ptr<npp_type>(), static_cast<int>(dst.step), srcsz, ctx) );
    }
  };
  #endif
}

void cv::cuda::transpose(InputArray _src, OutputArray _dst, Stream& stream)
{
    GpuMat src = getInputMat(_src, stream);

    const int srcType = src.type();
    const size_t elemSize = src.elemSize();

    GpuMat dst = getOutputMat(_dst, src.cols, src.rows, src.type(), stream);

    const bool isSupported =
      (elemSize == 1) || (elemSize == 2) || (elemSize == 3) || (elemSize == 4) ||
      (elemSize == 6) || (elemSize == 8) || (elemSize == 12) || (elemSize == 16);

    if (!isSupported)
        CV_Error(Error::StsUnsupportedFormat, "");
    else if (src.empty())
        CV_Error(Error::StsBadArg, "image is empty");

    if ((src.rows == 1) && (src.cols == 1))
        src.copyTo(dst, stream);
    else if (src.rows == 1)
        src.reshape(0, src.cols).copyTo(dst, stream);
    else if ((src.cols == 1) && src.isContinuous())
        src.reshape(0, src.cols).copyTo(dst, stream);
    else
    {
        #if USE_NPP_STREAM_CONTEXT
        constexpr const bool useNppStreamCtx = true;
        #else
        constexpr const bool useNppStreamCtx = false;
        #endif
        cudaStream_t _stream = StreamAccessor::getStream(stream);

        if (!_stream || !useNppStreamCtx)
        {
          typedef void (*func_t)(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cudaStream_t stream);
          //if no direct mapping exists between DEPTH+CHANNELS and the nppiTranspose supported type, we use a nppiTranspose of a similar elemSize
          static const func_t funcs[8][4] = {
            {NppTranspose<CV_8U,  nppiTranspose_8u_C1R>::call,  NppTranspose<CV_16U, nppiTranspose_16u_C1R>::call, NppTranspose<CV_8U, nppiTranspose_8u_C3R>::call,   NppTranspose<CV_8U, nppiTranspose_8u_C4R>::call},
            {NppTranspose<CV_8U,  nppiTranspose_8u_C1R>::call,  NppTranspose<CV_16U, nppiTranspose_16u_C1R>::call, NppTranspose<CV_8U, nppiTranspose_8u_C3R>::call,   NppTranspose<CV_8U, nppiTranspose_8u_C4R>::call},
            {NppTranspose<CV_16U, nppiTranspose_16u_C1R>::call, NppTranspose<CV_32S, nppiTranspose_32s_C1R>::call, NppTranspose<CV_16U, nppiTranspose_16u_C3R>::call, NppTranspose<CV_16U, nppiTranspose_16u_C4R>::call},
            {NppTranspose<CV_16S, nppiTranspose_16s_C1R>::call, NppTranspose<CV_32S, nppiTranspose_32s_C1R>::call, NppTranspose<CV_16S, nppiTranspose_16s_C3R>::call, NppTranspose<CV_16S, nppiTranspose_16s_C4R>::call},
            {NppTranspose<CV_32S, nppiTranspose_32s_C1R>::call, NppTranspose<CV_16S, nppiTranspose_16s_C4R>::call, NppTranspose<CV_32S, nppiTranspose_32s_C3R>::call, NppTranspose<CV_32S, nppiTranspose_32s_C4R>::call},
            {NppTranspose<CV_32F, nppiTranspose_32f_C1R>::call, NppTranspose<CV_16S, nppiTranspose_16s_C4R>::call, NppTranspose<CV_32F, nppiTranspose_32f_C3R>::call, NppTranspose<CV_32F, nppiTranspose_32f_C4R>::call},
            {NppTranspose<CV_16S, nppiTranspose_16s_C4R>::call, NppTranspose<CV_32S, nppiTranspose_32s_C4R>::call, nullptr, nullptr},
            {NppTranspose<CV_16U, nppiTranspose_16u_C1R>::call, NppTranspose<CV_32S, nppiTranspose_32s_C1R>::call, NppTranspose<CV_16U, nppiTranspose_16u_C3R>::call, NppTranspose<CV_16U, nppiTranspose_16u_C4R>::call}
          };
          
          const func_t func = funcs[src.depth()][src.channels() - 1];
          CV_Assert(func != nullptr);

          func(src, dst, _stream);
        }//end if (!_stream || !useNppStreamCtx)
        else//if ((_stream != 0) && useNppStreamCtx)
        {
          #if USE_NPP_STREAM_CONTEXT
          typedef void (*func_t)(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cudaStream_t stream);
          //if no direct mapping exists between DEPTH+CHANNELS and the nppiTranspose supported type, we use a nppiTranspose of a similar elemSize
          static const func_t funcs[8][4] = {
            {NppTransposeCtx<CV_8U,  nppiTranspose_8u_C1R_Ctx>::call,  NppTransposeCtx<CV_16U, nppiTranspose_16u_C1R_Ctx>::call, NppTransposeCtx<CV_8U,  nppiTranspose_8u_C3R_Ctx>::call,  NppTransposeCtx<CV_8U, nppiTranspose_8u_C4R_Ctx>::call},
            {NppTransposeCtx<CV_8U,  nppiTranspose_8u_C1R_Ctx>::call,  NppTransposeCtx<CV_16U, nppiTranspose_16u_C1R_Ctx>::call, NppTransposeCtx<CV_8U,  nppiTranspose_8u_C3R_Ctx>::call,  NppTransposeCtx<CV_8U, nppiTranspose_8u_C4R_Ctx>::call},
            {NppTransposeCtx<CV_16U, nppiTranspose_16u_C1R_Ctx>::call, NppTransposeCtx<CV_32S, nppiTranspose_32s_C1R_Ctx>::call, NppTransposeCtx<CV_16U, nppiTranspose_16u_C3R_Ctx>::call, NppTransposeCtx<CV_16U, nppiTranspose_16u_C4R_Ctx>::call},
            {NppTransposeCtx<CV_16S, nppiTranspose_16s_C1R_Ctx>::call, NppTransposeCtx<CV_32S, nppiTranspose_32s_C1R_Ctx>::call, NppTransposeCtx<CV_16S, nppiTranspose_16s_C3R_Ctx>::call, NppTransposeCtx<CV_16S, nppiTranspose_16s_C4R_Ctx>::call},
            {NppTransposeCtx<CV_32S, nppiTranspose_32s_C1R_Ctx>::call, NppTransposeCtx<CV_16S, nppiTranspose_16s_C4R_Ctx>::call, NppTransposeCtx<CV_32S, nppiTranspose_32s_C3R_Ctx>::call, NppTransposeCtx<CV_32S, nppiTranspose_32s_C4R_Ctx>::call},
            {NppTransposeCtx<CV_32F, nppiTranspose_32f_C1R_Ctx>::call, NppTransposeCtx<CV_16S, nppiTranspose_16s_C4R_Ctx>::call, NppTransposeCtx<CV_32F, nppiTranspose_32f_C3R_Ctx>::call, NppTransposeCtx<CV_32F, nppiTranspose_32f_C4R_Ctx>::call},
            {NppTransposeCtx<CV_16S, nppiTranspose_16s_C4R_Ctx>::call, NppTransposeCtx<CV_32S, nppiTranspose_32s_C4R_Ctx>::call, nullptr, nullptr},
            {NppTransposeCtx<CV_16U, nppiTranspose_16u_C1R_Ctx>::call, NppTransposeCtx<CV_32S, nppiTranspose_32s_C1R_Ctx>::call, NppTransposeCtx<CV_16U, nppiTranspose_16u_C3R_Ctx>::call, NppTransposeCtx<CV_16U, nppiTranspose_16u_C4R_Ctx>::call}
          };

          const func_t func = funcs[src.depth()][src.channels() - 1];
          CV_Assert(func != nullptr);

          func(src, dst, _stream);
          #endif
        }//end if ((_stream != 0) && useNppStreamCtx)
    }//end if
}

#endif
