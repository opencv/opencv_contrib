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

#include "precomp.hpp"

using namespace cv;
using namespace cv::cuda;

#if !defined (HAVE_CUDA) || defined (CUDA_DISABLER)

void cv::cuda::merge(const GpuMat*, size_t, OutputArray, Stream&) { throw_no_cuda(); }
void cv::cuda::merge(const std::vector<GpuMat>&, OutputArray, Stream&) { throw_no_cuda(); }

void cv::cuda::split(InputArray, GpuMat*, Stream&) { throw_no_cuda(); }
void cv::cuda::split(InputArray, std::vector<GpuMat>&, Stream&) { throw_no_cuda(); }

void cv::cuda::transpose(InputArray, OutputArray, Stream&) { throw_no_cuda(); }

void cv::cuda::flip(InputArray, OutputArray, int, Stream&) { throw_no_cuda(); }

void cv::cuda::copyMakeBorder(InputArray, OutputArray, int, int, int, int, int, Scalar, Stream&) { throw_no_cuda(); }

#else /* !defined (HAVE_CUDA) */

////////////////////////////////////////////////////////////////////////
// flip

namespace
{
    template<int DEPTH> struct NppTypeTraits;
    template<> struct NppTypeTraits<CV_8U>  { typedef Npp8u npp_t; };
    template<> struct NppTypeTraits<CV_8S>  { typedef Npp8s npp_t; };
    template<> struct NppTypeTraits<CV_16U> { typedef Npp16u npp_t; };
    template<> struct NppTypeTraits<CV_16S> { typedef Npp16s npp_t; };
    template<> struct NppTypeTraits<CV_32S> { typedef Npp32s npp_t; };
    template<> struct NppTypeTraits<CV_32F> { typedef Npp32f npp_t; };
    template<> struct NppTypeTraits<CV_64F> { typedef Npp64f npp_t; };

    template <int DEPTH> struct NppMirrorFunc
    {
        typedef typename NppTypeTraits<DEPTH>::npp_t npp_t;

#if USE_NPP_STREAM_CTX
        typedef NppStatus (*func_t)(const npp_t* pSrc, int nSrcStep, npp_t* pDst, int nDstStep, NppiSize oROI, NppiAxis flip, NppStreamContext ctx);
#else
        typedef NppStatus(*func_t)(const npp_t* pSrc, int nSrcStep, npp_t* pDst, int nDstStep, NppiSize oROI, NppiAxis flip);
#endif
    };

    template <int DEPTH, typename NppMirrorFunc<DEPTH>::func_t func> struct NppMirror
    {
        typedef typename NppMirrorFunc<DEPTH>::npp_t npp_t;

        static void call(const GpuMat& src, GpuMat& dst, int flipCode, cudaStream_t stream)
        {
            NppStreamHandler h(stream);

            NppiSize sz;
            sz.width  = src.cols;
            sz.height = src.rows;

#if USE_NPP_STREAM_CTX
            nppSafeCall( func(src.ptr<npp_t>(), static_cast<int>(src.step),
                dst.ptr<npp_t>(), static_cast<int>(dst.step), sz,
                (flipCode == 0 ? NPP_HORIZONTAL_AXIS : (flipCode > 0 ? NPP_VERTICAL_AXIS : NPP_BOTH_AXIS)), h) );
#else
            nppSafeCall( func(src.ptr<npp_t>(), static_cast<int>(src.step),
                dst.ptr<npp_t>(), static_cast<int>(dst.step), sz,
                    (flipCode == 0 ? NPP_HORIZONTAL_AXIS : (flipCode > 0 ? NPP_VERTICAL_AXIS : NPP_BOTH_AXIS))) );
#endif

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }
    };

    template <int DEPTH> struct NppMirrorIFunc
    {
        typedef typename NppTypeTraits<DEPTH>::npp_t npp_t;

#if USE_NPP_STREAM_CTX
        typedef NppStatus (*func_t)(npp_t* pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip, NppStreamContext ctx);
#else
        typedef NppStatus(*func_t)(npp_t* pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip);
#endif
    };

    template <int DEPTH, typename NppMirrorIFunc<DEPTH>::func_t func> struct NppMirrorI
    {
        typedef typename NppMirrorIFunc<DEPTH>::npp_t npp_t;

        static void call(GpuMat& srcDst, int flipCode, cudaStream_t stream)
        {
            NppStreamHandler h(stream);

            NppiSize sz;
            sz.width  = srcDst.cols;
            sz.height = srcDst.rows;
#if USE_NPP_STREAM_CTX
            nppSafeCall(func(srcDst.ptr<npp_t>(), static_cast<int>(srcDst.step),
                sz,
                (flipCode == 0 ? NPP_HORIZONTAL_AXIS : (flipCode > 0 ? NPP_VERTICAL_AXIS : NPP_BOTH_AXIS)), h) );
#else
            nppSafeCall( func(srcDst.ptr<npp_t>(), static_cast<int>(srcDst.step),
                sz,
                (flipCode == 0 ? NPP_HORIZONTAL_AXIS : (flipCode > 0 ? NPP_VERTICAL_AXIS : NPP_BOTH_AXIS))) );
#endif

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }
    };
}

void cv::cuda::flip(InputArray _src, OutputArray _dst, int flipCode, Stream& stream)
{
    typedef void (*func_t)(const GpuMat& src, GpuMat& dst, int flipCode, cudaStream_t stream);
    static const func_t funcs[6][4] =
    {
#if USE_NPP_STREAM_CTX
        {NppMirror<CV_8U, nppiMirror_8u_C1R_Ctx>::call, 0, NppMirror<CV_8U, nppiMirror_8u_C3R_Ctx>::call, NppMirror<CV_8U, nppiMirror_8u_C4R_Ctx>::call},
        {0,0,0,0},
        {NppMirror<CV_16U, nppiMirror_16u_C1R_Ctx>::call, 0, NppMirror<CV_16U, nppiMirror_16u_C3R_Ctx>::call, NppMirror<CV_16U, nppiMirror_16u_C4R_Ctx>::call},
        {0,0,0,0},
        {NppMirror<CV_32S, nppiMirror_32s_C1R_Ctx>::call, 0, NppMirror<CV_32S, nppiMirror_32s_C3R_Ctx>::call, NppMirror<CV_32S, nppiMirror_32s_C4R_Ctx>::call},
        {NppMirror<CV_32F, nppiMirror_32f_C1R_Ctx>::call, 0, NppMirror<CV_32F, nppiMirror_32f_C3R_Ctx>::call, NppMirror<CV_32F, nppiMirror_32f_C4R_Ctx>::call}
#else
        { NppMirror<CV_8U, nppiMirror_8u_C1R>::call, 0, NppMirror<CV_8U, nppiMirror_8u_C3R>::call, NppMirror<CV_8U, nppiMirror_8u_C4R>::call },
        {0,0,0,0},
        {NppMirror<CV_16U, nppiMirror_16u_C1R>::call, 0, NppMirror<CV_16U, nppiMirror_16u_C3R>::call, NppMirror<CV_16U, nppiMirror_16u_C4R>::call},
        {0,0,0,0},
        {NppMirror<CV_32S, nppiMirror_32s_C1R>::call, 0, NppMirror<CV_32S, nppiMirror_32s_C3R>::call, NppMirror<CV_32S, nppiMirror_32s_C4R>::call},
        {NppMirror<CV_32F, nppiMirror_32f_C1R>::call, 0, NppMirror<CV_32F, nppiMirror_32f_C3R>::call, NppMirror<CV_32F, nppiMirror_32f_C4R>::call}
#endif
    };

    typedef void (*ifunc_t)(GpuMat& srcDst, int flipCode, cudaStream_t stream);
    static const ifunc_t ifuncs[6][4] =
    {
#if USE_NPP_STREAM_CTX
        {NppMirrorI<CV_8U, nppiMirror_8u_C1IR_Ctx>::call, 0, NppMirrorI<CV_8U, nppiMirror_8u_C3IR_Ctx>::call, NppMirrorI<CV_8U, nppiMirror_8u_C4IR_Ctx>::call},
        {0,0,0,0},
        {NppMirrorI<CV_16U, nppiMirror_16u_C1IR_Ctx>::call, 0, NppMirrorI<CV_16U, nppiMirror_16u_C3IR_Ctx>::call, NppMirrorI<CV_16U, nppiMirror_16u_C4IR_Ctx>::call},
        {0,0,0,0},
        {NppMirrorI<CV_32S, nppiMirror_32s_C1IR_Ctx>::call, 0, NppMirrorI<CV_32S, nppiMirror_32s_C3IR_Ctx>::call, NppMirrorI<CV_32S, nppiMirror_32s_C4IR_Ctx>::call},
        {NppMirrorI<CV_32F, nppiMirror_32f_C1IR_Ctx>::call, 0, NppMirrorI<CV_32F, nppiMirror_32f_C3IR_Ctx>::call, NppMirrorI<CV_32F, nppiMirror_32f_C4IR_Ctx>::call}
#else
        { NppMirrorI<CV_8U, nppiMirror_8u_C1IR>::call, 0, NppMirrorI<CV_8U, nppiMirror_8u_C3IR>::call, NppMirrorI<CV_8U, nppiMirror_8u_C4IR>::call },
        {0,0,0,0},
        {NppMirrorI<CV_16U, nppiMirror_16u_C1IR>::call, 0, NppMirrorI<CV_16U, nppiMirror_16u_C3IR>::call, NppMirrorI<CV_16U, nppiMirror_16u_C4IR>::call},
        {0,0,0,0},
        {NppMirrorI<CV_32S, nppiMirror_32s_C1IR>::call, 0, NppMirrorI<CV_32S, nppiMirror_32s_C3IR>::call, NppMirrorI<CV_32S, nppiMirror_32s_C4IR>::call},
        {NppMirrorI<CV_32F, nppiMirror_32f_C1IR>::call, 0, NppMirrorI<CV_32F, nppiMirror_32f_C3IR>::call, NppMirrorI<CV_32F, nppiMirror_32f_C4IR>::call}
#endif
    };

    GpuMat src = getInputMat(_src, stream);

    CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32S || src.depth() == CV_32F);
    CV_Assert(src.channels() == 1 || src.channels() == 3 || src.channels() == 4);

    _dst.create(src.size(), src.type());
    GpuMat dst = getOutputMat(_dst, src.size(), src.type(), stream);
    bool isInplace = (src.data == dst.data);
    bool isSizeOdd = (src.cols & 1) == 1 || (src.rows & 1) == 1;
    if (isInplace && isSizeOdd)
        CV_Error(Error::BadROISize, "In-place version of flip only accepts even width/height");

    if (isInplace == false)
        funcs[src.depth()][src.channels() - 1](src, dst, flipCode, StreamAccessor::getStream(stream));
    else // in-place
        ifuncs[src.depth()][src.channels() - 1](src, flipCode, StreamAccessor::getStream(stream));

    syncOutput(dst, _dst, stream);
}

#endif /* !defined (HAVE_CUDA) */
