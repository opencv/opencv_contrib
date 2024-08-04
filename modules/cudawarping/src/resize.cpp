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

#if !defined HAVE_CUDA || defined(CUDA_DISABLER)

void cv::cuda::resize(InputArray /*src*/, OutputArray /*dst*/, Size /*dsize*/,
    double /*fx*/, double /*fy*/, int /*interpolation*/, Stream& /*stream*/)
{ throw_no_cuda(); }

void cv::cuda::resizeOnnx(InputArray /*src*/, OutputArray /*dst*/, Size /*dsize*/,
    Point2d /*scale*/, int /*interpolation*/, float /*cubicCoeff*/, Stream& /*stream*/)
{ throw_no_cuda(); }

#else // HAVE_CUDA

namespace cv { namespace cuda { namespace device
{
template <typename T>
void resize(const PtrStepSzb& src, const PtrStepSzb& srcWhole, int yoff, int xoff,
    const PtrStepSzb& dst, float fy, float fx, int interpolation, cudaStream_t stream);

template <typename T, typename W>
void resizeOnnx(int cn, float A, PtrStepSzb const& src, PtrStepSzb const& dst,
    Matx22f const& M, Point2f const& scale, int interpolation, cudaStream_t stream);

void resizeOnnxNN(size_t elemSize, PtrStepSzb const& src, PtrStepSzb const& dst,
    Matx22f const& M, int mode, cudaStream_t stream);
}}}

namespace cv
{
static Vec2f interCoordinate(int coordinate, int dst, int src, double scale)
{
    float a, b;
    if (coordinate == INTER_HALF_PIXEL
        || coordinate == INTER_HALF_PIXEL_SYMMETRIC
        || coordinate == INTER_HALF_PIXEL_PYTORCH)
    {
        a = static_cast<float>(1.0 / scale);
        b = static_cast<float>(0.5 / scale - 0.5);
        if (coordinate == INTER_HALF_PIXEL_SYMMETRIC)
            b += static_cast<float>(0.5 * (src - dst / scale));
        if (coordinate == INTER_HALF_PIXEL_PYTORCH && dst <= 1)
        {
            a = 0.f;
            b = -0.5f;
        }
    }
    else if (coordinate == INTER_ALIGN_CORNERS)
    {
        a = static_cast<float>((src - 1.0) / (src * scale - 1.0));
        b = 0.f;
    }
    else if (coordinate == INTER_ASYMMETRIC)
    {
        a = static_cast<float>(1.0 / scale);
        b = 0.f;
    }
    else
        CV_Error(Error::StsBadArg, format("Unknown coordinate transformation mode %d", coordinate));
    return Vec2f(a, b);
}
}

void cv::cuda::resize(InputArray _src, OutputArray _dst, Size dsize, double fx, double fy, int interpolation, Stream& stream)
{
    GpuMat src = _src.getGpuMat();

    typedef void (*func_t)(const PtrStepSzb& src, const PtrStepSzb& srcWhole, int yoff, int xoff, const PtrStepSzb& dst, float fy, float fx, int interpolation, cudaStream_t stream);
    static const func_t funcs[6][4] =
    {
        {device::resize<uchar>      , 0 /*device::resize<uchar2>*/ , device::resize<uchar3>     , device::resize<uchar4>     },
        {0 /*device::resize<schar>*/, 0 /*device::resize<char2>*/  , 0 /*device::resize<char3>*/, 0 /*device::resize<char4>*/},
        {device::resize<ushort>     , 0 /*device::resize<ushort2>*/, device::resize<ushort3>    , device::resize<ushort4>    },
        {device::resize<short>      , 0 /*device::resize<short2>*/ , device::resize<short3>     , device::resize<short4>     },
        {0 /*device::resize<int>*/  , 0 /*device::resize<int2>*/   , 0 /*device::resize<int3>*/ , 0 /*device::resize<int4>*/ },
        {device::resize<float>      , 0 /*device::resize<float2>*/ , device::resize<float3>     , device::resize<float4>     }
    };

    CV_Assert( src.depth() <= CV_32F && src.channels() <= 4 );
    CV_Assert( interpolation == INTER_NEAREST || interpolation == INTER_LINEAR || interpolation == INTER_CUBIC || interpolation == INTER_AREA );
    CV_Assert( !(dsize == Size()) || (fx > 0 && fy > 0) );

    if (dsize == Size())
    {
        dsize = Size(saturate_cast<int>(src.cols * fx), saturate_cast<int>(src.rows * fy));
    }
    else
    {
        fx = static_cast<double>(dsize.width) / src.cols;
        fy = static_cast<double>(dsize.height) / src.rows;
    }

    _dst.create(dsize, src.type());
    GpuMat dst = _dst.getGpuMat();

    if (dsize == src.size())
    {
        src.copyTo(dst, stream);
        return;
    }

    const func_t func = funcs[src.depth()][src.channels() - 1];

    if (!func)
        CV_Error(Error::StsUnsupportedFormat, "Unsupported combination of source and destination types");

    Size wholeSize;
    Point ofs;
    src.locateROI(wholeSize, ofs);
    PtrStepSzb wholeSrc(wholeSize.height, wholeSize.width, src.datastart, src.step);

    func(src, wholeSrc, ofs.y, ofs.x, dst, static_cast<float>(1.0 / fy), static_cast<float>(1.0 / fx), interpolation, StreamAccessor::getStream(stream));
}


void cv::cuda::resizeOnnx(InputArray _src, OutputArray _dst,
    Size dsize, Point2d scale, int interpolation, float cubicCoeff, Stream& stream)
{
    GpuMat src = _src.getGpuMat();
    Size ssize = _src.size();
    CV_CheckEQ(_src.dims(), 2, "only 2 dim image is support now");
    CV_CheckFalse(ssize.empty(), "src size must not be empty");
    if (dsize.empty())
    {
        CV_CheckGT(scale.x, 0.0, "scale must > 0 if no dsize given");
        CV_CheckGT(scale.y, 0.0, "scale must > 0 if no dsize given");
        dsize.width = static_cast<int>(scale.x * ssize.width);
        dsize.height = static_cast<int>(scale.y * ssize.height);
    }
    if (scale.x == 0 || scale.y == 0)
    {
        scale.x = static_cast<double>(dsize.width) / ssize.width;
        scale.y = static_cast<double>(dsize.height) / ssize.height;
    }
    CV_CheckFalse(dsize.empty(), "dst size must not empty");
    CV_CheckGT(scale.x, 0.0, "require computed or given scale > 0");
    CV_CheckGT(scale.y, 0.0, "require computed or given scale > 0");

    int sampler = interpolation & INTER_SAMPLER_MASK;
    int nearest = interpolation & INTER_NEAREST_MODE_MASK;
    int coordinate = interpolation & INTER_COORDINATE_MASK;
    CV_Assert(
        sampler == INTER_NEAREST ||
        sampler == INTER_LINEAR ||
        sampler == INTER_CUBIC);
    CV_Assert(
        nearest == INTER_NEAREST_PREFER_FLOOR ||
        nearest == INTER_NEAREST_PREFER_CEIL ||
        nearest == INTER_NEAREST_FLOOR ||
        nearest == INTER_NEAREST_CEIL);
    CV_Assert(
        coordinate == INTER_HALF_PIXEL ||
        coordinate == INTER_HALF_PIXEL_PYTORCH ||
        coordinate == INTER_HALF_PIXEL_SYMMETRIC ||
        coordinate == INTER_ALIGN_CORNERS ||
        coordinate == INTER_ASYMMETRIC);

    _dst.create(dsize, _src.type());
    GpuMat dst = _dst.getGpuMat();
    if (dsize == ssize)
    {
        src.copyTo(dst, stream);
        return;
    }
    if (scale.x >= 1.0 && scale.y >= 1.0)
        interpolation &= ~INTER_ANTIALIAS_MASK;

    Point2f scalef = static_cast<Point2f>(scale);
    Matx22f M;
    Vec2f xcoef = interCoordinate(coordinate, dsize.width, ssize.width, scale.x);
    Vec2f ycoef = interCoordinate(coordinate, dsize.height, ssize.height, scale.y);
    M(0, 0) = xcoef[0];
    M(0, 1) = xcoef[1];
    M(1, 0) = ycoef[0];
    M(1, 1) = ycoef[1];

    if (sampler == INTER_NEAREST)
    {
        device::resizeOnnxNN(src.elemSize(),
            src, dst, M, nearest, StreamAccessor::getStream(stream));
        return;
    }

    int depth = src.depth(), cn = src.channels();
    CV_CheckDepth(depth, depth <= CV_64F,
        "only support float in cuda kernel when not use nearest sampler");

    using Func = void(*)(int cn, float A,
        PtrStepSzb const& src, PtrStepSzb const& dst, Matx22f const& M,
        Point2f const& scale, int interpolation, cudaStream_t stream);
    static Func const funcs[CV_DEPTH_MAX] =
    {
        device::resizeOnnx<uchar, float>,
        device::resizeOnnx<schar, float>,
        device::resizeOnnx<ushort, float>,
        device::resizeOnnx<short, float>,
        device::resizeOnnx<int, double>,
        device::resizeOnnx<float, float>,
        device::resizeOnnx<double, double>,
        /*device::resizeOnnx<__half, float>*/ nullptr,
    };

    Func const func = funcs[depth];
    if (!func)
        CV_Error(Error::StsUnsupportedFormat, "Unsupported depth");
    func(cn, cubicCoeff, src, dst, M, scalef, interpolation,
        StreamAccessor::getStream(stream));
}

#endif // HAVE_CUDA
