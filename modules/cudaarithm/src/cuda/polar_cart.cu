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

//do not use implicit cv::cuda to avoid clash of tuples from ::cuda::std
/*using namespace cv;
using namespace cv::cuda;*/

using namespace cv::cudev;

void cv::cuda::magnitude(InputArray _x, InputArray _y, OutputArray _dst, Stream& stream)
{
    GpuMat x = getInputMat(_x, stream);
    GpuMat y = getInputMat(_y, stream);

    CV_Assert( x.depth() == CV_32F );
    CV_Assert( y.type() == x.type() && y.size() == x.size() );

    GpuMat dst = getOutputMat(_dst, x.size(), CV_32FC1, stream);

    gridTransformBinary(globPtr<float>(x), globPtr<float>(y), globPtr<float>(dst), magnitude_func<float>(), stream);

    syncOutput(dst, _dst, stream);
}

void cv::cuda::magnitudeSqr(InputArray _x, InputArray _y, OutputArray _dst, Stream& stream)
{
    GpuMat x = getInputMat(_x, stream);
    GpuMat y = getInputMat(_y, stream);

    CV_Assert( x.depth() == CV_32F );
    CV_Assert( y.type() == x.type() && y.size() == x.size() );

    GpuMat dst = getOutputMat(_dst, x.size(), CV_32FC1, stream);

    gridTransformBinary(globPtr<float>(x), globPtr<float>(y), globPtr<float>(dst), magnitude_sqr_func<float>(), stream);

    syncOutput(dst, _dst, stream);
}

void cv::cuda::phase(InputArray _x, InputArray _y, OutputArray _dst, bool angleInDegrees, Stream& stream)
{
    GpuMat x = getInputMat(_x, stream);
    GpuMat y = getInputMat(_y, stream);

    CV_Assert( x.depth() == CV_32F );
    CV_Assert( y.type() == x.type() && y.size() == x.size() );

    GpuMat dst = getOutputMat(_dst, x.size(), CV_32FC1, stream);

    if (angleInDegrees)
        gridTransformBinary(globPtr<float>(x), globPtr<float>(y), globPtr<float>(dst), direction_func<float, true>(), stream);
    else
        gridTransformBinary(globPtr<float>(x), globPtr<float>(y), globPtr<float>(dst), direction_func<float, false>(), stream);

    syncOutput(dst, _dst, stream);
}

void cv::cuda::phase(InputArray _xy, OutputArray _dst, bool angleInDegrees, Stream& stream)
{
    GpuMat xy = getInputMat(_xy, stream);

    CV_Assert( xy.type() == CV_32FC2 );

    GpuMat dst = getOutputMat(_dst, xy.size(), CV_32FC1, stream);

    if (angleInDegrees)
        gridTransformUnary(globPtr<float2>(xy), globPtr<float>(dst), direction_interleaved_func<float2, true>(), stream);
    else
        gridTransformUnary(globPtr<float2>(xy), globPtr<float>(dst), direction_interleaved_func<float2, false>(), stream);

    syncOutput(dst, _dst, stream);
}

void cv::cuda::cartToPolar(InputArray _x, InputArray _y, OutputArray _mag, OutputArray _angle, bool angleInDegrees, Stream& stream)
{
    GpuMat x = getInputMat(_x, stream);
    GpuMat y = getInputMat(_y, stream);

    CV_Assert( x.depth() == CV_32F );
    CV_Assert( y.type() == x.type() && y.size() == x.size() );

    GpuMat mag = getOutputMat(_mag, x.size(), CV_32FC1, stream);
    GpuMat angle = getOutputMat(_angle, x.size(), CV_32FC1, stream);

    GpuMat_<float> xc(x);
    GpuMat_<float> yc(y);
    GpuMat_<float> magc(mag);
    GpuMat_<float> anglec(angle);

    if (angleInDegrees)
        gridTransformBinary(xc, yc, magc, anglec, magnitude_func<float>(), direction_func<float, true>(), stream);
    else
        gridTransformBinary(xc, yc, magc, anglec, magnitude_func<float>(), direction_func<float, false>(), stream);

    syncOutput(mag, _mag, stream);
    syncOutput(angle, _angle, stream);
}

void cv::cuda::cartToPolar(InputArray _xy, OutputArray _mag, OutputArray _angle, bool angleInDegrees, Stream& stream)
{
    GpuMat xy = getInputMat(_xy, stream);

    CV_Assert( xy.type() == CV_32FC2 );

    GpuMat mag = getOutputMat(_mag, xy.size(), CV_32FC1, stream);
    GpuMat angle = getOutputMat(_angle, xy.size(), CV_32FC1, stream);

    GpuMat_<float> magc(mag);
    GpuMat_<float> anglec(angle);

    gridTransformUnary(globPtr<float2>(xy), globPtr<float>(magc), magnitude_interleaved_func<float2>(), stream);

    if (angleInDegrees)
    {
        gridTransformUnary(globPtr<float2>(xy), globPtr<float>(anglec), direction_interleaved_func<float2, true>(), stream);
    }
    else
    {
        gridTransformUnary(globPtr<float2>(xy), globPtr<float>(anglec), direction_interleaved_func<float2, false>(), stream);
    }

    syncOutput(mag, _mag, stream);
    syncOutput(angle, _angle, stream);
}

void cv::cuda::cartToPolar(InputArray _xy, OutputArray _magAngle, bool angleInDegrees, Stream& stream)
{
    GpuMat xy = getInputMat(_xy, stream);

    CV_Assert( xy.type() == CV_32FC2 );

    GpuMat magAngle = getOutputMat(_magAngle, xy.size(), CV_32FC2, stream);

    if (angleInDegrees)
    {
        gridTransformUnary(globPtr<float2>(xy),
            globPtr<float2>(magAngle),
            magnitude_direction_interleaved_func<float2, true>(),
            stream);
    }
    else
    {
        gridTransformUnary(globPtr<float2>(xy),
            globPtr<float2>(magAngle),
            magnitude_direction_interleaved_func<float2, false>(),
            stream);
    }

    syncOutput(magAngle, _magAngle, stream);
}

namespace
{
    template <typename T> struct sincos_op
    {
        __device__ __forceinline__ void operator()(T a, T *sptr, T *cptr) const
        {
            ::sincos(a, sptr, cptr);
        }
    };
    template <> struct sincos_op<float>
    {
        __device__ __forceinline__ void operator()(float a, float *sptr, float *cptr) const
        {
            ::sincosf(a, sptr, cptr);
        }
    };

    template <typename T, bool useMag>
    __global__ void polarToCartImpl_(const PtrStep<T> mag, const PtrStepSz<T> angle, PtrStep<T> xmat, PtrStep<T> ymat, const T scale)
    {
        const int x = blockDim.x * blockIdx.x + threadIdx.x;
        const int y = blockDim.y * blockIdx.y + threadIdx.y;

        if (x >= angle.cols || y >= angle.rows)
            return;

        const T mag_val = useMag ? mag(y, x) : static_cast<T>(1.0);
        const T angle_val = angle(y, x);

        T sin_a, cos_a;
        sincos_op<T> op;
        op(scale * angle_val, &sin_a, &cos_a);

        xmat(y, x) = mag_val * cos_a;
        ymat(y, x) = mag_val * sin_a;
    }

    template <typename T, bool useMag>
    __global__ void polarToCartDstInterleavedImpl_(const PtrStep<T> mag, const PtrStepSz<T> angle, PtrStep<typename MakeVec<T, 2>::type > xymat, const T scale)
    {
        typedef typename MakeVec<T, 2>::type T2;
        const int x = blockDim.x * blockIdx.x + threadIdx.x;
        const int y = blockDim.y * blockIdx.y + threadIdx.y;

        if (x >= angle.cols || y >= angle.rows)
            return;

        const T mag_val = useMag ? mag(y, x) : static_cast<T>(1.0);
        const T angle_val = angle(y, x);

        T sin_a, cos_a;
        sincos_op<T> op;
        op(scale * angle_val, &sin_a, &cos_a);

        const T2 xy = {mag_val * cos_a, mag_val * sin_a};
        xymat(y, x) = xy;
    }

    template <typename T>
    __global__ void polarToCartInterleavedImpl_(const PtrStepSz<typename MakeVec<T, 2>::type > magAngle, PtrStep<typename MakeVec<T, 2>::type > xymat, const T scale)
    {
        typedef typename MakeVec<T, 2>::type T2;
        const int x = blockDim.x * blockIdx.x + threadIdx.x;
        const int y = blockDim.y * blockIdx.y + threadIdx.y;

        if (x >= magAngle.cols || y >= magAngle.rows)
            return;

        const T2 magAngle_val = magAngle(y, x);
        const T mag_val = magAngle_val.x;
        const T angle_val = magAngle_val.y;

        T sin_a, cos_a;
        sincos_op<T> op;
        op(scale * angle_val, &sin_a, &cos_a);

        const T2 xy = {mag_val * cos_a, mag_val * sin_a};
        xymat(y, x) = xy;
    }

    template <typename T>
    void polarToCartImpl(const GpuMat& mag, const GpuMat& angle, GpuMat& x, GpuMat& y, bool angleInDegrees, cudaStream_t& stream)
    {
        const dim3 block(32, 8);
        const dim3 grid(divUp(angle.cols, block.x), divUp(angle.rows, block.y));

        const T scale = angleInDegrees ? static_cast<T>(CV_PI / 180.0) : static_cast<T>(1.0);

        if (mag.empty())
            polarToCartImpl_<T, false> << <grid, block, 0, stream >> >(mag, angle, x, y, scale);
        else
            polarToCartImpl_<T, true> << <grid, block, 0, stream >> >(mag, angle, x, y, scale);
    }

    template <typename T>
    void polarToCartDstInterleavedImpl(const GpuMat& mag, const GpuMat& angle, GpuMat& xy, bool angleInDegrees, cudaStream_t& stream)
    {
        typedef typename MakeVec<T, 2>::type T2;

        const dim3 block(32, 8);
        const dim3 grid(divUp(angle.cols, block.x), divUp(angle.rows, block.y));

        const T scale = angleInDegrees ? static_cast<T>(CV_PI / 180.0) : static_cast<T>(1.0);

        if (mag.empty())
            polarToCartDstInterleavedImpl_<T, false> << <grid, block, 0, stream >> >(mag, angle, xy, scale);
        else
            polarToCartDstInterleavedImpl_<T, true> << <grid, block, 0, stream >> >(mag, angle, xy, scale);
    }

    template <typename T>
    void polarToCartInterleavedImpl(const GpuMat& magAngle, GpuMat& xy, bool angleInDegrees, cudaStream_t& stream)
    {
        typedef typename MakeVec<T, 2>::type T2;

        const dim3 block(32, 8);
        const dim3 grid(divUp(magAngle.cols, block.x), divUp(magAngle.rows, block.y));

        const T scale = angleInDegrees ? static_cast<T>(CV_PI / 180.0) : static_cast<T>(1.0);

        polarToCartInterleavedImpl_<T> << <grid, block, 0, stream >> >(magAngle, xy, scale);
    }
}

void cv::cuda::polarToCart(InputArray _mag, InputArray _angle, OutputArray _x, OutputArray _y, bool angleInDegrees, Stream& _stream)
{
    typedef void(*func_t)(const GpuMat& mag, const GpuMat& angle, GpuMat& x, GpuMat& y, bool angleInDegrees, cudaStream_t& stream);
    static const func_t funcs[7] = { 0, 0, 0, 0, 0, polarToCartImpl<float>, polarToCartImpl<double> };

    GpuMat mag = getInputMat(_mag, _stream);
    GpuMat angle = getInputMat(_angle, _stream);

    CV_Assert(angle.depth() == CV_32F || angle.depth() == CV_64F);
    CV_Assert( mag.empty() || (mag.type() == angle.type() && mag.size() == angle.size()) );

    GpuMat x = getOutputMat(_x, angle.size(), CV_MAKETYPE(angle.depth(), 1), _stream);
    GpuMat y = getOutputMat(_y, angle.size(), CV_MAKETYPE(angle.depth(), 1), _stream);

    cudaStream_t stream = StreamAccessor::getStream(_stream);
    funcs[angle.depth()](mag, angle, x, y, angleInDegrees, stream);
    CV_CUDEV_SAFE_CALL( cudaGetLastError() );

    syncOutput(x, _x, _stream);
    syncOutput(y, _y, _stream);

    if (stream == 0)
        CV_CUDEV_SAFE_CALL( cudaDeviceSynchronize() );
}

void cv::cuda::polarToCart(InputArray _mag, InputArray _angle, OutputArray _xy, bool angleInDegrees, Stream& _stream)
{
    typedef void(*func_t)(const GpuMat& mag, const GpuMat& angle, GpuMat& xy, bool angleInDegrees, cudaStream_t& stream);
    static const func_t funcs[7] = { 0, 0, 0, 0, 0, polarToCartDstInterleavedImpl<float>, polarToCartDstInterleavedImpl<double> };

    GpuMat mag = getInputMat(_mag, _stream);
    GpuMat angle = getInputMat(_angle, _stream);

    CV_Assert(angle.depth() == CV_32F || angle.depth() == CV_64F);
    CV_Assert( mag.empty() || (mag.type() == angle.type() && mag.size() == angle.size()) );

    GpuMat xy = getOutputMat(_xy, angle.size(), CV_MAKETYPE(angle.depth(), 2), _stream);

    cudaStream_t stream = StreamAccessor::getStream(_stream);
    funcs[angle.depth()](mag, angle, xy, angleInDegrees, stream);
    CV_CUDEV_SAFE_CALL( cudaGetLastError() );

    syncOutput(xy, _xy, _stream);

    if (stream == 0)
        CV_CUDEV_SAFE_CALL( cudaDeviceSynchronize() );
}

void cv::cuda::polarToCart(InputArray _magAngle, OutputArray _xy, bool angleInDegrees, Stream& _stream)
{
    typedef void(*func_t)(const GpuMat& magAngle, GpuMat& xy, bool angleInDegrees, cudaStream_t& stream);
    static const func_t funcs[7] = { 0, 0, 0, 0, 0, polarToCartInterleavedImpl<float>, polarToCartInterleavedImpl<double> };

    GpuMat magAngle = getInputMat(_magAngle, _stream);

    CV_Assert(magAngle.type() == CV_32FC2 || magAngle.type() == CV_64FC2);

    GpuMat xy = getOutputMat(_xy, magAngle.size(), magAngle.type(), _stream);

    cudaStream_t stream = StreamAccessor::getStream(_stream);
    funcs[magAngle.depth()](magAngle, xy, angleInDegrees, stream);
    CV_CUDEV_SAFE_CALL( cudaGetLastError() );

    syncOutput(xy, _xy, _stream);

    if (stream == 0)
        CV_CUDEV_SAFE_CALL( cudaDeviceSynchronize() );
}

#endif
