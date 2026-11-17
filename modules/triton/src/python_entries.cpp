// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright 2026, The FlagOS Contributors.
#include "precomp.hpp"
#include "opencv2/triton.hpp"
namespace cv
{
namespace triton
{

void resizeForPython(const cuda::GpuMat &src, cuda::GpuMat &dst, Size dsize, double fx, double fy,
                     int interpolation, Ptr<cuda::Stream> stream)
{
#ifdef OPENCV_TRITON_HAVE_CUDA_ADAPTER
    resize(src, dst, dsize, fx, fy, interpolation, stream ? *stream : cuda::Stream::Null());
#else
    CV_UNUSED(src);
    CV_UNUSED(dst);
    CV_UNUSED(dsize);
    CV_UNUSED(fx);
    CV_UNUSED(fy);
    CV_UNUSED(interpolation);
    CV_UNUSED(stream);
    CV_Error(
        Error::StsNotImplemented,
        "cv::triton::resize: this HIP build serves C++ device buffers; the Python GpuMat adapter needs CUDA");
#endif
}

void cvtColorForPython(const cuda::GpuMat &src, cuda::GpuMat &dst, int code,
                       Ptr<cuda::Stream> stream)
{
#ifdef OPENCV_TRITON_HAVE_CUDA_ADAPTER
    cvtColor(src, dst, code, stream ? *stream : cuda::Stream::Null());
#else
    CV_UNUSED(src);
    CV_UNUSED(dst);
    CV_UNUSED(code);
    CV_UNUSED(stream);
    CV_Error(
        Error::StsNotImplemented,
        "cv::triton::cvtColor: this HIP build serves C++ device buffers; the Python GpuMat adapter needs CUDA");
#endif
}

void warpAffineForPython(const cuda::GpuMat &src, cuda::GpuMat &dst, InputArray M, Size dsize,
                         int flags, int borderMode, const Scalar &borderValue,
                         Ptr<cuda::Stream> stream)
{
#ifdef OPENCV_TRITON_HAVE_CUDA_ADAPTER
    warpAffine(src, dst, M, dsize, flags, borderMode, borderValue,
               stream ? *stream : cuda::Stream::Null());
#else
    CV_UNUSED(src);
    CV_UNUSED(dst);
    CV_UNUSED(M);
    CV_UNUSED(dsize);
    CV_UNUSED(flags);
    CV_UNUSED(borderMode);
    CV_UNUSED(borderValue);
    CV_UNUSED(stream);
    CV_Error(
        Error::StsNotImplemented,
        "cv::triton::warpAffine: this HIP build serves C++ device buffers; the Python GpuMat adapter needs CUDA");
#endif
}

}
} // namespace cv::triton
