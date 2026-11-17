// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright 2026, The FlagOS Contributors.
// The GpuMat entries of opencv2/triton.hpp on a build without cv::cuda (TRITON_BACKEND=hip).
// Compiled INSTEAD of cuda_resize_adapter.cpp, never alongside it: both define the same
// symbols, so the public API is identical across backends.
//
// cv::cuda::GpuMat and cv::cuda::Stream exist as types on every OpenCV build, but without
// WITH_CUDA their methods throw and Stream carries no device stream, so these entries cannot
// run here. They raise StsNotImplemented and name the entries that do run on this backend
// (opencv2/triton/device.hpp). haveBackend() answers for the backend this module WAS built
// for, so a caller can tell "built for hip, device present" from "no device".
#include "precomp.hpp"
#include "opencv2/triton.hpp"
#include "opencv2/core/private.hpp"
#include "runtime/runtime.hpp"
namespace cv
{
namespace triton
{

namespace
{

[[noreturn]] void unavailable(const char *op)
{
    CV_Error(
        Error::StsNotImplemented,
        cv::format(
            "cv::triton::%s (GpuMat overload) needs cv::cuda, which this build does not "
            "have (backend %s). Use the cv::triton::%s DeviceImage overload from opencv2/triton/device.hpp "
            "with device buffers and a %s stream instead.",
            op, runtime::name(), op, runtime::name()));
}

} // namespace

bool haveBackend()
{
    // Query logical device 0 after the runtime's visibility filters.
    return runtime::deviceAvailable(0);
}

String backendName()
{
    return haveBackend() ? String(runtime::name()) : String();
}

void releaseGraphResources()
{
    detail::releaseCapturePins();
}

void resize(InputArray, OutputArray, Size, double, double, int, cuda::Stream &)
{
    unavailable("resize");
}
void resize(const cuda::GpuMat &, cuda::GpuMat &, Size, double, double, int, cuda::Stream &)
{
    unavailable("resize");
}
void cvtColor(InputArray, OutputArray, int, cuda::Stream &)
{
    unavailable("cvtColor");
}
void cvtColor(const cuda::GpuMat &, cuda::GpuMat &, int, cuda::Stream &)
{
    unavailable("cvtColor");
}
void warpAffine(InputArray, OutputArray, InputArray, Size, int, int, const Scalar &, cuda::Stream &)
{
    unavailable("warpAffine");
}
void warpAffine(const cuda::GpuMat &, cuda::GpuMat &, InputArray, Size, int, int, const Scalar &,
                cuda::Stream &)
{
    unavailable("warpAffine");
}

}
} // namespace cv::triton
