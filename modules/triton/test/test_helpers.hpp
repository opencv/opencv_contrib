// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Shared by test_resize.cpp and test_runtime.cpp.
#ifndef OPENCV_TRITON_TEST_HELPERS_HPP
#define OPENCV_TRITON_TEST_HELPERS_HPP
#include "test_precomp.hpp"
#include "opencv2/ts/cuda_test.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/core/cuda_stream_accessor.hpp"
#include "opencv2/triton.hpp"
#include "../validation/reference.hpp"
#include "../validation/guards.hpp"
#include "resize_reference.hpp"
#include "../src/runtime/runtime.hpp"
namespace opencv_test
{
namespace triton_test
{

namespace td = cv::triton::detail;

/** A CUDA context must exist before anything device-side runs. */
inline bool ready()
{
    static const bool ok = []()
    {
        if (cv::cuda::getCudaEnabledDeviceCount() < 1)
            return false;
        cuda::GpuMat warm(4, 4, CV_8UC1);
        warm.setTo(Scalar::all(0));
        cuda::Stream::Null().waitForCompletion();
        return cv::triton::haveBackend();
    }();
    return ok;
}

/** Upload into a padded parent and return the ROI view; `parent` keeps the guard bands. */
inline cuda::GpuMat roiOf(cuda::GpuMat &parent, Size sz, int type, Point ofs, int padX, int padY,
                          double guard)
{
    parent.create(sz.height + padY, sz.width + padX, type);
    parent.setTo(Scalar::all(guard));
    return parent(Rect(ofs, sz));
}

/** Everything in `parent` outside `r` still equals `guard`. */
inline bool guardsIntact(const cuda::GpuMat &parent, Rect r, double guard)
{
    Mat m;
    parent.download(m);
    const Mat expected(m.size(), m.type(), Scalar::all(guard));
    return guardsEqual(expected, m, r);
}

}
} // namespace opencv_test::triton_test
#endif
