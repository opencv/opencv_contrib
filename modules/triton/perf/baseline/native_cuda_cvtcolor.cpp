// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Optional native CUDA public-call timing on the same device/workload subset.
// Explicit stream, preallocated output, 20 warmups, and call+completion timing
// match the Triton rows. Sampling/rounding may differ; this is not an accuracy oracle.
// Four existing codes are a comparison subset, not the CUDA support catalogue.
#include "native_cuda_precomp.hpp"
namespace opencv_test
{
namespace
{

PERF_TEST_P(Sz_Code, NativeCudaCvtColor_Reference,
            testing::Combine(colorPerfSizes(),
                             testing::Values((int)cv::COLOR_BGR2GRAY, (int)cv::COLOR_BGR2YUV,
                                             (int)cv::COLOR_BGR2YCrCb, (int)cv::COLOR_BGR2HSV)))
{
    const cv::Size size = GET_PARAM(0);
    const int code = GET_PARAM(1);

    cv::Mat host(size, CV_8UC(3));
    declare.in(host, WARMUP_RNG);

    const cv::cuda::GpuMat src(host);
    cv::cuda::Stream stream; // explicit non-default stream
    cv::cuda::GpuMat dst(size, CV_8UC(code == cv::COLOR_BGR2GRAY ? 1 : 3));

    for (int i = 0; i < 20; ++i)
        cv::cuda::cvtColor(src, dst, code, 0, stream);
    stream.waitForCompletion();

    TEST_CYCLE()
    {
        cv::cuda::cvtColor(src, dst, code, 0, stream);
        stream.waitForCompletion();
    }

    CUDA_SANITY_CHECK(dst, 1e-10);
}

}
} // namespace
