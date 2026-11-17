// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Optional native CUDA public-call timing on the same device/workload subset.
// Explicit stream, preallocated output, 20 warmups, and call+completion timing
// match the Triton rows. Sampling/rounding may differ; this is not an accuracy oracle.
#include "native_cuda_precomp.hpp"
namespace opencv_test
{
namespace
{

PERF_TEST_P(Sz_Interp_Dtype, NativeCudaWarpAffine_Reference, warpPerfCases())
{
    const cv::Size size = GET_PARAM(0);
    const int geom = GET_PARAM(1);
    const int depth = GET_PARAM(2);

    const int type = CV_MAKETYPE(depth, 3);
    cv::Mat host(size, type);
    declare.in(host, WARMUP_RNG);
    const cv::Mat M = warpPerfMatrix(geom);

    const cv::cuda::GpuMat src(host);
    cv::cuda::Stream stream; // explicit non-default stream
    cv::cuda::GpuMat dst(size, type);

    for (int i = 0; i < 20; ++i)
        cv::cuda::warpAffine(src, dst, M, size, GET_PARAM(3), cv::BORDER_REPLICATE, cv::Scalar(),
                             stream);
    stream.waitForCompletion();

    TEST_CYCLE()
    {
        cv::cuda::warpAffine(src, dst, M, size, GET_PARAM(3), cv::BORDER_REPLICATE, cv::Scalar(),
                             stream);
        stream.waitForCompletion();
    }

    CUDA_SANITY_CHECK(dst, 1e-10);
}

}
} // namespace
