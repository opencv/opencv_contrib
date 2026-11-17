// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright 2026, The FlagOS Contributors.
// Optional native CUDA public-call timing on the same device/workload subset.
// Explicit stream, preallocated output, 20 warmups, and call+completion timing
// match the Triton rows. Sampling/rounding may differ; this is not an accuracy oracle.
#include "native_cuda_precomp.hpp"
namespace opencv_test
{
namespace
{

PERF_TEST_P(Sz_Dsize_Cn_Interp, NativeCudaResize_Reference, resizePerfCases())
{
    const cv::Size srcSize = GET_PARAM(0);
    const cv::Size dsize = GET_PARAM(1);
    const int channels = GET_PARAM(2);
    const int interp = GET_PARAM(3);

    cv::Mat host(srcSize, CV_MAKETYPE(GET_PARAM(4), channels));
    declare.in(host, WARMUP_RNG);

    const cv::cuda::GpuMat src(host);
    cv::cuda::Stream stream; // explicit non-default stream
    cv::cuda::GpuMat dst(dsize, src.type());

    for (int i = 0; i < 20; ++i)
        cv::cuda::resize(src, dst, dsize, 0, 0, interp, stream);
    stream.waitForCompletion();

    TEST_CYCLE()
    {
        cv::cuda::resize(src, dst, dsize, 0, 0, interp, stream);
        stream.waitForCompletion();
    }

    CUDA_SANITY_CHECK(dst, 1e-10);
}

}
} // namespace
