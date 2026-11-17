// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Warmed public-call timing, including submission and stream completion, not isolated
// kernel or graph time. Compilation, first load and reusable output allocation precede
// the timed cycle. Optional native CUDA rows live in baseline/.
// cv::cuda::resize can use different sampling/rounding from this module's CPU reference;
// a timing comparison therefore does not establish numerical equivalence.
#include "perf_cases.hpp"
namespace opencv_test
{
namespace
{

PERF_TEST_P(Sz_Dsize_Cn_Interp, TritonResize, resizePerfCases())
{
    const cv::Size srcSize = GET_PARAM(0);
    const cv::Size dsize = GET_PARAM(1);
    const int channels = GET_PARAM(2);
    const int interp = GET_PARAM(3);

    if (!triton_validation::available())
        throw SkipTestException("no Triton backend available");

    cv::Mat host(srcSize, CV_MAKETYPE(GET_PARAM(4), channels));
    declare.in(host, WARMUP_RNG);

    triton_validation::Image src;
    src.upload(host);
    triton_validation::Stream stream;                // explicit non-default stream
    triton_validation::Image dst(dsize, src.type()); // preallocated: reusable, excluded

    // Warm up compilation/module load and first-touch before measuring.
    for (int i = 0; i < 20; ++i)
        triton_validation::resize(src, dst, dsize, 0, 0, interp, stream);
    stream.wait();

    TEST_CYCLE()
    {
        triton_validation::resize(src, dst, dsize, 0, 0, interp, stream);
        stream.wait();
    }

    const cv::Mat result = dst.download();
    SANITY_CHECK(result, 1e-10);
}

// All four modes are exercised, including fractional AREA downsampling.

}
} // namespace
