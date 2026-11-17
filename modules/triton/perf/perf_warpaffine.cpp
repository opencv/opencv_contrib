// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Warmed public-call timing with optional same-device native CUDA comparisons.
// The selected CPU reference and cv::cuda::warpAffine use different sampling
// arithmetic; timing ratios do not imply numerical equivalence.
// Compilation, first load and output allocation occur before the timed cycle.
// Public submission and stream completion are included. Host overhead can
// dominate short operations; these rows are not isolated kernel timings.
#include "perf_cases.hpp"
namespace opencv_test
{
namespace
{

// Typed so a 16-bit row is visible in the output name; the depth is one value per run rather
// than a range, because a perf row should answer one question.
PERF_TEST_P(Sz_Interp_Dtype, TritonWarpAffine, warpPerfCases())
{
    const cv::Size size = GET_PARAM(0);
    const int geom = GET_PARAM(1);
    const int depth = GET_PARAM(2);

    if (!triton_validation::available())
        throw SkipTestException("no Triton backend available");

    const int type = CV_MAKETYPE(depth, 3);
    cv::Mat host(size, type);
    declare.in(host, WARMUP_RNG);
    const cv::Mat M = warpPerfMatrix(geom);

    triton_validation::Image src;
    src.upload(host);
    triton_validation::Stream stream;         // explicit non-default stream
    triton_validation::Image dst(size, type); // preallocated: reused, excluded from the cycle

    for (int i = 0; i < 20; ++i)
        triton_validation::warpAffine(src, dst, M, size, GET_PARAM(3), cv::BORDER_REPLICATE,
                                      cv::Scalar(), stream);
    stream.wait();

    TEST_CYCLE()
    {
        triton_validation::warpAffine(src, dst, M, size, GET_PARAM(3), cv::BORDER_REPLICATE,
                                      cv::Scalar(), stream);
        stream.wait();
    }

    const cv::Mat result = dst.download();
    SANITY_CHECK(result, 1e-10);
}

// AREA aliases LINEAR. Transparent reads existing dst, so these routes have their
// own rows rather than implying an equivalent native CUDA comparison. The initial
// background is defined before warmup; subsequent calls reuse its current contents.
typedef ::perf::TestBaseWithParam<testing::tuple<cv::Size, int, int, int>> Sz_Interp_Border_Type;

PERF_TEST_P(Sz_Interp_Border_Type, TritonWarpAffineAreaTransparent,
            testing::Combine(
                testing::Values(cv::Size(1920, 1080), cv::Size(3840, 2160)),
                testing::Values((int)cv::INTER_NEAREST, (int)cv::INTER_LINEAR, (int)cv::INTER_AREA),
                testing::Values((int)cv::BORDER_REPLICATE, (int)cv::BORDER_TRANSPARENT),
                testing::Values(CV_8UC3, CV_16UC1, CV_32FC1)))
{
    const cv::Size size = GET_PARAM(0);
    const int interp = GET_PARAM(1), border = GET_PARAM(2), type = GET_PARAM(3);
    if (!triton_validation::available())
        throw SkipTestException("no Triton backend available");

    cv::Mat host(size, type);
    declare.in(host, WARMUP_RNG);
    const cv::Mat M = warpPerfMatrix(2);
    triton_validation::Image src;
    src.upload(host);
    triton_validation::Stream stream; // explicit non-default stream
    triton_validation::Image dst;
    dst.upload(host);
    for (int i = 0; i < 20; ++i)
        triton_validation::warpAffine(src, dst, M, size, interp, border, cv::Scalar(), stream);
    stream.wait();

    TEST_CYCLE()
    {
        triton_validation::warpAffine(src, dst, M, size, interp, border, cv::Scalar(), stream);
        stream.wait();
    }
    const cv::Mat result = dst.download();
    SANITY_CHECK(result, 1e-10);
}

}
} // namespace
