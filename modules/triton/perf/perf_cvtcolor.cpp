// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Warmed public-call timing with optional same-device native CUDA comparisons.
// Native comparison rows are absent on HIP.
//
// Selected and extended rows cover only their named conversions and depths.
// The optional cv::cuda baseline is identified separately; its presence does not
// establish numerical equivalence. Measurements include public submission and
// stream completion, not pure device time.
//
// Compile and first-load costs are excluded by warming up before the measured cycle, and the
// destination is preallocated so allocation is not in the timed region.
//
// Host submission overhead can dominate short operations. These public-call
// timings do not isolate device execution time.
#include "perf_cases.hpp"
#include "../validation/cvtcolor_cases.hpp"
#include "../validation/reference.hpp"
namespace opencv_test
{
namespace
{

// Seven selected codes with distinct cost structures; this is not coverage of every
// supported code/depth. All candidate paths use the same private validation adapter on
// each backend; the optional native comparison in baseline/ is specifically CUDA.
const int kCodes[] = {
    cv::COLOR_BGR2GRAY, cv::COLOR_BGRA2GRAY, cv::COLOR_BGR2YUV,  cv::COLOR_BGR2YCrCb,
    cv::COLOR_BGR2HSV,  cv::COLOR_BGR2BGRA,  cv::COLOR_BGRA2BGR,
};

PERF_TEST_P(Sz_Code, TritonCvtColor, testing::Combine(colorPerfSizes(), testing::ValuesIn(kCodes)))
{
    const cv::Size size = GET_PARAM(0);
    const int code = GET_PARAM(1);

    if (!triton_validation::available())
        throw SkipTestException("no Triton backend available");

    const auto *row = triton_test::cvt_cases::find(code, CV_8U);
    CV_Assert(row != nullptr);
    cv::Mat host(size, CV_8UC(row->scn));
    declare.in(host, WARMUP_RNG);

    triton_validation::Image src;
    src.upload(host);
    triton_validation::Stream stream;                     // explicit non-default stream
    triton_validation::Image dst(size, CV_8UC(row->dcn)); // preallocated: reusable, excluded

    for (int i = 0; i < 20; ++i)
        triton_validation::cvtColor(src, dst, code, stream);
    stream.wait();

    TEST_CYCLE()
    {
        triton_validation::cvtColor(src, dst, code, stream);
        stream.wait();
    }

    const cv::Mat result = dst.download();
    ASSERT_TRUE(triton_test::meetsCvtContract(result, triton_test::cpuCvt(host, row->code), host,
                                              triton_test::cvt_cases::floatArithmetic(row->code),
                                              triton_test::cvt_cases::floatArithmetic(row->code) &&
                                                  cv_triton_color::float_cpu_separate4()));
    SANITY_CHECK(result, 1e-10);
}

// Representative depth/layout families; the accuracy tests enumerate all
// 124 admitted code/depth pairs. Keep extended rows separate from the optional
// native CUDA reference, whose supported conversions and semantics differ.
struct ExtendedCvt
{
    int code, depth;
};
static const ExtendedCvt kExtended[] = {
    {cv::COLOR_BGR2GRAY, CV_32F},    {cv::COLOR_BGR2YUV, CV_32F},
    {cv::COLOR_BGR2BGRA, CV_32F},    {cv::COLOR_BGR2GRAY, CV_16U},
    {cv::COLOR_BGR2YCrCb, CV_16U},   {cv::COLOR_GRAY2BGRA, CV_16U},
    {cv::COLOR_BayerBG2BGR, CV_16U}, {cv::COLOR_BGR5652BGRA, CV_8U},
    {cv::COLOR_BayerBG2GRAY, CV_8U}, {cv::COLOR_BGR2XYZ, CV_8U},
    {cv::COLOR_BGR2Lab, CV_8U},      {cv::COLOR_BGR2Luv, CV_8U},
    {cv::COLOR_Luv2BGR, CV_8U},      {cv::COLOR_YUV2BGR_UYVY, CV_8U},
    {cv::COLOR_YUV2BGR_NV12, CV_8U}, {cv::COLOR_YUV2BGRA_NV21, CV_8U}};

typedef ::perf::TestBaseWithParam<testing::tuple<cv::Size, int>> Sz_ExtendedCvt;
PERF_TEST_P(Sz_ExtendedCvt, TritonCvtColorExtended,
            testing::Combine(testing::Values(cv::Size(1920, 1080), cv::Size(3840, 2160)),
                             testing::Range(0, static_cast<int>(sizeof(kExtended) /
                                                                sizeof(kExtended[0])))))
{
    if (!triton_validation::available())
        throw SkipTestException("no Triton backend available");
    const cv::Size size = GET_PARAM(0);
    const ExtendedCvt &selected = kExtended[GET_PARAM(1)];
    const auto *row = triton_test::cvt_cases::find(selected.code, selected.depth);
    CV_Assert(row != nullptr);
    cv::Mat host(row->yuv420 ? size.height / 2 * 3 : size.height, size.width,
                 CV_MAKETYPE(row->depth, row->scn));
    declare.in(host, WARMUP_RNG);
    triton_validation::Image src;
    src.upload(host);
    triton_validation::Image dst(size, CV_MAKETYPE(row->depth, row->dcn));
    triton_validation::Stream stream; // explicit non-default stream
    for (int i = 0; i < 20; ++i)
        triton_validation::cvtColor(src, dst, row->code, stream);
    stream.wait();
    TEST_CYCLE()
    {
        triton_validation::cvtColor(src, dst, row->code, stream);
        stream.wait();
    }
    const cv::Mat result = dst.download();
    ASSERT_TRUE(triton_test::meetsCvtContract(result, triton_test::cpuCvt(host, row->code), host,
                                              triton_test::cvt_cases::floatArithmetic(row->code),
                                              triton_test::cvt_cases::floatArithmetic(row->code) &&
                                                  cv_triton_color::float_cpu_separate4()));
    SANITY_CHECK(result, 1e-10);
}

}
} // namespace
