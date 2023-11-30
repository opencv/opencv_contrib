// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"
#include "opencv2/cann_interface.hpp"

namespace opencv_test
{
namespace
{

#define CVT_COLORS_3                                                                         \
    Values(COLOR_BGR2BGRA, COLOR_BGRA2BGR, COLOR_BGR2RGBA, COLOR_RGBA2BGR, COLOR_BGR2RGB,    \
           COLOR_BGRA2RGBA, COLOR_BGR2GRAY, COLOR_BGRA2GRAY, COLOR_RGBA2GRAY, COLOR_BGR2XYZ, \
           COLOR_RGB2XYZ, COLOR_XYZ2BGR, COLOR_XYZ2RGB, COLOR_BGR2YCrCb, COLOR_RGB2YCrCb,    \
           COLOR_YCrCb2BGR, COLOR_YCrCb2RGB, COLOR_BGR2YUV, COLOR_RGB2YUV, COLOR_YUV2BGR,    \
           COLOR_YUV2RGB)
#define CVT_COLORS_1 Values(COLOR_GRAY2BGR, COLOR_GRAY2BGRA)
#define TYPICAL_ASCEND_MAT_SIZES \
    Values(::perf::sz1080p, ::perf::sz2K)
#define DEF_PARAM_TEST(name, ...) \
    typedef ::perf::TestBaseWithParam<testing::tuple<__VA_ARGS__>> name

DEF_PARAM_TEST(NPU, Size, ColorConversionCodes);
DEF_PARAM_TEST(CPU, Size, ColorConversionCodes);

PERF_TEST_P(NPU, CVT_COLOR_3, testing::Combine(TYPICAL_ASCEND_MAT_SIZES, CVT_COLORS_3))
{
    Mat mat(GET_PARAM(0), CV_32FC3);
    Mat dst;
    declare.in(mat, WARMUP_RNG);
    cv::cann::setDevice(DEVICE_ID);
    TEST_CYCLE() { cv::cann::cvtColor(mat, dst, GET_PARAM(1)); }
    cv::cann::resetDevice();
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(CPU, CVT_COLOR_3, testing::Combine(TYPICAL_ASCEND_MAT_SIZES, CVT_COLORS_3))
{
    Mat mat(GET_PARAM(0), CV_32FC3);
    Mat dst;
    declare.in(mat, WARMUP_RNG);
    TEST_CYCLE() { cv::cvtColor(mat, dst, GET_PARAM(1)); }
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(NPU, CVT_COLOR_1, testing::Combine(TYPICAL_ASCEND_MAT_SIZES, CVT_COLORS_1))
{
    Mat mat(GET_PARAM(0), CV_32FC1);
    Mat dst;
    declare.in(mat, WARMUP_RNG);
    cv::cann::setDevice(DEVICE_ID);
    TEST_CYCLE() { cv::cann::cvtColor(mat, dst, GET_PARAM(1)); }
    cv::cann::resetDevice();
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(CPU, CVT_COLOR_1, testing::Combine(TYPICAL_ASCEND_MAT_SIZES, CVT_COLORS_1))
{
    Mat mat(GET_PARAM(0), CV_32FC1);
    Mat dst;
    declare.in(mat, WARMUP_RNG);
    TEST_CYCLE() { cv::cvtColor(mat, dst, GET_PARAM(1)); }
    SANITY_CHECK_NOTHING();
}

} // namespace
} // namespace opencv_test
