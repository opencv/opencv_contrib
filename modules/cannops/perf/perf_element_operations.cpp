// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"
#include "opencv2/cann_interface.hpp"

namespace opencv_test
{
namespace
{

#define ARITHM_MAT_DEPTH Values(CV_32S, CV_32SC3)
#define TYPICAL_ASCEND_MAT_SIZES \
    Values(::perf::sz1080p, ::perf::sz2K, ::perf::sz2160p, ::perf::sz4320p)
#define DEF_PARAM_TEST(name, ...) \
    typedef ::perf::TestBaseWithParam<testing::tuple<__VA_ARGS__>> name

DEF_PARAM_TEST(NPU, Size, int);
DEF_PARAM_TEST(CPU, Size, int);

PERF_TEST_P(NPU, MAT_ADD_MAT, testing::Combine(TYPICAL_ASCEND_MAT_SIZES, ARITHM_MAT_DEPTH))
{
    Mat mat1(GET_PARAM(0), GET_PARAM(1));
    Mat mat2(GET_PARAM(0), GET_PARAM(1));
    Mat dst;
    declare.in(mat1, WARMUP_RNG);
    declare.in(mat2, WARMUP_RNG);
    cv::cann::setDevice(DEVICE_ID);
    TEST_CYCLE() { cv::cann::add(mat1, mat2, dst, noArray(), -1); }
    cv::cann::resetDevice();
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(CPU, MAT_ADD_MAT, testing::Combine(TYPICAL_ASCEND_MAT_SIZES, ARITHM_MAT_DEPTH))
{
    Mat mat1(GET_PARAM(0), GET_PARAM(1));
    Mat mat2(GET_PARAM(0), GET_PARAM(1));
    Mat dst;
    declare.in(mat1, WARMUP_RNG);
    declare.in(mat2, WARMUP_RNG);
    TEST_CYCLE() { cv::add(mat1, mat2, dst, noArray(), -1); }
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(NPU, MAT_SUB_MAT, testing::Combine(TYPICAL_ASCEND_MAT_SIZES, ARITHM_MAT_DEPTH))
{
    Mat mat1(GET_PARAM(0), GET_PARAM(1));
    Mat mat2(GET_PARAM(0), GET_PARAM(1));
    Mat dst;
    declare.in(mat1, WARMUP_RNG);
    declare.in(mat2, WARMUP_RNG);
    cv::cann::setDevice(DEVICE_ID);
    TEST_CYCLE() { cv::cann::subtract(mat1, mat2, dst, noArray(), -1); }
    cv::cann::resetDevice();
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(CPU, MAT_SUB_MAT, testing::Combine(TYPICAL_ASCEND_MAT_SIZES, ARITHM_MAT_DEPTH))
{
    Mat mat1(GET_PARAM(0), GET_PARAM(1));
    Mat mat2(GET_PARAM(0), GET_PARAM(1));
    Mat dst;
    declare.in(mat1, WARMUP_RNG);
    declare.in(mat2, WARMUP_RNG);
    TEST_CYCLE() { cv::subtract(mat1, mat2, dst, noArray(), -1); }
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(NPU, MAT_MUL_MAT, testing::Combine(TYPICAL_ASCEND_MAT_SIZES, ARITHM_MAT_DEPTH))
{
    Mat mat1(GET_PARAM(0), GET_PARAM(1));
    Mat mat2(GET_PARAM(0), GET_PARAM(1));
    Mat dst;
    declare.in(mat1, WARMUP_RNG);
    declare.in(mat2, WARMUP_RNG);
    cv::cann::setDevice(DEVICE_ID);
    TEST_CYCLE() { cv::cann::multiply(mat1, mat2, dst, 1, -1); }
    cv::cann::resetDevice();
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(CPU, MAT_MUL_MAT, testing::Combine(TYPICAL_ASCEND_MAT_SIZES, ARITHM_MAT_DEPTH))
{
    Mat mat1(GET_PARAM(0), GET_PARAM(1));
    Mat mat2(GET_PARAM(0), GET_PARAM(1));
    Mat dst;
    declare.in(mat1, WARMUP_RNG);
    declare.in(mat2, WARMUP_RNG);
    TEST_CYCLE() { cv::multiply(mat1, mat2, dst, 1, -1); }
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(NPU, MAT_DIV_MAT, testing::Combine(TYPICAL_ASCEND_MAT_SIZES, ARITHM_MAT_DEPTH))
{
    Mat mat1(GET_PARAM(0), GET_PARAM(1));
    Mat mat2(GET_PARAM(0), GET_PARAM(1));
    Mat dst;
    declare.in(mat1, WARMUP_RNG);
    declare.in(mat2, WARMUP_RNG);
    cv::cann::setDevice(DEVICE_ID);
    TEST_CYCLE() { cv::cann::divide(mat1, mat2, dst, 1, -1); }
    cv::cann::resetDevice();
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(CPU, MAT_DIV_MAT, testing::Combine(TYPICAL_ASCEND_MAT_SIZES, ARITHM_MAT_DEPTH))
{
    Mat mat1(GET_PARAM(0), GET_PARAM(1));
    Mat mat2(GET_PARAM(0), GET_PARAM(1));
    Mat dst;
    declare.in(mat1, WARMUP_RNG);
    declare.in(mat2, WARMUP_RNG);
    TEST_CYCLE() { cv::divide(mat1, mat2, dst, 1, -1); }
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(NPU, MAT_BITWISE_AND_MAT, testing::Combine(TYPICAL_ASCEND_MAT_SIZES, ARITHM_MAT_DEPTH))
{
    Mat mat1(GET_PARAM(0), GET_PARAM(1));
    Mat mat2(GET_PARAM(0), GET_PARAM(1));
    Mat dst;
    declare.in(mat1, WARMUP_RNG);
    declare.in(mat2, WARMUP_RNG);
    cv::cann::setDevice(DEVICE_ID);
    TEST_CYCLE() { cv::cann::bitwise_and(mat1, mat2, dst, noArray()); }
    cv::cann::resetDevice();
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(CPU, MAT_BITWISE_AND_MAT, testing::Combine(TYPICAL_ASCEND_MAT_SIZES, ARITHM_MAT_DEPTH))
{
    Mat mat1(GET_PARAM(0), GET_PARAM(1));
    Mat mat2(GET_PARAM(0), GET_PARAM(1));
    Mat dst;
    declare.in(mat1, WARMUP_RNG);
    declare.in(mat2, WARMUP_RNG);
    TEST_CYCLE() { cv::bitwise_and(mat1, mat2, dst, noArray()); }
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(NPU, MAT_BITWISE_OR_MAT, testing::Combine(TYPICAL_ASCEND_MAT_SIZES, ARITHM_MAT_DEPTH))
{
    Mat mat1(GET_PARAM(0), GET_PARAM(1));
    Mat mat2(GET_PARAM(0), GET_PARAM(1));
    Mat dst;
    declare.in(mat1, WARMUP_RNG);
    declare.in(mat2, WARMUP_RNG);
    cv::cann::setDevice(DEVICE_ID);
    TEST_CYCLE() { cv::cann::bitwise_or(mat1, mat2, dst, noArray()); }
    cv::cann::resetDevice();
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(CPU, MAT_BITWISE_OR_MAT, testing::Combine(TYPICAL_ASCEND_MAT_SIZES, ARITHM_MAT_DEPTH))
{
    Mat mat1(GET_PARAM(0), GET_PARAM(1));
    Mat mat2(GET_PARAM(0), GET_PARAM(1));
    Mat dst;
    declare.in(mat1, WARMUP_RNG);
    declare.in(mat2, WARMUP_RNG);
    TEST_CYCLE() { cv::bitwise_or(mat1, mat2, dst, noArray()); }
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(NPU, MAT_BITWISE_XOR_MAT, testing::Combine(TYPICAL_ASCEND_MAT_SIZES, ARITHM_MAT_DEPTH))
{
    Mat mat1(GET_PARAM(0), GET_PARAM(1));
    Mat mat2(GET_PARAM(0), GET_PARAM(1));
    Mat dst;
    declare.in(mat1, WARMUP_RNG);
    declare.in(mat2, WARMUP_RNG);
    cv::cann::setDevice(DEVICE_ID);
    TEST_CYCLE() { cv::cann::bitwise_xor(mat1, mat2, dst, noArray()); }
    cv::cann::resetDevice();
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(CPU, MAT_BITWISE_XOR_MAT, testing::Combine(TYPICAL_ASCEND_MAT_SIZES, ARITHM_MAT_DEPTH))
{
    Mat mat1(GET_PARAM(0), GET_PARAM(1));
    Mat mat2(GET_PARAM(0), GET_PARAM(1));
    Mat dst;
    declare.in(mat1, WARMUP_RNG);
    declare.in(mat2, WARMUP_RNG);
    TEST_CYCLE() { cv::bitwise_xor(mat1, mat2, dst, noArray()); }
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(NPU, MAT_BITWISE_NOT_MAT, testing::Combine(TYPICAL_ASCEND_MAT_SIZES, ARITHM_MAT_DEPTH))
{
    Mat mat(GET_PARAM(0), GET_PARAM(1));
    Mat dst;
    declare.in(mat, WARMUP_RNG);
    cv::cann::setDevice(DEVICE_ID);
    TEST_CYCLE() { cv::cann::bitwise_not(mat, dst, noArray()); }
    cv::cann::resetDevice();
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(CPU, MAT_BITWISE_NOT_MAT, testing::Combine(TYPICAL_ASCEND_MAT_SIZES, ARITHM_MAT_DEPTH))
{
    Mat mat(GET_PARAM(0), GET_PARAM(1));
    Mat dst;
    declare.in(mat, WARMUP_RNG);
    TEST_CYCLE() { cv::bitwise_not(mat, dst, noArray()); }
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(NPU, THRESHOLD_ASCENDC, testing::Combine(TYPICAL_ASCEND_MAT_SIZES,  Values(CV_8U, CV_16S, CV_32F)))
{
    Mat mat(GET_PARAM(0), GET_PARAM(1));
    AscendMat dst;
    AscendMat src;
    src.upload(mat);
    declare.in(mat, WARMUP_RNG);
    TEST_CYCLE_N(10) { cv::cann::threshold(src, dst, 100.0, 255.0, cv::THRESH_BINARY); }
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(CPU, THRESHOLD, testing::Combine(TYPICAL_ASCEND_MAT_SIZES, Values(CV_8U, CV_16S, CV_32F)))
{
    Mat mat(GET_PARAM(0), GET_PARAM(1));
    Mat dst;
    declare.in(mat, WARMUP_RNG);
    TEST_CYCLE_N(10) { cv::threshold(mat, dst, 100.0, 255.0, cv::THRESH_BINARY); }
    SANITY_CHECK_NOTHING();
}

} // namespace
} // namespace opencv_test
