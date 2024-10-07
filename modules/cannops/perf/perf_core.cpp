// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"
#include "opencv2/cann_interface.hpp"

namespace opencv_test
{
namespace
{
#define TYPICAL_ASCEND_MAT_SIZES \
    Values(::perf::sz1080p, ::perf::sz2K, ::perf::sz2160p, ::perf::sz4320p)
#define DVPP_ASCEND_MAT_SIZES Values(::perf::sz1080p, ::perf::sz2K, ::perf::sz2160p, ::perf::sz5MP)
#define DEF_PARAM_TEST(name, ...) \
    typedef ::perf::TestBaseWithParam<testing::tuple<__VA_ARGS__>> name

DEF_PARAM_TEST(NPU, Size);
DEF_PARAM_TEST(CPU, Size);

PERF_TEST_P(NPU, MERGE, TYPICAL_ASCEND_MAT_SIZES)
{
    Mat mat(GET_PARAM(0), CV_8UC1);
    Mat dst;
    declare.in(mat, WARMUP_RNG);
    cv::cann::setDevice(DEVICE_ID);
    AscendMat ascendMat[3];
    ascendMat[0].upload(mat);
    ascendMat[1].upload(mat);
    ascendMat[2].upload(mat);

    TEST_CYCLE() { cv::cann::merge(&ascendMat[0], 3, dst); }
    cv::cann::resetDevice();
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(CPU, MERGE, TYPICAL_ASCEND_MAT_SIZES)
{
    Mat mat(GET_PARAM(0), CV_8UC1);
    Mat dst;
    declare.in(mat, WARMUP_RNG);
    Mat mats[3] = {mat, mat, mat};
    TEST_CYCLE() { cv::merge(&mats[0], 3, dst); }
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(NPU, SPLIT, TYPICAL_ASCEND_MAT_SIZES)
{
    Mat mat(GET_PARAM(0), CV_8UC3);
    declare.in(mat, WARMUP_RNG);
    cv::cann::setDevice(DEVICE_ID);
    AscendMat ascendMat[3];

    TEST_CYCLE() { cv::cann::split(mat, &ascendMat[0]); }
    cv::cann::resetDevice();
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(CPU, SPLIT, TYPICAL_ASCEND_MAT_SIZES)
{
    Mat mat(GET_PARAM(0), CV_8UC3);
    declare.in(mat, WARMUP_RNG);
    Mat mats[3] = {mat, mat, mat};
    TEST_CYCLE() { cv::split(mat, &mats[0]); }
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(NPU, TRANSPOSE, TYPICAL_ASCEND_MAT_SIZES)
{
    Mat mat(GET_PARAM(0), CV_8UC3);
    Mat dst;
    declare.in(mat, WARMUP_RNG);
    cv::cann::setDevice(DEVICE_ID);
    TEST_CYCLE() { cv::cann::transpose(mat, dst); }
    cv::cann::resetDevice();
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(CPU, TRANSPOSE, TYPICAL_ASCEND_MAT_SIZES)
{
    Mat mat(GET_PARAM(0), CV_8UC3);
    Mat dst;
    declare.in(mat, WARMUP_RNG);
    TEST_CYCLE() { cv::transpose(mat, dst); }
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(NPU, FLIP, TYPICAL_ASCEND_MAT_SIZES)
{
    Mat mat(GET_PARAM(0), CV_8UC3);
    Mat dst;
    declare.in(mat, WARMUP_RNG);
    cv::cann::setDevice(DEVICE_ID);
    TEST_CYCLE() { cv::cann::flip(mat, dst, -1); }
    cv::cann::resetDevice();
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(CPU, FLIP, TYPICAL_ASCEND_MAT_SIZES)
{
    Mat mat(GET_PARAM(0), CV_8UC3);
    Mat dst;
    declare.in(mat, WARMUP_RNG);
    TEST_CYCLE() { cv::flip(mat, dst, -1); }
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(NPU, ROTATE, TYPICAL_ASCEND_MAT_SIZES)
{
    Mat mat(GET_PARAM(0), CV_8UC3);
    Mat dst;
    declare.in(mat, WARMUP_RNG);
    cv::cann::setDevice(DEVICE_ID);
    TEST_CYCLE() { cv::cann::rotate(mat, dst, 1); }
    cv::cann::resetDevice();
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(CPU, ROTATE, TYPICAL_ASCEND_MAT_SIZES)
{
    Mat mat(GET_PARAM(0), CV_8UC3);
    Mat dst;
    declare.in(mat, WARMUP_RNG);
    TEST_CYCLE() { cv::rotate(mat, dst, 1); }
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(NPU, CROP, TYPICAL_ASCEND_MAT_SIZES)
{
    Mat mat(GET_PARAM(0), CV_8UC3);
    Mat dst;
    declare.in(mat, WARMUP_RNG);
    Rect b(1, 2, 4, 4);
    cv::cann::setDevice(DEVICE_ID);
    TEST_CYCLE() { AscendMat cropped_cann(mat, b); }
    cv::cann::resetDevice();
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(CPU, CROP, TYPICAL_ASCEND_MAT_SIZES)
{
    Mat mat(GET_PARAM(0), CV_8UC3);
    Mat dst;
    declare.in(mat, WARMUP_RNG);
    Rect b(1, 2, 4, 4);
    TEST_CYCLE() { Mat cropped_cv(mat, b); }
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(NPU, CROP_OVERLOAD, TYPICAL_ASCEND_MAT_SIZES)
{
    Mat mat(GET_PARAM(0), CV_8UC3);
    Mat dst;
    declare.in(mat, WARMUP_RNG);
    Rect b(1, 2, 4, 4);
    cv::cann::setDevice(DEVICE_ID);
    TEST_CYCLE() { cv::cann::crop(mat, b); }
    cv::cann::resetDevice();
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(CPU, RESIZE, DVPP_ASCEND_MAT_SIZES)
{
    Mat mat(GET_PARAM(0), CV_8UC3);
    Mat dst;
    declare.in(mat, WARMUP_RNG);
    Size dsize = Size(256, 256);
    TEST_CYCLE_N(10) { cv::resize(mat, dst, dsize, 0, 0, 1); }
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(NPU, RESIZE, DVPP_ASCEND_MAT_SIZES)
{
    Mat mat(GET_PARAM(0), CV_32FC3);
    AscendMat dst;
    AscendMat src;
    src.upload(mat);
    declare.in(mat, WARMUP_RNG);
    Size dsize = Size(256, 256);
    TEST_CYCLE_N(10) { cv::cann::resize(src, dst, dsize, 0, 0, 3); }
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(NPU, THRESHOLD, TYPICAL_ASCEND_MAT_SIZES)
{
    Mat mat(GET_PARAM(0), CV_32FC3);
    AscendMat dst;
    AscendMat src;
    src.upload(mat);
    declare.in(mat, WARMUP_RNG);
    TEST_CYCLE_N(10) { cv::cann::threshold(src, dst, 100.0, 255.0, cv::THRESH_BINARY); }
    SANITY_CHECK_NOTHING();
}
PERF_TEST_P(CPU, THRESHOLD, TYPICAL_ASCEND_MAT_SIZES)
{
    Mat mat(GET_PARAM(0), CV_32FC3);
    Mat dst;
    declare.in(mat, WARMUP_RNG);
    TEST_CYCLE_N(10) { cv::threshold(mat, dst, 100.0, 255.0, cv::THRESH_BINARY); }
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(NPU, RESIZE_INTER_NEAREST, DVPP_ASCEND_MAT_SIZES)
{
    Mat mat(GET_PARAM(0), CV_8UC3);
    Mat dst;
    declare.in(mat, WARMUP_RNG);
    Size dsize = Size(256, 256);
    TEST_CYCLE_N(10) { cv::cann::resize(mat, dst, dsize, 0, 0, 0); }
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(NPU, COPY_MAKE_BORDER, DVPP_ASCEND_MAT_SIZES)
{
    Mat resized_cv, checker, cpuOpRet, cpuMat(GET_PARAM(0), CV_8UC3);
    declare.in(cpuMat, WARMUP_RNG);
    int top, bottom, left, right;
    top = (int)(20);
    bottom = top;
    left = (int)(20);
    right = left;
    int borderType = 1;
    float scalarV[3] = {0, 0, 255};
    Scalar value = {scalarV[0], scalarV[1], scalarV[2]};

    TEST_CYCLE_N(10)
    {
        cv::cann::copyMakeBorder(cpuMat, checker, top, bottom, left, right, borderType, value);
    }

    SANITY_CHECK_NOTHING();
}
PERF_TEST_P(CPU, COPY_MAKE_BORDER, DVPP_ASCEND_MAT_SIZES)
{
    Mat resized_cv, checker, cpuOpRet, cpuMat(GET_PARAM(0), CV_8UC3);
    declare.in(cpuMat, WARMUP_RNG);
    int top, bottom, left, right;
    top = (int)(20);
    bottom = top;
    left = (int)(20);
    right = left;
    int borderType = 1;
    float scalarV[3] = {0, 0, 255};
    Scalar value = {scalarV[0], scalarV[1], scalarV[2]};

    TEST_CYCLE_N(10)
    {
        cv::copyMakeBorder(cpuMat, checker, top, bottom, left, right, borderType, value);
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(NPU, CROP_RESIZE_MAKE_BORDER, DVPP_ASCEND_MAT_SIZES)
{
    Size size = GET_PARAM(0);
    Mat resized_cv, checker, cpuOpRet, cpuMat(size, CV_8UC3);
    declare.in(cpuMat, WARMUP_RNG);

    const Rect b(1, 0, size.width / 2, size.height);
    Size dsize = Size(size.width / 4, size.height / 2);
    int top, left;
    top = (int)(20);
    left = (int)(20);
    int borderType = 0;
    float scalarV[3] = {1, 1, 1};
    Scalar value = {scalarV[0], scalarV[1], scalarV[2]};

    TEST_CYCLE_N(10)
    {
        cv::cann::cropResizeMakeBorder(cpuMat, checker, b, dsize, 0, 0, 1, top, left, borderType,
                                       value);
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(CPU, CROP_RESIZE_MAKE_BORDER, DVPP_ASCEND_MAT_SIZES)
{
    Size size = GET_PARAM(0);
    Mat resized_cv, checker, cpuOpRet, cpuMat(size, CV_8UC3);
    declare.in(cpuMat, WARMUP_RNG);
    const Rect b(1, 0, size.width / 2, size.height);
    Size dsize = Size(size.width / 4, size.height / 2);
    int top, bottom, left, right;
    top = (int)(20);
    bottom = 0;
    left = (int)(20);
    right = 0;
    int borderType = 0;
    float scalarV[3] = {1, 1, 1};
    Scalar value = {scalarV[0], scalarV[1], scalarV[2]};

    TEST_CYCLE_N(10)
    {
        Mat cropped_cv(cpuMat, b);
        cv::resize(cropped_cv, resized_cv, dsize, 0, 0, 1);
        cv::copyMakeBorder(resized_cv, cpuOpRet, top, bottom, left, right, borderType, value);
    }
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(NPU, CROP_RESIZE, DVPP_ASCEND_MAT_SIZES)
{
    Size size = GET_PARAM(0);
    Mat resized_cv, checker, cpuOpRet, cpuMat(size, CV_8UC3);
    declare.in(cpuMat, WARMUP_RNG);
    const Rect b(1, 0, size.width / 2, size.height);
    Size dsize = Size(size.width / 4, size.height / 2);

    TEST_CYCLE_N(10) { cv::cann::cropResize(cpuMat, checker, b, dsize, 0, 0, 1); }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(CPU, CROP_RESIZE, DVPP_ASCEND_MAT_SIZES)
{
    Size size = GET_PARAM(0);
    Mat resized_cv, checker, cpuOpRet, cpuMat(size, CV_8UC3);
    declare.in(cpuMat, WARMUP_RNG);
    const Rect b(1, 0, size.width / 2, size.height);
    Size dsize = Size(size.width / 4, size.height / 2);

    TEST_CYCLE_N(10)
    {
        Mat cropped_cv(cpuMat, b);
        cv::resize(cropped_cv, resized_cv, dsize, 0, 0, 1);
    }
    SANITY_CHECK_NOTHING();
}

} // namespace
} // namespace opencv_test
