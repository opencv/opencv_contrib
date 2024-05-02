// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include <iostream>

namespace opencv_test
{
namespace
{
template <typename FCV, typename FCANN, typename... PARAMS>
void testMatOpMat(FCV cvFunc, FCANN cannFunc, PARAMS... param)
{
    cv::cann::setDevice(DEVICE_ID);
    Mat mat1 = randomMat(10, 10, CV_32SC3);
    Mat mat2 = randomMat(10, 10, CV_32SC3);
    Mat cpuDst, check;

    cvFunc(mat1, mat2, cpuDst, param...);
    cannFunc(mat1, mat2, check, param..., AscendStream::Null());
    EXPECT_MAT_NEAR(cpuDst, check, 1.0);

    AscendStream stream;
    cannFunc(mat1, mat2, check, param..., stream);
    stream.waitForCompletion();
    EXPECT_MAT_NEAR(cpuDst, check, 1.0);

    cv::cann::resetDevice();
}

template <typename FCV, typename FCANN, typename DTMASK, typename... PARAMS>
void testAscendMatOpAscendMatMask(FCV cvFunc, FCANN cannFunc, DTMASK mask = AscendMat(),
                                  PARAMS... param)
{
    cv::cann::setDevice(DEVICE_ID);
    Mat mat1 = randomMat(10, 10, CV_32SC3);
    Mat mat2 = randomMat(10, 10, CV_32SC3);
    Mat cpuDst, check, cpuMask;
    AscendMat npuMat1, npuMat2, npuCheck;
    npuMat1.upload(mat1);
    npuMat2.upload(mat2);
    if (mask.empty())
    {
        cvFunc(mat1, mat2, cpuDst, noArray(), param...);
    }
    else
    {
        mask.download(cpuMask);
        cvFunc(mat1, mat2, cpuDst, cpuMask, param...);
    }

    cannFunc(npuMat1, npuMat2, npuCheck, mask, param..., AscendStream::Null());
    npuCheck.download(check);
    EXPECT_MAT_NEAR(cpuDst, check, 1.0);

    AscendStream stream;
    cannFunc(npuMat1, npuMat2, npuCheck, mask, param..., stream);
    npuCheck.download(check);
    stream.waitForCompletion();
    EXPECT_MAT_NEAR(cpuDst, check, 1.0);

    cv::cann::resetDevice();
}

template <typename FCV, typename FCANN, typename... PARAMS>
void testAscendMatOpAscendMat(FCV cvFunc, FCANN cannFunc, PARAMS... param)
{
    cv::cann::setDevice(DEVICE_ID);
    Mat mat1 = randomMat(10, 10, CV_32SC3);
    Mat mat2 = randomMat(10, 10, CV_32SC3);
    Mat cpuDst, check;
    AscendMat npuMat1, npuMat2, npuCheck;
    npuMat1.upload(mat1);
    npuMat2.upload(mat2);
    cvFunc(mat1, mat2, cpuDst, param...);
    cannFunc(npuMat1, npuMat2, npuCheck, param..., AscendStream::Null());
    npuCheck.download(check);
    EXPECT_MAT_NEAR(cpuDst, check, 1.0);

    AscendStream stream;
    cannFunc(npuMat1, npuMat2, npuCheck, param..., stream);
    npuCheck.download(check);
    stream.waitForCompletion();
    EXPECT_MAT_NEAR(cpuDst, check, 1.0);

    cv::cann::resetDevice();
}

TEST(ELEMENTWISE_OP, MAT_ADD_MAT)
{
    testMatOpMat(
        cv::add,
        [](const InputArray src1, const InputArray src2, OutputArray dst, const InputArray mask,
           int dtype, AscendStream& stream)
        { cv::cann::add(src1, src2, dst, mask, dtype, stream); },
        noArray(), -1);
    testAscendMatOpAscendMatMask(
        cv::add,
        [](const AscendMat& src1, const AscendMat& src2, AscendMat& dst, const AscendMat& mask,
           int dtype, AscendStream& stream)
        { cv::cann::add(src1, src2, dst, mask, dtype, stream); },
        AscendMat(), -1);
}

TEST(ELEMENTWISE_OP, MAT_SUB_MAT)
{
    testMatOpMat(
        cv::subtract,
        [](const InputArray src1, const InputArray src2, OutputArray dst, const InputArray mask,
           int dtype, AscendStream& stream)
        { cv::cann::subtract(src1, src2, dst, mask, dtype, stream); },
        noArray(), -1);
    testAscendMatOpAscendMatMask(
        cv::subtract,
        [](const AscendMat& src1, const AscendMat& src2, AscendMat& dst, const AscendMat& mask,
           int dtype, AscendStream& stream)
        { cv::cann::subtract(src1, src2, dst, mask, dtype, stream); },
        AscendMat(), -1);
}

TEST(ELEMENTWISE_OP, MAT_MUL_MAT)
{
    testMatOpMat(
        cv::multiply,
        [](const InputArray src1, const InputArray src2, OutputArray dst, float scale, int dtype,
           AscendStream& stream) { cv::cann::multiply(src1, src2, dst, scale, dtype, stream); },
        1, -1);
    testAscendMatOpAscendMat(
        cv::multiply,
        [](const AscendMat& src1, const AscendMat& src2, AscendMat& dst, float scale, int dtype,
           AscendStream& stream) { cv::cann::multiply(src1, src2, dst, scale, dtype, stream); },
        1, -1);
}

TEST(ELEMENTWISE_OP, MAT_DIV_MAT)
{
    testMatOpMat([](const cv::Mat& src1, const cv::Mat& src2, cv::Mat& dst, double scale, int dtype)
                 { cv::divide(src1, src2, dst, scale, dtype); },
                 [](const InputArray src1, const InputArray src2, OutputArray dst, float scale,
                    int dtype, AscendStream& stream)
                 { cv::cann::divide(src1, src2, dst, scale, dtype, stream); },
                 1, -1);
    testAscendMatOpAscendMat(
        [](const cv::Mat& src1, const cv::Mat& src2, cv::Mat& dst, double scale, int dtype)
        { cv::divide(src1, src2, dst, scale, dtype); },
        [](const AscendMat& src1, const AscendMat& src2, AscendMat& dst, float scale, int dtype,
           AscendStream& stream) { cv::cann::divide(src1, src2, dst, scale, dtype, stream); },
        1, -1);
}

TEST(ELEMENTWISE_OP, MAT_BITWISE_AND_MAT)
{
    testMatOpMat(
        cv::bitwise_and,
        [](const InputArray src1, const InputArray src2, OutputArray dst, const InputArray mask,
           AscendStream& stream) { cv::cann::bitwise_and(src1, src2, dst, mask, stream); },
        noArray());
    testAscendMatOpAscendMatMask(
        cv::bitwise_and,
        [](const AscendMat& src1, const AscendMat& src2, AscendMat& dst, const AscendMat& mask,
           AscendStream& stream) { cv::cann::bitwise_and(src1, src2, dst, mask, stream); },
        AscendMat());
}

TEST(ELEMENTWISE_OP, MAT_BITWISE_OR_MAT)
{
    testMatOpMat(
        cv::bitwise_or,
        [](const InputArray src1, const InputArray src2, OutputArray dst, const InputArray mask,
           AscendStream& stream) { cv::cann::bitwise_or(src1, src2, dst, mask, stream); },
        noArray());
    testAscendMatOpAscendMatMask(
        cv::bitwise_or,
        [](const AscendMat& src1, const AscendMat& src2, AscendMat& dst, const AscendMat& mask,
           AscendStream& stream) { cv::cann::bitwise_or(src1, src2, dst, mask, stream); },
        AscendMat());
}

TEST(ELEMENTWISE_OP, MAT_BITWISE_XOR_MAT)
{
    testMatOpMat(
        cv::bitwise_xor,
        [](const InputArray src1, const InputArray src2, OutputArray dst, const InputArray mask,
           AscendStream& stream) { cv::cann::bitwise_xor(src1, src2, dst, mask, stream); },
        noArray());
    testAscendMatOpAscendMatMask(
        cv::bitwise_xor,
        [](const AscendMat& src1, const AscendMat& src2, AscendMat& dst, const AscendMat& mask,
           AscendStream& stream) { cv::cann::bitwise_xor(src1, src2, dst, mask, stream); },
        AscendMat());
}

TEST(ELEMENTWISE_OP, MAT_ADD_MAT_WITH_MASK_AND_DTYPE)
{
    testMatOpMat(
        cv::add,
        [](const InputArray src1, const InputArray src2, OutputArray dst, const InputArray mask,
           int dtype, AscendStream& stream)
        { cv::cann::add(src1, src2, dst, mask, dtype, stream); },
        genMask(), CV_32SC3);
    testAscendMatOpAscendMatMask(
        cv::add,
        [](const AscendMat& src1, const AscendMat& src2, AscendMat& dst, const AscendMat& mask,
           int dtype, AscendStream& stream)
        { cv::cann::add(src1, src2, dst, mask, dtype, stream); },
        AscendMat(), CV_32SC3);
}

TEST(ELEMENTWISE_OP, MAT_SUB_MAT_WITH_MASK_AND_DTYPE)
{
    testMatOpMat(
        cv::subtract,
        [](const InputArray src1, const InputArray src2, OutputArray dst, const InputArray mask,
           int dtype, AscendStream& stream)
        { cv::cann::subtract(src1, src2, dst, mask, dtype, stream); },
        genMask(), CV_32SC3);
    testAscendMatOpAscendMatMask(
        cv::subtract,
        [](const AscendMat& src1, const AscendMat& src2, AscendMat& dst, const AscendMat& mask,
           int dtype, AscendStream& stream)
        { cv::cann::subtract(src1, src2, dst, mask, dtype, stream); },
        AscendMat(), CV_32SC3);
}

TEST(ELEMENTWISE_OP, MAT_BITWISE_AND_MAT_WITH_MASK)
{
    testMatOpMat(
        cv::bitwise_and,
        [](const InputArray src1, const InputArray src2, OutputArray dst, const InputArray mask,
           AscendStream& stream) { cv::cann::bitwise_and(src1, src2, dst, mask, stream); },
        genMask());
    testAscendMatOpAscendMatMask(
        cv::bitwise_and,
        [](const AscendMat& src1, const AscendMat& src2, AscendMat& dst, const AscendMat& mask,
           AscendStream& stream) { cv::cann::bitwise_and(src1, src2, dst, mask, stream); },
        genNpuMask());
}

TEST(ELEMENTWISE_OP, MAT_BITWISE_OR_MAT_WITH_MASK)
{
    testMatOpMat(
        cv::bitwise_or,
        [](const InputArray src1, const InputArray src2, OutputArray dst, const InputArray mask,
           AscendStream& stream) { cv::cann::bitwise_or(src1, src2, dst, mask, stream); },
        genMask());
    testAscendMatOpAscendMatMask(
        cv::bitwise_or,
        [](const AscendMat& src1, const AscendMat& src2, AscendMat& dst, const AscendMat& mask,
           AscendStream& stream) { cv::cann::bitwise_or(src1, src2, dst, mask, stream); },
        genNpuMask());
}

TEST(ELEMENTWISE_OP, MAT_BITWISE_XOR_MAT_WITH_MASK)
{
    testMatOpMat(
        cv::bitwise_xor,
        [](const InputArray src1, const InputArray src2, OutputArray dst, const InputArray mask,
           AscendStream& stream) { cv::cann::bitwise_xor(src1, src2, dst, mask, stream); },
        genMask());
    testAscendMatOpAscendMatMask(
        cv::bitwise_xor,
        [](const AscendMat& src1, const AscendMat& src2, AscendMat& dst, const AscendMat& mask,
           AscendStream& stream) { cv::cann::bitwise_xor(src1, src2, dst, mask, stream); },
        genNpuMask());
}

float randomScale = randomNum();
TEST(ELEMENTWISE_OP, MAT_MUL_MAT_WITH_SCALE)
{
    testMatOpMat(
        cv::multiply,
        [](const InputArray src1, const InputArray src2, OutputArray dst, float scale, int dtype,
           AscendStream& stream) { cv::cann::multiply(src1, src2, dst, scale, dtype, stream); },
        randomScale, -1);
    testAscendMatOpAscendMat(
        cv::multiply,
        [](const AscendMat& src1, const AscendMat& src2, AscendMat& dst, float scale, int dtype,
           AscendStream& stream) { cv::cann::multiply(src1, src2, dst, scale, dtype, stream); },
        randomScale, -1);
}

TEST(ELEMENTWISE_OP, MAT_DIV_MAT_WITH_SCALE)
{
    testMatOpMat([](const cv::Mat& src1, const cv::Mat& src2, cv::Mat& dst, double scale, int dtype)
                 { cv::divide(src1, src2, dst, scale, dtype); },
                 [](const InputArray src1, const InputArray src2, OutputArray dst, float scale,
                    int dtype, AscendStream& stream)
                 { cv::cann::divide(src1, src2, dst, scale, dtype, stream); },
                 randomScale, -1);
    testAscendMatOpAscendMat(
        [](const cv::Mat& src1, const cv::Mat& src2, cv::Mat& dst, double scale, int dtype)
        { cv::divide(src1, src2, dst, scale, dtype); },
        [](const AscendMat& src1, const AscendMat& src2, AscendMat& dst, float scale, int dtype,
           AscendStream& stream) { cv::cann::divide(src1, src2, dst, scale, dtype, stream); },
        randomScale, -1);
}

template <typename FCV, typename FCANN, typename... PARAMS>
void testMatOpScalar(FCV cvFunc, FCANN cannFunc, PARAMS... param)
{
    Scalar scalar = randomScalar();
    Mat mat(10, 10, CV_32SC3, randomScalar());
    Mat cpuDst1, cpuDst2, checker1, checker2;

    cvFunc(Mat(10, 10, CV_32SC3, scalar), mat, cpuDst1, param...);
    cvFunc(mat, Mat(10, 10, CV_32SC3, scalar), cpuDst2, param...);
    cv::cann::setDevice(DEVICE_ID);

    cannFunc(scalar, mat, checker1, param..., AscendStream::Null());
    cannFunc(mat, scalar, checker2, param..., AscendStream::Null());

    EXPECT_MAT_NEAR(cpuDst1, checker1, 1.0);
    EXPECT_MAT_NEAR(cpuDst2, checker2, 1.0);

    AscendStream stream;
    cannFunc(scalar, mat, checker1, param..., stream);
    cannFunc(mat, scalar, checker2, param..., stream);
    stream.waitForCompletion();
    EXPECT_MAT_NEAR(cpuDst1, checker1, 1.0);
    EXPECT_MAT_NEAR(cpuDst2, checker2, 1.0);

    cv::cann::resetDevice();
}

template <typename FCV, typename FCANN, typename DTMASK, typename... PARAMS>
void testAscendMatOpScalarMask(FCV cvFunc, FCANN cannFunc, DTMASK mask, PARAMS... param)
{
    Scalar scalar = randomScalar();
    Mat mat(10, 10, CV_32SC3, randomScalar());
    Mat cpuDst, checker, cpuMask;
    AscendMat npuMat, npuChecker;
    npuMat.upload(mat);
    if (mask.empty())
    {
        cvFunc(mat, Mat(10, 10, CV_32SC3, scalar), cpuDst, noArray(), param...);
    }
    else
    {
        mask.download(cpuMask);
        cvFunc(mat, Mat(10, 10, CV_32SC3, scalar), cpuDst, cpuMask, param...);
    }
    cv::cann::setDevice(DEVICE_ID);

    cannFunc(npuMat, scalar, npuChecker, mask, param..., AscendStream::Null());
    npuChecker.download(checker);
    EXPECT_MAT_NEAR(cpuDst, checker, 1.0);

    AscendStream stream;
    cannFunc(npuMat, scalar, npuChecker, mask, param..., stream);
    stream.waitForCompletion();
    npuChecker.download(checker);
    EXPECT_MAT_NEAR(cpuDst, checker, 1.0);

    cv::cann::resetDevice();
}
template <typename FCV, typename FCANN, typename DTMASK, typename... PARAMS>
void testScalarOpAscendMatMask(FCV cvFunc, FCANN cannFunc, DTMASK mask, PARAMS... param)
{
    Scalar scalar = randomScalar();
    Mat mat(10, 10, CV_32SC3, randomScalar());
    Mat cpuDst, checker, cpuMask;
    AscendMat npuMat, npuChecker;
    npuMat.upload(mat);
    if (mask.empty())
    {
        cvFunc(Mat(10, 10, CV_32SC3, scalar), mat, cpuDst, noArray(), param...);
    }
    else
    {
        mask.download(cpuMask);
        cvFunc(Mat(10, 10, CV_32SC3, scalar), mat, cpuDst, cpuMask, param...);
    }
    cv::cann::setDevice(DEVICE_ID);

    cannFunc(scalar, npuMat, npuChecker, mask, param..., AscendStream::Null());
    npuChecker.download(checker);
    EXPECT_MAT_NEAR(cpuDst, checker, 1.0);

    AscendStream stream;
    cannFunc(scalar, npuMat, npuChecker, mask, param..., stream);
    stream.waitForCompletion();
    npuChecker.download(checker);
    EXPECT_MAT_NEAR(cpuDst, checker, 1.0);

    cv::cann::resetDevice();
}
template <typename FCV, typename FCANN, typename... PARAMS>
void testAscendMatOpScalar(FCV cvFunc, FCANN cannFunc, PARAMS... param)
{
    Scalar scalar = randomScalar();
    Mat mat(10, 10, CV_32SC3, randomScalar());
    Mat cpuDst, checker;
    AscendMat npuMat, npuChecker;
    npuMat.upload(mat);

    cvFunc(mat, Mat(10, 10, CV_32SC3, scalar), cpuDst, param...);
    cv::cann::setDevice(DEVICE_ID);

    cannFunc(npuMat, scalar, npuChecker, param..., AscendStream::Null());
    npuChecker.download(checker);
    EXPECT_MAT_NEAR(cpuDst, checker, 1.0);

    AscendStream stream;
    cannFunc(npuMat, scalar, npuChecker, param..., stream);
    stream.waitForCompletion();
    npuChecker.download(checker);

    cv::cann::resetDevice();
}

TEST(ELEMENTWISE_OP, MAT_ADD_SCALAR)
{
    testMatOpScalar(
        cv::add,
        [](const InputArray src1, const InputArray src2, OutputArray dst, const InputArray mask,
           int dtype, AscendStream& stream)
        { cv::cann::add(src1, src2, dst, mask, dtype, stream); },
        noArray(), -1);
    testAscendMatOpScalarMask(
        cv::add,
        [](const AscendMat& src1, const Scalar& src2, AscendMat& dst, const AscendMat& mask,
           int dtype, AscendStream& stream)
        { cv::cann::add(src1, src2, dst, mask, dtype, stream); },
        AscendMat(), -1);
    testScalarOpAscendMatMask(
        cv::add,
        [](const Scalar& src1, const AscendMat& src2, AscendMat& dst, const AscendMat& mask,
           int dtype, AscendStream& stream)
        { cv::cann::add(src1, src2, dst, mask, dtype, stream); },
        AscendMat(), -1);
}

TEST(ELEMENTWISE_OP, MAT_SUB_SCALAR)
{
    testMatOpScalar(
        cv::subtract,
        [](const InputArray src1, const InputArray src2, OutputArray dst, const InputArray mask,
           int dtype, AscendStream& stream)
        { cv::cann::subtract(src1, src2, dst, mask, dtype, stream); },
        noArray(), -1);
    testAscendMatOpScalarMask(
        cv::subtract,
        [](const AscendMat& src1, const Scalar& src2, AscendMat& dst, const AscendMat& mask,
           int dtype, AscendStream& stream)
        { cv::cann::subtract(src1, src2, dst, mask, dtype, stream); },
        AscendMat(), -1);
}

TEST(ELEMENTWISE_OP, MAT_MUL_SCALAR)
{
    testMatOpScalar(
        cv::multiply,
        [](const InputArray src1, const InputArray src2, OutputArray dst, float scale, int dtype,
           AscendStream& stream) { cv::cann::multiply(src1, src2, dst, scale, dtype, stream); },
        1, -1);
    testAscendMatOpScalar(
        cv::multiply,
        [](const AscendMat& src1, const Scalar& src2, AscendMat& dst, float scale, int dtype,
           AscendStream& stream) { cv::cann::multiply(src1, src2, dst, scale, dtype, stream); },
        1, -1);
}

TEST(ELEMENTWISE_OP, MAT_DIV_SCALAR)
{
    testMatOpScalar(
        [](const cv::Mat& src1, const cv::Mat& src2, cv::Mat& dst, double scale, int dtype)
        { cv::divide(src1, src2, dst, scale, dtype); },
        [](const InputArray src1, const InputArray src2, OutputArray dst, float scale, int dtype,
           AscendStream& stream) { cv::cann::divide(src1, src2, dst, scale, dtype, stream); },
        1, -1);
    testAscendMatOpScalar(
        [](const cv::Mat& src1, const cv::Mat& src2, cv::Mat& dst, double scale, int dtype)
        { cv::divide(src1, src2, dst, scale, dtype); },
        [](const AscendMat& src1, const Scalar& src2, AscendMat& dst, float scale, int dtype,
           AscendStream& stream) { cv::cann::divide(src1, src2, dst, scale, dtype, stream); },
        1, -1);
}

TEST(ELEMENTWISE_OP, MAT_BITWISE_AND_SCALAR)
{
    testMatOpScalar(
        cv::bitwise_and,
        [](const InputArray src1, const InputArray src2, OutputArray dst, const InputArray mask,
           AscendStream& stream) { cv::cann::bitwise_and(src1, src2, dst, mask, stream); },
        noArray());
    testAscendMatOpScalarMask(
        cv::bitwise_and,
        [](const AscendMat& src1, const Scalar& src2, AscendMat& dst, const AscendMat& mask,
           AscendStream& stream) { cv::cann::bitwise_and(src1, src2, dst, mask, stream); },
        AscendMat());
}

TEST(ELEMENTWISE_OP, MAT_BITWISE_OR_SCALAR)
{
    testMatOpScalar(
        cv::bitwise_or,
        [](const InputArray src1, const InputArray src2, OutputArray dst, const InputArray mask,
           AscendStream& stream) { cv::cann::bitwise_or(src1, src2, dst, mask, stream); },
        noArray());
    testAscendMatOpScalarMask(
        cv::bitwise_or,
        [](const AscendMat& src1, const Scalar& src2, AscendMat& dst, const AscendMat& mask,
           AscendStream& stream) { cv::cann::bitwise_or(src1, src2, dst, mask, stream); },
        AscendMat());
}

TEST(ELEMENTWISE_OP, MAT_BITWISE_XOR_SCALAR)
{
    testMatOpScalar(
        cv::bitwise_xor,
        [](const InputArray src1, const InputArray src2, OutputArray dst, const InputArray mask,
           AscendStream& stream) { cv::cann::bitwise_xor(src1, src2, dst, mask, stream); },
        noArray());
    testAscendMatOpScalarMask(
        cv::bitwise_xor,
        [](const AscendMat& src1, const Scalar& src2, AscendMat& dst, const AscendMat& mask,
           AscendStream& stream) { cv::cann::bitwise_xor(src1, src2, dst, mask, stream); },
        AscendMat());
}

TEST(ELEMENTWISE_OP, MAT_ADD_SCALAR_WITH_MASK_AND_DETYPE)
{
    testMatOpScalar(
        cv::add,
        [](const InputArray src1, const InputArray src2, OutputArray dst, const InputArray mask,
           int dtype, AscendStream& stream)
        { cv::cann::add(src1, src2, dst, mask, dtype, stream); },
        genMask(), CV_32SC3);
    testAscendMatOpScalarMask(
        cv::add,
        [](const AscendMat& src1, const Scalar& src2, AscendMat& dst, const AscendMat& mask,
           int dtype, AscendStream& stream)
        { cv::cann::add(src1, src2, dst, mask, dtype, stream); },
        genNpuMask(), CV_32SC3);
}

TEST(ELEMENTWISE_OP, MAT_SUB_SCALAR_WITH_MASK_AND_DETYPE)
{
    testMatOpScalar(
        cv::subtract,
        [](const InputArray src1, const InputArray src2, OutputArray dst, const InputArray mask,
           int dtype, AscendStream& stream)
        { cv::cann::subtract(src1, src2, dst, mask, dtype, stream); },
        genMask(), CV_32SC3);
    testAscendMatOpScalarMask(
        cv::subtract,
        [](const AscendMat& src1, const Scalar& src2, AscendMat& dst, const AscendMat& mask,
           int dtype, AscendStream& stream)
        { cv::cann::subtract(src1, src2, dst, mask, dtype, stream); },
        genNpuMask(), CV_32SC3);
}

TEST(ELEMENTWISE_OP, MAT_BITWISE_AND_SCALAR_WITH_MASK)
{
    testMatOpScalar(
        cv::bitwise_and,
        [](const InputArray src1, const InputArray src2, OutputArray dst, const InputArray mask,
           AscendStream& stream) { cv::cann::bitwise_and(src1, src2, dst, mask, stream); },
        genMask());
    testAscendMatOpScalarMask(
        cv::bitwise_and,
        [](const AscendMat& src1, const Scalar& src2, AscendMat& dst, const AscendMat& mask,
           AscendStream& stream) { cv::cann::bitwise_and(src1, src2, dst, mask, stream); },
        genNpuMask());
}

TEST(ELEMENTWISE_OP, MAT_BITWISE_OR_SCALAR_WITH_MASK)
{
    testMatOpScalar(
        cv::bitwise_or,
        [](const InputArray src1, const InputArray src2, OutputArray dst, const InputArray mask,
           AscendStream& stream) { cv::cann::bitwise_or(src1, src2, dst, mask, stream); },
        genMask());
    testAscendMatOpScalarMask(
        cv::bitwise_or,
        [](const AscendMat& src1, const Scalar& src2, AscendMat& dst, const AscendMat& mask,
           AscendStream& stream) { cv::cann::bitwise_or(src1, src2, dst, mask, stream); },
        genNpuMask());
}

TEST(ELEMENTWISE_OP, MAT_BITWISE_XOR_SCALAR_WITH_MASK)
{
    testMatOpScalar(
        cv::bitwise_xor,
        [](const InputArray src1, const InputArray src2, OutputArray dst, const InputArray mask,
           AscendStream& stream) { cv::cann::bitwise_xor(src1, src2, dst, mask, stream); },
        genMask());
    testAscendMatOpScalarMask(
        cv::bitwise_xor,
        [](const AscendMat& src1, const Scalar& src2, AscendMat& dst, const AscendMat& mask,
           AscendStream& stream) { cv::cann::bitwise_xor(src1, src2, dst, mask, stream); },
        genNpuMask());
}

// TODO: I think the cv result is wrong, which has truncated middle result.
// Disable these two test case bacause it't not stable.
// TEST(ELEMENTWISE_OP, MAT_MUL_SCALAR_WITH_SCALE)
// {
//     testMatOpScalar(
//         cv::multiply,
//         [](const InputArray src1, const InputArray src2, OutputArray dst, float scale, int dtype,
//            AscendStream& stream) { cv::cann::multiply(src1, src2, dst, scale, dtype, stream); },
//         randomScale, CV_32SC3);
//     testAscendMatOpScalar(
//         [](const cv::Mat& src1, const cv::Mat& src2, cv::Mat& dst, double scale, int dtype)
//         { cv::divide(src1, src2, dst, scale, dtype); },
//         [](const AscendMat& src1, const Scalar& src2, AscendMat& dst, float scale, int dtype,
//            AscendStream& stream) { cv::cann::divide(src1, src2, dst, scale, dtype, stream); },
//         randomScale, -1);
// }

// TEST(ELEMENTWISE_OP, MAT_DIV_SCALAR_WITH_SCALE)
// {
//     testMatOpScalar(
//         [](const cv::Mat& src1, const cv::Mat& src2, cv::Mat& dst, double scale, int dtype)
//         { cv::divide(src1, src2, dst, scale, dtype); },
//         [](const InputArray src1, const InputArray src2, OutputArray dst, float scale, int dtype,
//            AscendStream& stream) { cv::cann::divide(src1, src2, dst, scale, dtype, stream); },
//         randomScale, -1);
//     testAscendMatOpScalar(
//         [](const cv::Mat& src1, const cv::Mat& src2, cv::Mat& dst, double scale, int dtype)
//         { cv::divide(src1, src2, dst, scale, dtype); },
//         [](const AscendMat& src1, const Scalar& src2, AscendMat& dst, float scale, int dtype,
//            AscendStream& stream) { cv::cann::divide(src1, src2, dst, scale, dtype, stream); },
//         randomScale, -1);
// }

TEST(ELEMENTWISE_OP, MAT_BITWISE_NOT)
{
    Mat cpuOpRet, checker, cpuMat = randomMat(10, 10, CV_32SC3);
    cv::cann::setDevice(DEVICE_ID);
    cv::bitwise_not(cpuMat, cpuOpRet);
    cv::cann::bitwise_not(cpuMat, checker);
    EXPECT_MAT_NEAR(cpuOpRet, checker, 0.0);

    AscendMat npuMat, npuOpRet;
    npuMat.upload(cpuMat);
    cv::cann::bitwise_not(npuMat, npuOpRet);
    npuOpRet.download(checker);
    EXPECT_MAT_NEAR(cpuOpRet, checker, 0.0);

    cv::cann::resetDevice();
}

// TODO random test matrix
TEST(ELEMENTWISE_OP, MAT_ADD_WEIGHTED)
{
    Mat cpuOpRet, checker, cpuMat1 = Mat::ones(5, 5, CV_32S), cpuMat2 = Mat::ones(5, 5, CV_32S);

    cv::cann::setDevice(DEVICE_ID);
    cv::addWeighted(cpuMat1, 2, cpuMat2, 3, 5, cpuOpRet);
    cv::cann::addWeighted(cpuMat1, 2, cpuMat2, 3, 5, checker);
    EXPECT_MAT_NEAR(cpuOpRet, checker, 0.0);

    AscendMat npuOpRet, npuMat1, npuMat2;
    npuMat1.upload(cpuMat1);
    npuMat2.upload(cpuMat2);
    cv::cann::addWeighted(npuMat1, 2, npuMat2, 3, 5, npuOpRet);
    npuOpRet.download(checker);
    EXPECT_MAT_NEAR(cpuOpRet, checker, 0.0);

    cv::cann::resetDevice();
}

TEST(ELEMENTWISE_OP, MAT_THRESHOLD)
{
    Mat cpuOpRet, checker, cpuMat = randomMat(10, 10, CV_16SC3, 0.0, 255.0);

    AscendMat ascendMat, ascendMat16F, aclOpRet, aclOpRet16S;
    cv::cann::setDevice(DEVICE_ID);
    ascendMat.upload(cpuMat);
    ascendMat.convertTo(ascendMat16F, CV_16F);

    Mat cpuMat16F, checker16F;
    cpuMat.convertTo(cpuMat16F, CV_16F);

    for (int i = 0; i <= 4; i++)
    {
        cv::threshold(cpuMat, cpuOpRet, 128, 250, i);
        cv::cann::threshold(ascendMat16F, aclOpRet, 128, 250, i);
        aclOpRet.convertTo(aclOpRet16S, CV_16S);
        aclOpRet16S.download(checker);

        EXPECT_MAT_NEAR(cpuOpRet, checker, 1e-10);

        cv::cann::threshold(cpuMat16F, checker16F, 128, 250, i);
        checker16F.convertTo(checker, CV_16S);
        EXPECT_MAT_NEAR(cpuOpRet, checker, 1e-10);
    }

    cv::cann::resetDevice();
}

TEST(ELEMENTWISE_OP, MAT_THRESHOLD_ASCENDC)
{
    cv::cann::setDevice(DEVICE_ID);
    Mat cpuRet, npuRet;
    AscendMat npuImg, npuTmpMat;

    // opencv do not support CV_8S, CV_32S, CV_16F
    // ascend do not support CV_16U, CV_64F
    uint8_t dtypes[] = {CV_8U, CV_16S, CV_32F};

    for (uint i = 0; i <= 4; i++)
    {
        for (uint j = 0; j < sizeof(dtypes) / sizeof(dtypes[0]); j++)
        {
            double thresh = 90.5;
            double maxVal = 85.2;

            Mat img = randomMat(10, 10, CV_MAKETYPE(dtypes[j], 3), 0.0f, 128.0f);
            npuImg.upload(img);
            npuTmpMat.create(npuImg.rows, npuImg.cols, npuImg.type());

            cv::threshold(img, cpuRet, thresh, maxVal, i);
            cv::cann::threshold(npuImg, npuTmpMat, thresh, maxVal, i);

            npuTmpMat.download(npuRet);
            EXPECT_MAT_NEAR(cpuRet, npuRet, 10.0f);
        }
    }

    cv::cann::resetDevice();
}

} // namespace
} // namespace opencv_test
