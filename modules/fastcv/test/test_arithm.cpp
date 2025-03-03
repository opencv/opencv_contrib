/*
 * Copyright (c) 2024-2025 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "test_precomp.hpp"

namespace opencv_test { namespace {

typedef std::tuple<int /*rows1*/, int /*cols1*/, int /*cols2*/> MatMulTestParams;
class MatMulTest : public ::testing::TestWithParam<MatMulTestParams> {};

typedef std::tuple<Size, int /*depth*/, int /*op type*/> ArithmOpTestParams;
class ArithmOpTest : public ::testing::TestWithParam<ArithmOpTestParams> {};

TEST_P(MatMulTest, accuracy)
{
    auto p = GetParam();
    int rows1 = std::get<0>(p);
    int cols1 = std::get<1>(p);
    int cols2 = std::get<2>(p);

    RNG& rng = cv::theRNG();
    Mat src1(rows1, cols1, CV_8SC1), src2(cols1, cols2, CV_8SC1);
    cvtest::randUni(rng, src1, Scalar::all(-128), Scalar::all(128));
    cvtest::randUni(rng, src2, Scalar::all(-128), Scalar::all(128));

    Mat dst;
    cv::fastcv::matmuls8s32(src1, src2, dst);
    Mat fdst;
    dst.convertTo(fdst, CV_32F);

    Mat fsrc1, fsrc2;
    src1.convertTo(fsrc1, CV_32F);
    src2.convertTo(fsrc2, CV_32F);
    Mat ref;
    cv::gemm(fsrc1, fsrc2, 1.0, noArray(), 0, ref, 0);

    double normInf = cvtest::norm(ref, fdst, cv::NORM_INF);
    double normL2  = cvtest::norm(ref, fdst, cv::NORM_L2);

    EXPECT_EQ(normInf, 0);
    EXPECT_EQ(normL2, 0);

    if (cvtest::debugLevel > 0 && (normInf > 0 || normL2 > 0))
    {
        std::ofstream of(cv::format("out_%d_%d_%d.txt", rows1, cols1, cols2));
        of << ref << std::endl;
        of << dst << std::endl;
        of.close();
    }
}

TEST_P(ArithmOpTest, accuracy)
{
    auto p = GetParam();
    Size sz = std::get<0>(p);
    int depth = std::get<1>(p);
    int op = std::get<2>(p);
    RNG& rng = cv::theRNG();
    Mat src1(sz, depth), src2(sz, depth);

    cvtest::randUni(rng, src1, Scalar::all(0), Scalar::all(128));
    cvtest::randUni(rng, src2, Scalar::all(0), Scalar::all(128));

    Mat dst;
    cv::fastcv::arithmetic_op(src1, src2, dst, op);

    Mat ref;
    if(op == 0)
        cv::add(src1, src2, ref);
    else if(op == 1)
        cv::subtract(src1, src2, ref);

    double normInf = cvtest::norm(ref, dst, cv::NORM_INF);
    double normL2  = cvtest::norm(ref, dst, cv::NORM_L2);

    EXPECT_EQ(normInf, 0);
    EXPECT_EQ(normL2, 0);
}

INSTANTIATE_TEST_CASE_P(FastCV_Extension, MatMulTest,
                         ::testing::Combine(::testing::Values(8, 16, 128, 256),   // rows1
                                            ::testing::Values(8, 16, 128, 256),   // cols1
                                            ::testing::Values(8, 16, 128, 256))); // cols2

INSTANTIATE_TEST_CASE_P(FastCV_Extension, ArithmOpTest,
                         ::testing::Combine(::testing::Values(perf::szVGA, perf::sz720p, perf::sz1080p),   // sz
                                            ::testing::Values(CV_8U, CV_16S), // depth
                                            ::testing::Values(0,1))); // op type

}} // namespaces opencv_test, ::
