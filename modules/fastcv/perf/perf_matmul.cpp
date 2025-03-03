/*
 * Copyright (c) 2024-2025 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "perf_precomp.hpp"

namespace opencv_test {

typedef std::tuple<int /*rows1*/, int /*cols1*/, int /*cols2*/> MatMulPerfParams;
typedef perf::TestBaseWithParam<MatMulPerfParams> MatMulPerfTest;

typedef std::tuple<int /*rows1*/, int /*cols1*/, int /*cols2*/, float> MatMulGemmPerfParams;
typedef perf::TestBaseWithParam<MatMulGemmPerfParams> MatMulGemmPerfTest;

PERF_TEST_P(MatMulPerfTest, run,
    ::testing::Combine(::testing::Values(8, 16, 128, 256), // rows1
                       ::testing::Values(8, 16, 128, 256), // cols1
                       ::testing::Values(8, 16, 128, 256)) // cols2
           )
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
    while(next())
    {
        startTimer();
        cv::fastcv::matmuls8s32(src1, src2, dst);
        stopTimer();
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(MatMulGemmPerfTest, run,
    ::testing::Combine(::testing::Values(8, 16, 128, 256), // rows1
                       ::testing::Values(8, 16, 128, 256), // cols1
                       ::testing::Values(8, 16, 128, 256), // cols2
                       ::testing::Values(2.5, 5.8))   // alpha
           )
{
    auto p = GetParam();
    int rows1 = std::get<0>(p);
    int cols1 = std::get<1>(p);
    int cols2 = std::get<2>(p);
    float alpha = std::get<3>(p);

    RNG& rng = cv::theRNG();
    Mat src1(rows1, cols1, CV_32FC1), src2(cols1, cols2, CV_32FC1);
    cvtest::randUni(rng, src1, Scalar::all(-128.0), Scalar::all(128.0));
    cvtest::randUni(rng, src2, Scalar::all(-128.0), Scalar::all(128.0));

    Mat dst;

    while (next())
    {
        startTimer();
        cv::fastcv::gemm(src1, src2, dst, alpha, noArray(), 0);
        stopTimer();
    }

    SANITY_CHECK_NOTHING();
}

} // namespace
