/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "perf_precomp.hpp"

namespace opencv_test {

typedef std::tuple<float /*sigmaColor*/, float /*sigmaSpace*/> BilateralRecursivePerfParams;
typedef perf::TestBaseWithParam<BilateralRecursivePerfParams> BilateralRecursivePerfTest;

PERF_TEST_P(BilateralRecursivePerfTest, run,
    ::testing::Combine(::testing::Values(0.01f, 0.03f, 0.1f, 1.f, 5.f),
                       ::testing::Values(0.01f, 0.05f, 0.1f, 1.f, 5.f))
           )
{
    auto p = GetParam();
    float sigmaColor = std::get<0>(p);
    float sigmaSpace = std::get<1>(p);

    cv::Mat src = imread(cvtest::findDataFile("cv/shared/baboon.png"), cv::IMREAD_GRAYSCALE);
    Mat dst;

    while(next())
    {
        startTimer();
        cv::fastcv::bilateralRecursive(src, dst, sigmaColor, sigmaSpace);
        stopTimer();
    }

    SANITY_CHECK_NOTHING();
}


typedef std::tuple<float /*sigmaColor*/, float /*sigmaSpace*/, cv::Size, int > BilateralPerfParams;
typedef perf::TestBaseWithParam<BilateralPerfParams> BilateralPerfTest;


PERF_TEST_P(BilateralPerfTest, run,
    ::testing::Combine(::testing::Values(0.01f, 0.03f, 0.1f, 1.f, 5.f),
                       ::testing::Values(0.01f, 0.05f, 0.1f, 1.f, 5.f),
                       ::testing::Values(Size(8, 8), Size(640, 480), Size(800, 600)),
                       ::testing::Values(5, 7, 9))
           )
{
    auto p = GetParam();
    float sigmaColor = std::get<0>(p);
    float sigmaSpace = std::get<1>(p);
    cv::Size size  = std::get<2>(p);
    int d = get<3>(p);

    RNG& rng = cv::theRNG();
    Mat src(size, CV_8UC1);
    cvtest::randUni(rng, src, Scalar::all(0), Scalar::all(255));
    Mat dst;

    while (next())
    {
        startTimer();
        cv::fastcv::bilateralFilter(src, dst, d, sigmaColor, sigmaSpace);
        stopTimer();
    }

    SANITY_CHECK_NOTHING();
}

} // namespace
