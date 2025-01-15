/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "perf_precomp.hpp"

namespace opencv_test {

typedef perf::TestBaseWithParam<tuple<Size, int, int, int>> SobelPerfTest;

PERF_TEST_P(SobelPerfTest, run,
    ::testing::Combine(::testing::Values(perf::szVGA, perf::sz720p, perf::sz1080p), // image size
                       ::testing::Values(3,5,7),                                    // kernel size
                       ::testing::Values(BORDER_CONSTANT, BORDER_REPLICATE),        // border type
                       ::testing::Values(0)                                         // border value
                       )
           )
{
    Size srcSize = get<0>(GetParam());
    int ksize = get<1>(GetParam());
    int border = get<2>(GetParam());
    int borderValue = get<3>(GetParam());

    cv::Mat dx, dy, src(srcSize, CV_8U);
    RNG& rng = cv::theRNG();
    cvtest::randUni(rng, src, Scalar::all(0), Scalar::all(255));

    while (next())
    {
        startTimer();
        cv::fastcv::sobel(src,dx,dy,ksize,border,borderValue);
        stopTimer();
    }

    SANITY_CHECK_NOTHING();
}

typedef perf::TestBaseWithParam<tuple<Size, int, int>> Sobel3x3u8PerfTest;

PERF_TEST_P(Sobel3x3u8PerfTest, run,
    ::testing::Combine(::testing::Values(perf::szVGA, perf::sz720p, perf::sz1080p), // image size
                       ::testing::Values(CV_8S, CV_16S, CV_32F),                    // image depth
                       ::testing::Values(0, 1)                                      // normalization
                       )
           )
{
    Size srcSize = get<0>(GetParam());
    int ddepth = get<1>(GetParam());
    int normalization = get<2>(GetParam());

    cv::Mat dx, dy, src(srcSize, CV_8U);
    RNG& rng = cv::theRNG();
    cvtest::randUni(rng, src, Scalar::all(0), Scalar::all(255));

    if((normalization ==0) && (ddepth == CV_8S))
        throw ::perf::TestBase::PerfSkipTestException();

    while (next())
    {
        startTimer();
        cv::fastcv::sobel3x3u8(src, dx, dy, ddepth, normalization);
        stopTimer();
    }

    SANITY_CHECK_NOTHING();
}
} //namespace