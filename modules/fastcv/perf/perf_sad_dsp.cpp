/*
 * Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "perf_precomp.hpp"

namespace opencv_test {

typedef std::tuple<cv::Size /*srcSize*/> SumOfAbsDiffsPerfParams;
typedef perf::TestBaseWithParam<SumOfAbsDiffsPerfParams> SumOfAbsDiffsPerfTest;

PERF_TEST_P(SumOfAbsDiffsPerfTest, run,
    ::testing::Values(cv::Size(640, 480),  // VGA
        cv::Size(1280, 720),               // 720p
        cv::Size(1920, 1080))              // 1080p
)
{
    // Initialize FastCV DSP
    int initStatus = cv::fastcv::dsp::fastcvq6init();
    ASSERT_EQ(initStatus, 0) << "Failed to initialize FastCV DSP";

    auto p = GetParam();
    cv::Size srcSize = std::get<0>(p);

    RNG& rng = cv::theRNG();
    cv::Mat patch(8, 8, CV_8UC1);
    cv::Mat src(srcSize, CV_8UC1);
    cvtest::randUni(rng, patch, cv::Scalar::all(0), cv::Scalar::all(255));
    cvtest::randUni(rng, src, cv::Scalar::all(0), cv::Scalar::all(255));

    cv::Mat dst;
    while(next())
    {
        startTimer();
        cv::fastcv::dsp::sumOfAbsoluteDiffs(patch, src, dst);
        stopTimer();
    }
    SANITY_CHECK_NOTHING();
}

} // namespace
