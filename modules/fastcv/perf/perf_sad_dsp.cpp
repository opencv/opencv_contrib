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
    applyTestTag(CV_TEST_TAG_FASTCV_SKIP_DSP);

    // Initialize FastCV DSP
    int initStatus = cv::fastcv::dsp::fcvdspinit();
    ASSERT_EQ(initStatus, 0) << "Failed to initialize FastCV DSP";

    auto p = GetParam();
    cv::Size srcSize = std::get<0>(p);

    RNG& rng = cv::theRNG();
    cv::Mat patch, src;

    patch.allocator = cv::fastcv::getQcAllocator(); // Use FastCV allocator for patch
    src.allocator = cv::fastcv::getQcAllocator(); // Use FastCV allocator for src

    patch.create(8, 8, CV_8UC1);
    src.create(srcSize, CV_8UC1);

    cvtest::randUni(rng, patch, cv::Scalar::all(0), cv::Scalar::all(255));
    cvtest::randUni(rng, src, cv::Scalar::all(0), cv::Scalar::all(255));

    cv::Mat dst;
    dst.allocator = cv::fastcv::getQcAllocator(); // Use FastCV allocator for dst

    while(next())
    {
        startTimer();
        cv::fastcv::dsp::sumOfAbsoluteDiffs(patch, src, dst);
        stopTimer();
    }
    SANITY_CHECK_NOTHING();
}

} // namespace
