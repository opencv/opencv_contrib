/*
 * Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "perf_precomp.hpp"

namespace opencv_test {

typedef std::tuple<cv::Size, bool /*type*/> ThresholdOtsuPerfParams;
typedef perf::TestBaseWithParam<ThresholdOtsuPerfParams> ThresholdOtsuPerfTest;

PERF_TEST_P(ThresholdOtsuPerfTest, run,
    ::testing::Combine(::testing::Values(Size(320, 240), Size(640, 480), Size(1280, 720), Size(1920, 1080)),
        ::testing::Values(false, true) // type
    )
)
{
    applyTestTag(CV_TEST_TAG_FASTCV_SKIP_DSP);

    //Initialize DSP
    int initStatus = cv::fastcv::dsp::fcvdspinit();
    ASSERT_EQ(initStatus, 0) << "Failed to initialize FastCV DSP";

    auto p = GetParam();
    cv::Size size = std::get<0>(p);
    bool type = std::get<1>(p);

    RNG& rng = cv::theRNG();

    cv::Mat src;
    src.allocator = cv::fastcv::getQcAllocator();
    src.create(size, CV_8UC1);

    cvtest::randUni(rng, src, Scalar::all(0), Scalar::all(256));

    cv::Mat dst;
    dst.allocator = cv::fastcv::getQcAllocator();

    while (next())
    {
        startTimer();
        cv::fastcv::dsp::thresholdOtsu(src, dst, type);
        stopTimer();
    }

    //De-Initialize DSP
    cv::fastcv::dsp::fcvdspdeinit();
    SANITY_CHECK_NOTHING();
}

} // namespace
