/*
 * Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "perf_precomp.hpp"

namespace opencv_test {

typedef perf::TestBaseWithParam<tuple<Size, int, pair<int, int>, bool>> CannyPerfTest;

PERF_TEST_P(CannyPerfTest, run,
    ::testing::Combine(::testing::Values(perf::szVGA, perf::sz720p, perf::sz1080p), // image size
        ::testing::Values(3, 5, 7), // aperture size
        ::testing::Values(make_pair(0, 50), make_pair(100, 150), make_pair(50, 150)), // low and high thresholds
        ::testing::Values(false, true) // L2gradient
    )
)
{
    applyTestTag(CV_TEST_TAG_FASTCV_SKIP_DSP);

    //Initialize DSP
    int initStatus = cv::fastcv::dsp::fcvdspinit();
    ASSERT_EQ(initStatus, 0) << "Failed to initialize FastCV DSP";

    cv::Size srcSize = get<0>(GetParam());
    int apertureSize = get<1>(GetParam());
    auto thresholds = get<2>(GetParam());
    bool L2gradient = get<3>(GetParam());

    cv::Mat src;
    src.allocator = cv::fastcv::getQcAllocator();
    src.create(srcSize, CV_8UC1);

    cv::Mat dst;
    dst.allocator = cv::fastcv::getQcAllocator();

    cv::randu(src, 0, 256);

    int lowThreshold = thresholds.first;
    int highThreshold = thresholds.second;

    while (next())
    {
        startTimer();
        cv::fastcv::dsp::Canny(src, dst, lowThreshold, highThreshold, apertureSize, L2gradient);
        stopTimer();
    }

    //De-Initialize DSP
    cv::fastcv::dsp::fcvdspdeinit();

    SANITY_CHECK_NOTHING();
}

} //namespace
