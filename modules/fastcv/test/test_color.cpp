/*
 * Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "test_precomp.hpp"

namespace opencv_test { namespace {

static inline void fillRandom8U(cv::Mat& m)
{
    cv::RNG& rng = cv::theRNG();
    rng.fill(m, cv::RNG::UNIFORM, 0, 256);
}

TEST(Fastcv_cvtColor, YUV420_to_YUV422_and_back_roundtrip)
{
    const cv::Size sz(640, 480);

    cv::Mat bgr(sz, CV_8UC3);
    fillRandom8U(bgr);

    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    cv::Mat yuv420_before;
    yuv420_before.allocator = cv::fastcv::getQcAllocator();
    cv::fastcv::cvtColor(rgb, yuv420_before, cv::fastcv::COLOR_RGB2YUV_NV12);

    cv::Mat yuv422;
    yuv422.allocator = cv::fastcv::getQcAllocator();
    cv::fastcv::cvtColor(yuv420_before, yuv422, cv::fastcv::COLOR_YUV2YUV422sp_NV12);

    cv::Mat yuv422_to_bgr;

    cv::Mat yuv420_after;
    yuv420_after.allocator = cv::fastcv::getQcAllocator();
    cv::fastcv::cvtColor(yuv422, yuv420_after, cv::fastcv::COLOR_YUV422sp2YUV_NV12);

    ASSERT_EQ(yuv420_before.size(), yuv420_after.size());
    ASSERT_EQ(yuv420_before.type(), yuv420_after.type());

    double maxDiff = cv::norm(yuv420_before, yuv420_after, cv::NORM_INF);
    std::cout << "Max difference YUV420 before vs after = " << maxDiff << std::endl;
    EXPECT_LE(maxDiff, 1.0);
}

TEST(Fastcv_cvtColor, YUV444_to_YUV420_and_back_roundtrip)
{
    const cv::Size sz(640, 480);

    cv::Mat bgr(sz, CV_8UC3);
    fillRandom8U(bgr);
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    cv::Mat yuv444_initial;
    yuv444_initial.allocator = cv::fastcv::getQcAllocator();
    cv::fastcv::cvtColor(rgb, yuv444_initial, cv::fastcv::COLOR_RGB2YUV444sp);

    cv::Mat yuv420;
    yuv420.allocator = cv::fastcv::getQcAllocator();
    cv::fastcv::cvtColor(yuv444_initial, yuv420, cv::fastcv::COLOR_YUV444sp2YUV_NV12);

    cv::Mat yuv444_final;
    yuv444_final.allocator = cv::fastcv::getQcAllocator();
    cv::fastcv::cvtColor(yuv420, yuv444_final, cv::fastcv::COLOR_YUV2YUV444sp_NV12);

    ASSERT_EQ(yuv444_initial.size(), yuv444_final.size());
    ASSERT_EQ(yuv444_initial.type(), yuv444_final.type());

    double maxDiff = cv::norm(yuv444_initial, yuv444_final, cv::NORM_INF);
    std::cout << "Max difference YUV444 before vs after roundtrip = " << maxDiff << std::endl;
    EXPECT_LE(maxDiff, 2.0);
}

TEST(Fastcv_cvtColor, YUV444_to_YUV422_and_back_roundtrip) 
{
    const cv::Size sz(640, 480);

    cv::Mat bgr(sz, CV_8UC3);
    fillRandom8U(bgr);
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    cv::Mat yuv444_initial;
    yuv444_initial.allocator = cv::fastcv::getQcAllocator();
    cv::fastcv::cvtColor(rgb, yuv444_initial, cv::fastcv::COLOR_RGB2YUV444sp);

    cv::Mat yuv422;
    yuv422.allocator = cv::fastcv::getQcAllocator();
    cv::fastcv::cvtColor(yuv444_initial, yuv422, cv::fastcv::COLOR_YUV444sp2YUV422sp);

    cv::Mat yuv444_final;
    yuv444_final.allocator = cv::fastcv::getQcAllocator();
    cv::fastcv::cvtColor(yuv422, yuv444_final, cv::fastcv::COLOR_YUV422sp2YUV444sp);

    ASSERT_EQ(yuv444_initial.size(), yuv444_final.size());
    ASSERT_EQ(yuv444_initial.type(), yuv444_final.type());

    double maxDiff = cv::norm(yuv444_initial, yuv444_final, cv::NORM_INF);
    std::cout << "Max difference YUV444 before vs after roundtrip = " << maxDiff << std::endl;
    EXPECT_LE(maxDiff, 2.0);
}

TEST(Fastcv_cvtColor, YUV444_to_RGB565_and_back_roundtrip)
{
    const cv::Size sz(640, 480);
    cv::Mat bgr(sz, CV_8UC3);
    fillRandom8U(bgr);

    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    cv::Mat yuv444_initial;
    yuv444_initial.allocator = cv::fastcv::getQcAllocator();
    cv::fastcv::cvtColor(rgb, yuv444_initial, cv::fastcv::COLOR_RGB2YUV444sp);

    cv::Mat rgb565(sz, CV_8UC2);
    rgb565.allocator = cv::fastcv::getQcAllocator();
    cv::fastcv::cvtColor(yuv444_initial, rgb565, cv::fastcv::COLOR_YUV444sp2RGB565);

    cv::Mat yuv444_roundtrip;
    yuv444_roundtrip.allocator = cv::fastcv::getQcAllocator();
    cv::fastcv::cvtColor(rgb565, yuv444_roundtrip, cv::fastcv::COLOR_RGB5652YUV444sp);

    ASSERT_EQ(yuv444_initial.size(), yuv444_roundtrip.size());
    ASSERT_EQ(yuv444_initial.type(), yuv444_roundtrip.type());

    double maxDiff = cv::norm(yuv444_initial, yuv444_roundtrip, cv::NORM_INF);
    std::cout << "Max difference YUV444 after RGB565 roundtrip = " << maxDiff << std::endl;

    EXPECT_LE(maxDiff, 2.0);
}

}} // namespace opencv_test
