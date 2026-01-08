/*
 * Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "perf_precomp.hpp"

namespace opencv_test {

typedef perf::TestBaseWithParam<cv::Size> FFT_DSPExtPerfTest;

PERF_TEST_P_(FFT_DSPExtPerfTest, forward)
{
    applyTestTag(CV_TEST_TAG_FASTCV_SKIP_DSP);

    //Initialize DSP
    int initStatus = cv::fastcv::dsp::fcvdspinit();
    ASSERT_EQ(initStatus, 0) << "Failed to initialize FastCV DSP";

    Size size = GetParam();

    RNG& rng = cv::theRNG();

    Mat src;
    src.allocator = cv::fastcv::getQcAllocator();
    src.create(size, CV_8UC1);
    cvtest::randUni(rng, src, Scalar::all(0), Scalar::all(256));

    Mat dst;
    dst.allocator = cv::fastcv::getQcAllocator();

    while (next())
    {
        startTimer();
        cv::fastcv::dsp::FFT(src, dst);
        stopTimer();
    }

    //De-Initialize DSP
    cv::fastcv::dsp::fcvdspdeinit();

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(FFT_DSPExtPerfTest, inverse)
{
    applyTestTag(CV_TEST_TAG_FASTCV_SKIP_DSP);

    //Initialize DSP
    int initStatus = cv::fastcv::dsp::fcvdspinit();
    ASSERT_EQ(initStatus, 0) << "Failed to initialize FastCV DSP";

    Size size = GetParam();

    RNG& rng = cv::theRNG();

    Mat src;
    src.allocator = cv::fastcv::getQcAllocator();
    src.create(size, CV_8UC1);

    cvtest::randUni(rng, src, Scalar::all(0), Scalar::all(256));

    Mat fwd, back;
    fwd.allocator = cv::fastcv::getQcAllocator();
    back.allocator = cv::fastcv::getQcAllocator();

    cv::fastcv::dsp::FFT(src, fwd);

    while (next())
    {
        startTimer();
        cv::fastcv::dsp::IFFT(fwd, back);
        stopTimer();
    }

    //De-Initialize DSP
    cv::fastcv::dsp::fcvdspdeinit();

    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(FastCV_Extension, FFT_DSPExtPerfTest,
    ::testing::Values(Size(256, 256), Size(512, 512)));

} // namespace
