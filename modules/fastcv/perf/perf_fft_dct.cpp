/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "perf_precomp.hpp"

namespace opencv_test {

typedef perf::TestBaseWithParam<cv::Size> FFTExtPerfTest;

PERF_TEST_P_(FFTExtPerfTest, forward)
{
    Size size = GetParam();

    RNG& rng = cv::theRNG();
    Mat src(size, CV_8UC1);
    cvtest::randUni(rng, src, Scalar::all(0), Scalar::all(256));

    Mat dst;

    while(next())
    {
        startTimer();
        cv::fastcv::FFT(src, dst);
        stopTimer();
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(FFTExtPerfTest, inverse)
{
    Size size = GetParam();

    RNG& rng = cv::theRNG();
    Mat src(size, CV_8UC1);
    cvtest::randUni(rng, src, Scalar::all(0), Scalar::all(256));

    Mat fwd, back;
    cv::fastcv::FFT(src, fwd);

    while(next())
    {
        startTimer();
        cv::fastcv::IFFT(fwd, back);
        stopTimer();
    }

    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(FastCV_Extension, FFTExtPerfTest,
    ::testing::Values(Size(8, 8), Size(128, 128), Size(32, 256), Size(512, 512),
                      Size(32, 1), Size(512, 1)));

/// DCT ///

typedef perf::TestBaseWithParam<cv::Size> DCTExtPerfTest;

PERF_TEST_P_(DCTExtPerfTest, forward)
{
    Size size = GetParam();

    RNG& rng = cv::theRNG();
    Mat src(size, CV_8UC1);
    cvtest::randUni(rng, src, Scalar::all(0), Scalar::all(256));

    Mat dst, ref;

    while(next())
    {
        startTimer();
        cv::fastcv::DCT(src, dst);
        stopTimer();
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(DCTExtPerfTest, inverse)
{
    Size size = GetParam();

    RNG& rng = cv::theRNG();
    Mat src(size, CV_8UC1);
    cvtest::randUni(rng, src, Scalar::all(0), Scalar::all(256));

    Mat fwd, back;
    cv::fastcv::DCT(src, fwd);

    while(next())
    {
        startTimer();
        cv::fastcv::IDCT(fwd, back);
        stopTimer();
    }

    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(FastCV_Extension, DCTExtPerfTest,
    ::testing::Values(Size(8, 8), Size(128, 128), Size(32, 256), Size(512, 512)));
} // namespace
