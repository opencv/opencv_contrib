/*
 * Copyright (c) 2024-2025 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "perf_precomp.hpp"

namespace opencv_test {

typedef perf::TestBaseWithParam<tuple<Size, int, int, bool>> GaussianBlurPerfTest;

PERF_TEST_P(GaussianBlurPerfTest, run,
    ::testing::Combine(::testing::Values(perf::szVGA, perf::sz720p, perf::sz1080p), // image size
                       ::testing::Values(CV_8U,CV_16S,CV_32S),                      // image depth
                       ::testing::Values(3, 5),                                     // kernel size
                       ::testing::Values(true,false)                                // blur border
                       )
           )
{
    cv::Size srcSize = get<0>(GetParam());
    int depth = get<1>(GetParam());
    int ksize = get<2>(GetParam());
    bool border = get<3>(GetParam());

    // For some cases FastCV not support, so skip them
    if((ksize!=5) && (depth!=CV_8U))
        throw ::perf::TestBase::PerfSkipTestException();

    cv::Mat src(srcSize, depth);
    cv::Mat dst;
    RNG& rng = cv::theRNG();
    cvtest::randUni(rng, src, Scalar::all(0), Scalar::all(255));

    while (next())
    {
        startTimer();
        cv::fastcv::gaussianBlur(src, dst, ksize, border);
        stopTimer();
    }

    SANITY_CHECK_NOTHING();
}

typedef perf::TestBaseWithParam<tuple<Size, int, int>> Filter2DPerfTest;

PERF_TEST_P(Filter2DPerfTest, run,
    ::testing::Combine(::testing::Values(perf::szVGA, perf::sz720p, perf::sz1080p), // image size
                       ::testing::Values(CV_8U,CV_16S,CV_32F),                      // dst image depth
                       ::testing::Values(3, 5, 7, 9, 11)                            // kernel size
                       )
           )
{
    cv::Size srcSize = get<0>(GetParam());
    int ddepth = get<1>(GetParam());
    int ksize = get<2>(GetParam());

    cv::Mat src(srcSize, CV_8U);
    cv::Mat kernel;
    cv::Mat dst;

    switch (ddepth)
    {
        case CV_8U:
        case CV_16S:
        {
            kernel.create(ksize,ksize,CV_8S);
            break;
        }
        case CV_32F:
        {
            kernel.create(ksize,ksize,CV_32F);
            break;
        }
        default:
            break;
    }

    cv::randu(src, 0, 256);
    cv::randu(kernel, INT8_MIN, INT8_MAX);
    RNG& rng = cv::theRNG();
    cvtest::randUni(rng, src, Scalar::all(0), Scalar::all(255));

    while (next())
    {
        startTimer();
        cv::fastcv::filter2D(src, dst, ddepth, kernel);
        stopTimer();
    }

    SANITY_CHECK_NOTHING();
}

typedef perf::TestBaseWithParam<tuple<Size, int, int>> SepFilter2DPerfTest;

PERF_TEST_P(SepFilter2DPerfTest, run,
    ::testing::Combine(::testing::Values(perf::szVGA, perf::sz720p, perf::sz1080p), // image size
                       ::testing::Values(CV_8U,CV_16S),                             // dst image depth
                       ::testing::Values(3, 5, 7, 9, 11, 13, 15, 17)                // kernel size
                       )
           )
{
    cv::Size srcSize = get<0>(GetParam());
    int ddepth = get<1>(GetParam());
    int ksize = get<2>(GetParam());

    cv::Mat src(srcSize, ddepth);
    cv::Mat kernel(1, ksize, ddepth);
    cv::Mat dst;
    RNG& rng = cv::theRNG();
    cvtest::randUni(rng, src, Scalar::all(0), Scalar::all(255));
    cvtest::randUni(rng, kernel, Scalar::all(INT8_MIN), Scalar::all(INT8_MAX));

    while (next())
    {
        startTimer();
        cv::fastcv::sepFilter2D(src, dst, ddepth, kernel, kernel);
        stopTimer();
    }

    SANITY_CHECK_NOTHING();
}

typedef perf::TestBaseWithParam<tuple<Size, int, Size, int>> NormalizeLocalBoxPerfTest;

PERF_TEST_P(NormalizeLocalBoxPerfTest, run,
    ::testing::Combine(::testing::Values(perf::szVGA, perf::sz720p, perf::sz1080p), // image size
                       ::testing::Values(CV_8U,CV_32F),                             // src image depth
                       ::testing::Values(Size(3,3),Size(5,5)),                      // patch size
                       ::testing::Values(0,1)                                       // use std dev or not
                       )
           )
{
    cv::Size srcSize = get<0>(GetParam());
    int depth = get<1>(GetParam());
    Size sz = get<2>(GetParam());
    bool useStdDev = get<3>(GetParam());

    cv::Mat src(srcSize, depth);
    cv::Mat dst;
    RNG& rng = cv::theRNG();
    cvtest::randUni(rng, src, Scalar::all(0), Scalar::all(255));

    TEST_CYCLE() cv::fastcv::normalizeLocalBox(src, dst, sz, useStdDev);

    SANITY_CHECK_NOTHING();
}

} // namespace