/*
 * Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "perf_precomp.hpp"

namespace opencv_test {

typedef perf::TestBaseWithParam<tuple<Size, int, int>> Filter2DPerfTest_DSP;

PERF_TEST_P(Filter2DPerfTest_DSP, run,
    ::testing::Combine(::testing::Values(perf::szVGA, perf::sz720p),                // image size
                       ::testing::Values(CV_8U,CV_16S,CV_32F),                      // dst image depth
                       ::testing::Values(3, 5, 7)                                   // kernel size
                       )
           )
{
    applyTestTag(CV_TEST_TAG_FASTCV_SKIP_DSP);

    //Initialize DSP
    int initStatus = cv::fastcv::dsp::fcvdspinit();
    ASSERT_EQ(initStatus, 0) << "Failed to initialize FastCV DSP";

    cv::Size srcSize = get<0>(GetParam());
    int ddepth = get<1>(GetParam());
    int ksize = get<2>(GetParam());

    cv::Mat src;
    src.allocator = cv::fastcv::getQcAllocator();
    src.create(srcSize, CV_8U);

    cv::Mat kernel;
    cv::Mat dst;
    kernel.allocator = cv::fastcv::getQcAllocator();
    dst.allocator = cv::fastcv::getQcAllocator();

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
        cv::fastcv::dsp::filter2D(src, dst, ddepth, kernel);
        stopTimer();
    }

    //De-Initialize DSP
    cv::fastcv::dsp::fcvdspdeinit();

    SANITY_CHECK_NOTHING();
}

} // namespace
