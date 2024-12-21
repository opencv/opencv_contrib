/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "perf_precomp.hpp"

namespace opencv_test {

typedef std::tuple<cv::Size> HistogramPerfParams;
typedef perf::TestBaseWithParam<HistogramPerfParams> HistogramPerfTest;


PERF_TEST_P(HistogramPerfTest, run,
         testing::Values(perf::szQVGA, perf::szVGA, perf::sz720p, perf::sz1080p)
    )
{
    auto p = GetParam();
    cv::Size size  = std::get<0>(p);

    RNG& rng = cv::theRNG();
    Mat src(size, CV_8UC1);
    cvtest::randUni(rng, src, Scalar::all(0), Scalar::all(255));
    Mat hist(1, 256, CV_32SC1);

    while (next())
    {
        startTimer();
        cv::fastcv::calcHist(src, hist);
        stopTimer();
    }

    SANITY_CHECK_NOTHING();
}

} // namespace
