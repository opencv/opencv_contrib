/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "perf_precomp.hpp"

namespace opencv_test {

typedef std::tuple<bool /*useScores*/, int /*barrier*/, int /*border*/, bool /*nmsEnabled*/> FAST10PerfParams;
typedef perf::TestBaseWithParam<FAST10PerfParams> FAST10PerfTest;

PERF_TEST_P(FAST10PerfTest, run,
::testing::Combine(::testing::Bool(),   // useScores
                   ::testing::Values(10, 30, 50), // barrier
                   ::testing::Values( 4, 10, 32), // border
                   ::testing::Bool() // nonmax suppression
                  )
           )
{
    auto p = GetParam();
    bool useScores  = std::get<0>(p);
    int  barrier    = std::get<1>(p);
    int  border     = std::get<2>(p);
    bool nmsEnabled = std::get<3>(p);

    cv::Mat src = imread(cvtest::findDataFile("cv/shared/baboon.png"), cv::IMREAD_GRAYSCALE);

    std::vector<int> coords, scores;
    while(next())
    {
        coords.clear();
        scores.clear();
        startTimer();
        cv::fastcv::FAST10(src, noArray(), coords, useScores ? scores : noArray(), barrier, border, nmsEnabled);
        stopTimer();
    }

    SANITY_CHECK_NOTHING();
}

} // namespace
