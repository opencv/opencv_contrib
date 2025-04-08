/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "perf_precomp.hpp"

namespace opencv_test {

typedef std::tuple<std::string /* file name */, double /* threshold */ > HoughLinesPerfParams;
typedef perf::TestBaseWithParam<HoughLinesPerfParams> HoughLinesPerfTest;

PERF_TEST_P(HoughLinesPerfTest, run,
                        ::testing::Combine(::testing::Values("cv/shared/pic5.png",
                                                             "stitching/a1.png",
                                                             "cv/shared/pic5.png",
                                                             "cv/shared/pic1.png"), // images
                                           ::testing::Values(0.05, 0.25, 0.5, 0.75, 5) // threshold
                                           )
           )
{
    auto p = GetParam();
    std::string fname = std::get<0>(p);
    double thrld  = std::get<1>(p);

    cv::Mat src = imread(cvtest::findDataFile(fname), cv::IMREAD_GRAYSCALE);
    // make it aligned by 8
    cv::Mat withBorder;
    int bpix = ((src.cols & 0xfffffff8) + 8) - src.cols;
    cv::copyMakeBorder(src, withBorder, 0, 0, 0, bpix, BORDER_REFLECT101);
    src = withBorder;

    while(next())
    {
        std::vector<cv::Vec4f> lines;
        startTimer();
        cv::fastcv::houghLines(src, lines, thrld);
        stopTimer();
    }

    SANITY_CHECK_NOTHING();
}

} // namespace
