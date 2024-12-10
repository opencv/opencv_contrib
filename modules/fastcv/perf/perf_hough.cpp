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
    double threshold  = std::get<1>(p);

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
        cv::fastcv::houghLines(src, lines, threshold);
        stopTimer();
    }

    SANITY_CHECK_NOTHING();
}


typedef std::tuple<std::string /* file name */, uint32_t /* minDist */,   uint32_t /* cannyThreshold */,
                   uint32_t /* accThreshold */, uint32_t /* minRadius */, uint32_t /* maxRadius */> HoughCirclesPerfTestParams;
typedef ::perf::TestBaseWithParam<HoughCirclesPerfTestParams> HoughCirclesPerf;

// NOTE: test files should be manually loaded to folder on a device, for example like this:
// adb push fastcv/misc/hough/ /sdcard/testdata/fastcv/hough/

PERF_TEST_P(HoughCirclesPerf, run,
                ::testing::Values(
                            HoughCirclesPerfTestParams {"cv/cameracalibration/circles/circles4.png", 100, 100, 50, 10, 100 },
                            HoughCirclesPerfTestParams {"cv/cameracalibration/circles/circles4.png", 100, 100, 50, 30, 100 },
                            HoughCirclesPerfTestParams {"cv/cameracalibration/circles/circles4.png", 100, 100, 50, 50, 100 },
                            HoughCirclesPerfTestParams {"cv/cameracalibration/circles/circles4.png",  10, 100, 50, 10, 100 },
                            HoughCirclesPerfTestParams {"cv/cameracalibration/circles/circles4.png",  10, 100, 50, 30, 100 },
                            HoughCirclesPerfTestParams {"cv/cameracalibration/circles/circles4.png",  10, 100, 50, 50, 100 }
                         )
           )
{
    auto p = GetParam();
    std::string fname       = std::get<0>(p);
    uint32_t minDist        = std::get<1>(p);
    uint32_t cannyThreshold = std::get<2>(p);
    uint32_t accThreshold   = std::get<3>(p);
    uint32_t minRadius      = std::get<4>(p);
    uint32_t maxRadius      = std::get<5>(p);

    cv::Mat src = imread(cvtest::findDataFile(fname), cv::IMREAD_GRAYSCALE);
    // make it aligned by 8
    cv::Mat withBorder;
    int bpix = ((src.cols & 0xfffffff8) + 8) - src.cols;
    cv::copyMakeBorder(src, withBorder, 0, 0, 0, bpix, BORDER_REFLECT101);
    src = withBorder;

    while(next())
    {
        Mat icircles;
        startTimer();
        cv::fastcv::houghCircles(src, icircles, minDist,
                                 cannyThreshold, accThreshold,
                                 minRadius, maxRadius);
        stopTimer();
    }

    SANITY_CHECK_NOTHING();
}

} // namespace
