/*
 * Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "perf_precomp.hpp"

namespace opencv_test {

typedef perf::TestBaseWithParam<std::tuple<Size, int>> ResizePerfTest;

PERF_TEST_P(ResizePerfTest, run, ::testing::Combine(
    ::testing::Values(perf::szVGA, perf::sz720p, perf::sz1080p), // image size
    ::testing::Values(2, 4) // resize factor
))
{
    Size size = std::get<0>(GetParam());
    int factor = std::get<1>(GetParam());

    cv::Mat inputImage(size, CV_8UC1);
    cv::randu(inputImage, cv::Scalar::all(0), cv::Scalar::all(255));
    
    cv::Mat resized_image;
    Size dsize(inputImage.cols / factor, inputImage.rows / factor);

    while (next())
    {
        startTimer();
        cv::fastcv::resizeDown(inputImage, resized_image, dsize, 0, 0);
        stopTimer();
    }

    SANITY_CHECK_NOTHING();
}

typedef perf::TestBaseWithParam<std::tuple<Size, double, double, int>> ResizeByMnPerfTest;

PERF_TEST_P(ResizeByMnPerfTest, run, ::testing::Combine(
    ::testing::Values(perf::szVGA, perf::sz720p, perf::sz1080p), // image size
    ::testing::Values(0.35, 0.65), // inv_scale_x
    ::testing::Values(0.35, 0.65), // inv_scale_y
    ::testing::Values(CV_8UC1, CV_8UC2) // data type
))
{
    Size size = std::get<0>(GetParam());
    double inv_scale_x = std::get<1>(GetParam());
    double inv_scale_y = std::get<2>(GetParam());
    int type = std::get<3>(GetParam());

    cv::Mat inputImage(size, type);
    cv::randu(inputImage, cv::Scalar::all(0), cv::Scalar::all(255));
    
    Size dsize;
    cv::Mat resized_image;

    while (next())
    {
        startTimer();
        cv::fastcv::resizeDown(inputImage, resized_image, dsize, inv_scale_x, inv_scale_y);
        stopTimer();
    }

    SANITY_CHECK_NOTHING();
}

} // namespace