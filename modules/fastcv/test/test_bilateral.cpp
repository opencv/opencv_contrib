/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "test_precomp.hpp"

namespace opencv_test { namespace {

typedef testing::TestWithParam<tuple<cv::Size,int,int>> fcv_bilateralFilterTest;

TEST_P(fcv_bilateralFilterTest, accuracy)
{
    cv::Size size  = get<0>(GetParam());
    int d = get<1>(GetParam());
    double sigmaColor = get<2>(GetParam());
    double sigmaSpace = sigmaColor;

    RNG& rng = cv::theRNG();
    Mat src(size, CV_8UC1);
    cvtest::randUni(rng, src, Scalar::all(0), Scalar::all(255));

    cv::Mat dst;

    cv::fastcv::bilateralFilter(src, dst, d, sigmaColor, sigmaSpace);

    EXPECT_FALSE(dst.empty());
}

INSTANTIATE_TEST_CASE_P(/*nothing*/, fcv_bilateralFilterTest, Combine(
                   ::testing::Values(Size(8, 8), Size(640, 480), Size(800, 600)),
                   ::testing::Values(5, 7, 9),
                   ::testing::Values(1., 10.)
));

}
}

