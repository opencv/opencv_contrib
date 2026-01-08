/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "test_precomp.hpp"

namespace opencv_test { namespace {

typedef testing::TestWithParam<tuple<Size, int, int, int>> Sobel;
typedef testing::TestWithParam<tuple<Size, int>> Sobel3x3u8;

TEST_P(Sobel,accuracy)
{
    Size srcSize = get<0>(GetParam());
    int ksize = get<1>(GetParam());
    int border = get<2>(GetParam());
    int borderValue = get<3>(GetParam());

    cv::Mat dx, dy, src(srcSize, CV_8U), refx, refy;
    RNG& rng = cv::theRNG();
    cvtest::randUni(rng, src, Scalar::all(0), Scalar::all(255));
    cv::fastcv::sobel(src, dx, dy, ksize, border, borderValue);

    cv::Sobel(src, refx, CV_16S, 1, 0, ksize, 1.0, 0.0, border);
    cv::Sobel(src, refy, CV_16S, 0, 1, ksize, 1.0, 0.0, border);

    cv::Mat difference_x, difference_y;
    cv::absdiff(dx, refx, difference_x);
    cv::absdiff(dy, refy, difference_y);

    int num_diff_pixels_x = cv::countNonZero(difference_x);
    int num_diff_pixels_y = cv::countNonZero(difference_y);
    EXPECT_LT(num_diff_pixels_x, src.size().area()*0.1);
    EXPECT_LT(num_diff_pixels_y, src.size().area()*0.1);
}

TEST_P(Sobel3x3u8,accuracy)
{
    Size srcSize = get<0>(GetParam());
    int ddepth = get<1>(GetParam());

    cv::Mat dx, dy, src(srcSize, CV_8U), refx, refy;
    RNG& rng = cv::theRNG();
    cvtest::randUni(rng, src, Scalar::all(0), Scalar::all(255));

    cv::fastcv::sobel3x3u8(src, dx, dy, ddepth, 0);
    cv::Sobel(src, refx, ddepth, 1, 0);
    cv::Sobel(src, refy, ddepth, 0, 1);

    cv::Mat difference_x, difference_y;
    cv::absdiff(dx, refx, difference_x);
    cv::absdiff(dy, refy, difference_y);

    int num_diff_pixels_x = cv::countNonZero(difference_x);
    int num_diff_pixels_y = cv::countNonZero(difference_y);
    EXPECT_LT(num_diff_pixels_x, src.size().area()*0.1);
    EXPECT_LT(num_diff_pixels_y, src.size().area()*0.1);
}

INSTANTIATE_TEST_CASE_P(FastCV_Extension, Sobel, Combine(
/*image size*/      Values(perf::szVGA, perf::sz720p, perf::sz1080p),
/*kernel size*/     Values(3,5,7),
/*border*/          Values(BORDER_CONSTANT, BORDER_REPLICATE),
/*border value*/    Values(0)
));

INSTANTIATE_TEST_CASE_P(FastCV_Extension, Sobel3x3u8, Combine(
/*image size*/      Values(perf::szVGA, perf::sz720p, perf::sz1080p),
/*dst depth*/       Values(CV_16S, CV_32F)
));

}
}
