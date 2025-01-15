/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "test_precomp.hpp"

namespace opencv_test { namespace {

typedef std::tuple<cv::Size, int /*lowThresh*/, int /*highThresh*/, int /*trueValue*/, int /*falseValue*/> ThresholdRangeTestParams;
class ThresholdRangeTest : public ::testing::TestWithParam<ThresholdRangeTestParams> {};

TEST_P(ThresholdRangeTest, accuracy)
{
    auto p = GetParam();
    cv::Size size  = std::get<0>(p);
    int loThresh  = std::get<1>(p);
    int hiThresh = std::get<2>(p);
    int trueValue  = std::get<3>(p);
    int falseValue = std::get<4>(p);

    int lowThresh  = std::min(loThresh, hiThresh);
    int highThresh = std::max(loThresh, hiThresh);

    RNG& rng = cv::theRNG();
    Mat src(size, CV_8UC1);
    cvtest::randUni(rng, src, Scalar::all(0), Scalar::all(256));

    Mat dst;
    cv::fastcv::thresholdRange(src, dst, lowThresh, highThresh, trueValue, falseValue);

    Mat inr, ref(src.size(), CV_8UC1);
    cv::inRange(src, lowThresh, highThresh, inr);
    ref.setTo(trueValue, inr);
    ref.setTo(falseValue, ~inr);

    double normInf = cvtest::norm(ref, dst, cv::NORM_INF);

    EXPECT_EQ(normInf, 0);
}

INSTANTIATE_TEST_CASE_P(FastCV_Extension, ThresholdRangeTest,
                         ::testing::Combine(::testing::Values(Size(8, 8), Size(640, 480), Size(800, 600)),
                                            ::testing::Values(0, 15, 128, 255), // lowThresh
                                            ::testing::Values(0, 15, 128, 255), // highThresh
                                            ::testing::Values(0, 15, 128, 255), // trueValue
                                            ::testing::Values(0, 15, 128, 255)  // falseValue
                                            ));

}} // namespaces opencv_test, ::
