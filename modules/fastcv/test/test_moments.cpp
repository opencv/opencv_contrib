/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "test_precomp.hpp"

namespace opencv_test { namespace {

typedef testing::TestWithParam<tuple<bool,Size,int>> fcv_momentsTest;

TEST_P(fcv_momentsTest, accuracy)
{
    const bool binaryImage = get<0>(GetParam());
    const Size srcSize = get<1>(GetParam());
    const MatDepth srcType = get<2>(GetParam());
    Mat src(srcSize, srcType);
    cv::RNG& rng = cv::theRNG();
    if(srcType == CV_8UC1)
        rng.fill(src,  cv::RNG::UNIFORM, 0, 5);
    else if(srcType == CV_32SC1)
        rng.fill(src, cv::RNG::UNIFORM, 0, 5);
    else if(srcType == CV_32FC1)
        rng.fill(src, cv::RNG::UNIFORM, 0.f, 5.f);

    cv::Moments m = cv::fastcv::moments(src, binaryImage);

    cv::Scalar mean_val, stdDev;
    float mean_val_fcv = m.m00/(srcSize.width * srcSize.height);
    if(binaryImage)
    {
        cv::Mat src_binary(srcSize, CV_8UC1);
        cv::compare( src, 0, src_binary, cv::CMP_NE );
        mean_val = cv::mean(src_binary);
        mean_val_fcv *= 255;
    }
    else
        mean_val = cv::mean(src);

    EXPECT_NEAR(mean_val[0], mean_val_fcv, 2);
}

INSTANTIATE_TEST_CASE_P(/*nothing*/, fcv_momentsTest, Combine(
                   Values(false, true),
                   Values(TYPICAL_MAT_SIZES),
                   Values(CV_8UC1, CV_32SC1, CV_32FC1)
));

}
}
