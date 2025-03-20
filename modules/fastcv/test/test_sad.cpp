/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "test_precomp.hpp"

using namespace cv::fastcv::dsp;

namespace opencv_test { namespace {

TEST(SadTest, accuracy)
{
    // Initialize FastCV DSP
    int initStatus = cv::fastcv::dsp::fastcvq6init();
    ASSERT_EQ(initStatus, 0) << "Failed to initialize FastCV DSP";
    
    // Create an 8x8 template patch
    cv::Mat patch = cv::Mat::zeros(8, 8, CV_8UC1);

    // Create a source image
    cv::Mat src = cv::Mat::ones(512, 512, CV_8UC1) * 255;

    cv::Mat dst;

    cv::fastcv::dsp::sumOfAbsoluteDiffs(patch, src, dst);
    
    EXPECT_FALSE(dst.empty());

    // Explicitly deallocate memory
    patch.release();
    src.release();
    dst.release();

    cv::fastcv::dsp::fastcvq6deinit();
}

}
}
