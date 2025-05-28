/*
 * Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "test_precomp.hpp"

using namespace cv::fastcv::dsp;

namespace opencv_test { namespace {

TEST(SadTest, accuracy)
{
    applyTestTag(CV_TEST_TAG_FASTCV_SKIP_DSP);

    //Initialize DSP
    int initStatus = cv::fastcv::dsp::fcvdspinit();
    ASSERT_EQ(initStatus, 0) << "Failed to initialize FastCV DSP";

    // Create an 8x8 template patch
    cv::Mat patch;
    patch.allocator = cv::fastcv::getQcAllocator();
    patch.create(8, 8, CV_8UC1);
    patch.setTo(cv::Scalar(0));

    // Create a source image
    cv::Mat src;
    src.allocator = cv::fastcv::getQcAllocator();
    src.create(512, 512, CV_8UC1);
    src.setTo(cv::Scalar(255));

    cv::Mat dst;
    dst.allocator = cv::fastcv::getQcAllocator();

    cv::fastcv::dsp::sumOfAbsoluteDiffs(patch, src, dst);

    EXPECT_FALSE(dst.empty());

    //De-Initialize DSP
    cv::fastcv::dsp::fcvdspdeinit();
}

}
}
