/*
 * Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(DSP_CannyTest, accuracy)
{
    applyTestTag(CV_TEST_TAG_FASTCV_SKIP_DSP);

    //Initialize DSP
    int initStatus = cv::fastcv::dsp::fcvdspinit();
    ASSERT_EQ(initStatus, 0) << "Failed to initialize FastCV DSP";

    cv::Mat src;
    src.allocator = cv::fastcv::getQcAllocator();
    cv::imread(cvtest::findDataFile("cv/detectors_descriptors_evaluation/planar/box_in_scene.png"), src, cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(src.empty()) << "Could not read the image file.";

    cv::Mat dst;
    dst.allocator = cv::fastcv::getQcAllocator();

    int lowThreshold = 0;
    int highThreshold = 150;

    cv::fastcv::dsp::Canny(src, dst, lowThreshold, highThreshold, 3, true);

    //De-Initialize DSP
    cv::fastcv::dsp::fcvdspdeinit();

    EXPECT_FALSE(dst.empty());
    EXPECT_EQ(src.size(), dst.size());
}

}
}
