/*
 * Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(ThresholdOtsuTest, accuracy)
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

    bool type = 0;

    cv::fastcv::dsp::thresholdOtsu(src, dst, type);

    // De-Initialize DSP
    cv::fastcv::dsp::fcvdspdeinit();

    EXPECT_FALSE(dst.empty());
    EXPECT_EQ(src.size(), dst.size());

    // Compare the result against the reference cv::threshold function with Otsu's method
    cv::Mat referenceDst;
    cv::threshold(src, referenceDst, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    double maxDifference = 10.0;
    cv::Mat diff;
    cv::absdiff(dst, referenceDst, diff);
    double maxVal;
    cv::minMaxLoc(diff, nullptr, &maxVal);

    EXPECT_LE(maxVal, maxDifference) << "The custom threshold result differs from the reference result by more than the acceptable threshold.";
}

TEST(ThresholdOtsuTest, inPlaceAccuracy)
{
    applyTestTag(CV_TEST_TAG_FASTCV_SKIP_DSP);

    // Initialize DSP
    int initStatus = cv::fastcv::dsp::fcvdspinit();
    ASSERT_EQ(initStatus, 0) << "Failed to initialize FastCV DSP";

    cv::Mat src;
    src.allocator = cv::fastcv::getQcAllocator();
    cv::imread(cvtest::findDataFile("cv/detectors_descriptors_evaluation/planar/box_in_scene.png"), src, cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(src.empty()) << "Could not read the image file.";

    // Use the same buffer for in-place operation
    cv::Mat dst;
    dst.allocator = cv::fastcv::getQcAllocator();
    src.copyTo(dst);

    bool type = false;

    // Call the thresholdOtsu function for in-place operation
    cv::fastcv::dsp::thresholdOtsu(dst, dst, type);

    // De-Initialize DSP
    cv::fastcv::dsp::fcvdspdeinit();

    // Check if the output is not empty
    EXPECT_FALSE(dst.empty());
    EXPECT_EQ(src.size(), dst.size());

    // Compare the result against the reference cv::threshold function with Otsu's method
    cv::Mat referenceDst;
    cv::threshold(src, referenceDst, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    double maxDifference = 10.0;
    cv::Mat diff;
    cv::absdiff(dst, referenceDst, diff);
    double maxVal;
    cv::minMaxLoc(diff, nullptr, &maxVal);

    EXPECT_LE(maxVal, maxDifference) << "The in-place threshold result differs from the reference result by more than the acceptable threshold.";
}

}} // namespaces opencv_test, ::
