/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "test_precomp.hpp"

namespace opencv_test { namespace {

class ResizeBy2Test : public ::testing::TestWithParam<cv::Size> {};
class ResizeBy4Test : public ::testing::TestWithParam<cv::Size> {};

TEST(resizeDownBy2, accuracy)
{
    cv::Mat inputImage = cv::imread(cvtest::findDataFile("cv/shared/box_in_scene.png"), cv::IMREAD_GRAYSCALE);

    Size dsize;
    cv::Mat resized_image;

    cv::fastcv::resizeDownBy2(inputImage, resized_image);

    EXPECT_FALSE(resized_image.empty());

    cv::Mat resizedImageOpenCV;
    cv::resize(inputImage, resizedImageOpenCV, cv::Size(inputImage.cols / 2, inputImage.rows / 2), 0, 0, INTER_AREA);

    // Calculate the maximum difference
    double maxVal = cv::norm(resized_image, resizedImageOpenCV, cv::NORM_INF);

    // Assert if the difference is acceptable (max difference should be less than 10)
    CV_Assert(maxVal < 10 && "Difference between images is too high!");
}

TEST(resizeDownBy4, accuracy)
{
    cv::Mat inputImage = cv::imread(cvtest::findDataFile("cv/shared/box_in_scene.png"), cv::IMREAD_GRAYSCALE);

    Size dsize;
    cv::Mat resized_image;

    cv::fastcv::resizeDownBy4(inputImage, resized_image);

    EXPECT_FALSE(resized_image.empty());

    cv::Mat resizedImageOpenCV;
    cv::resize(inputImage, resizedImageOpenCV, cv::Size(inputImage.cols / 4, inputImage.rows / 4), 0, 0, INTER_AREA);

    // Calculate the maximum difference
    double maxVal = cv::norm(resized_image, resizedImageOpenCV, cv::NORM_INF);

    // Assert if the difference is acceptable (max difference should be less than 10)
    CV_Assert(maxVal < 10 && "Difference between images is too high!");
}

TEST_P(ResizeBy2Test, ResizeBy2) {

    //Size size = get<0>(GetParam());
    Size size = GetParam();
    cv::Mat inputImage(size, CV_8UC1);
    randu(inputImage, Scalar::all(0), Scalar::all(255)); // Fill with random values

    Size dsize;
    cv::Mat resized_image;

    // Resize the image by a factor of 2
    cv::fastcv::resizeDownBy2(inputImage, resized_image);

    // Check if the output size is correct
    EXPECT_EQ(resized_image.size().width, size.width * 0.5);
    EXPECT_EQ(resized_image.size().height, size.height * 0.5);
}

TEST_P(ResizeBy4Test, ResizeBy4) {

    //Size size = get<0>(GetParam());
    Size size = GetParam();
    cv::Mat inputImage(size, CV_8UC1);
    randu(inputImage, Scalar::all(0), Scalar::all(255)); // Fill with random values

    Size dsize;
    cv::Mat resized_image;

    // Resize the image by a factor of 4
    cv::fastcv::resizeDownBy4(inputImage, resized_image);

    // Check if the output size is correct
    EXPECT_EQ(resized_image.size().width, size.width * 0.25);
    EXPECT_EQ(resized_image.size().height, size.height * 0.25);
}

INSTANTIATE_TEST_CASE_P(
    ResizeTests,
    ResizeBy2Test,
    ::testing::Values(cv::Size(640, 480), cv::Size(1280, 720), cv::Size(1920, 1080)
));

INSTANTIATE_TEST_CASE_P(
    ResizeTests,
    ResizeBy4Test,
    ::testing::Values(cv::Size(640, 480), cv::Size(1280, 720), cv::Size(1920, 1080)
));


}} // namespaces opencv_test, ::