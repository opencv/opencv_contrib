/*
 * Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(resizeDownBy2, accuracy)
{
    cv::Mat inputImage = cv::imread(cvtest::findDataFile("cv/shared/box_in_scene.png"), cv::IMREAD_GRAYSCALE);

    cv::Mat resized_image;

    cv::fastcv::resizeDown(inputImage, resized_image, cv::Size(inputImage.cols / 2, inputImage.rows / 2), 0, 0);

    EXPECT_FALSE(resized_image.empty());

    cv::Mat resizedImageOpenCV;
    cv::resize(inputImage, resizedImageOpenCV, cv::Size(inputImage.cols / 2, inputImage.rows / 2), 0, 0, INTER_AREA);

    double maxVal = cv::norm(resized_image, resizedImageOpenCV, cv::NORM_INF);

    CV_Assert(maxVal < 10 && "Difference between images is too high!");
}

TEST(resizeDownBy4, accuracy)
{
    cv::Mat inputImage = cv::imread(cvtest::findDataFile("cv/shared/box_in_scene.png"), cv::IMREAD_GRAYSCALE);

    Size dsize;
    cv::Mat resized_image;

    cv::fastcv::resizeDown(inputImage, resized_image, dsize, 0.25, 0.25);

    EXPECT_FALSE(resized_image.empty());

    cv::Mat resizedImageOpenCV;
    cv::resize(inputImage, resizedImageOpenCV, cv::Size(inputImage.cols / 4, inputImage.rows / 4), 0, 0, INTER_AREA);

    double maxVal = cv::norm(resized_image, resizedImageOpenCV, cv::NORM_INF);

    CV_Assert(maxVal < 10 && "Difference between images is too high!");
}

TEST(resizeDownMN, accuracy)
{
    cv::Mat inputImage = cv::imread(cvtest::findDataFile("cv/cascadeandhog/images/class57.png"), cv::IMREAD_GRAYSCALE);

    cv::Mat resized_image;

    cv::fastcv::resizeDown(inputImage, resized_image, cv::Size(800, 640), 0, 0);

    EXPECT_FALSE(resized_image.empty());

    cv::Mat resizedImageOpenCV;
    cv::resize(inputImage, resizedImageOpenCV, cv::Size(800, 640), 0, 0, INTER_LINEAR);

    double maxVal = cv::norm(resized_image, resizedImageOpenCV, cv::NORM_INF);

    CV_Assert(maxVal < 78 && "Difference between images is too high!");
}

TEST(resizeDownInterleaved, accuracy)
{
    cv::Mat inputImage = cv::Mat::zeros(512, 512, CV_8UC2);
    cv::randu(inputImage, cv::Scalar(0), cv::Scalar(255));


    Size dsize;
    cv::Mat resized_image;

    cv::fastcv::resizeDown(inputImage, resized_image, dsize, 0.500, 0.125);

    EXPECT_FALSE(resized_image.empty());


    cv::Mat resizedImageOpenCV;
    cv::resize(inputImage, resizedImageOpenCV, dsize, 0.500, 0.125, INTER_AREA);

    double maxVal = cv::norm(resized_image, resizedImageOpenCV, cv::NORM_INF);

    CV_Assert(maxVal < 10 && "Difference between images is too high!");
}

}} // namespaces opencv_test, ::