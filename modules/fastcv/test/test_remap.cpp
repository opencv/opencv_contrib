/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "test_precomp.hpp"

namespace opencv_test { namespace {

class RemapTest : public ::testing::TestWithParam<tuple<int, int, Size>> {
protected:
    void SetUp() override {
        // Generate random source data
        Size size = get<2>(GetParam());
        src = Mat(size, get<0>(GetParam()));
        randu(src, Scalar::all(0), Scalar::all(255)); // Fill with random values

        ASSERT_FALSE(src.empty()) << "Unable to generate the image!";

        // Create map matrices
        map_x.create(src.size(), CV_32FC1);
        map_y.create(src.size(), CV_32FC1);

        // Initialize the map matrices
        for (int i = 0; i < src.rows; i++) {
            for (int j = 0; j < src.cols; j++) {
                map_x.at<float>(i, j) = static_cast<float>(src.cols - j); //Flips the image horizonally
                map_y.at<float>(i, j) = static_cast<float>(i); //Keep y coordinate unchanged
            }
        }
    }

    Mat src, map_x, map_y, dst;
};

class RemapTestRGBA : public ::testing::TestWithParam<tuple<int, int, Size>> {
protected:
    void SetUp() override {
        // Generate random source data
        Size size = get<2>(GetParam());
        src = Mat(size, get<0>(GetParam()));
        randu(src, Scalar::all(0), Scalar::all(255)); // Fill with random values

        ASSERT_FALSE(src.empty()) << "Unable to generate the image!";

        // Create map matrices
        map_x.create(src.size(), CV_32FC1);
        map_y.create(src.size(), CV_32FC1);

        // Initialize the map matrices
        for (int i = 0; i < src.rows; i++) {
            for (int j = 0; j < src.cols; j++) {
                map_x.at<float>(i, j) = static_cast<float>(src.cols - j); //Flips the image horizonally
                map_y.at<float>(i, j) = static_cast<float>(i); //Keep y coordinate unchanged
            }
        }
    }

    Mat src, map_x, map_y, dst;
};

TEST_P(RemapTest, accuracy)
{
    int type = get<0>(GetParam());
    int interpolation = get<1>(GetParam());

    // Convert source image to the specified type
    Mat src_converted;
    src.convertTo(src_converted, type);

    cv::fastcv::remap(src_converted, dst, map_x, map_y, interpolation);

    // Check if the remapped image is not empty
    ASSERT_FALSE(dst.empty()) << "Remapped image is empty!";

    cv::Mat remapOpenCV;
    cv::remap(src_converted, remapOpenCV, map_x, map_y, interpolation);

    // Calculate the maximum difference
    double maxVal = cv::norm(dst, remapOpenCV, cv::NORM_INF);

    // Assert if the difference is acceptable (max difference should be less than 10)
    CV_Assert(maxVal < 10 && "Difference between images is too high!");
}

TEST_P(RemapTestRGBA, accuracy)
{
    int type = get<0>(GetParam());
    int interpolation = get<1>(GetParam());

    // Convert source image to the specified type
    Mat src_converted;
    src.convertTo(src_converted, type);

    cv::fastcv::remapRGBA(src_converted, dst, map_x, map_y, interpolation);

    // Check if the remapped image is not empty
    ASSERT_FALSE(dst.empty()) << "Remapped image is empty!";

    cv::Mat remapOpenCV;
    cv::remap(src_converted, remapOpenCV, map_x, map_y, interpolation);

    // Calculate the maximum difference
    double maxVal = cv::norm(dst, remapOpenCV, cv::NORM_INF);

    // Assert if the difference is acceptable (max difference should be less than 10)
    CV_Assert(maxVal < 10 && "Difference between images is too high!");
}


INSTANTIATE_TEST_CASE_P(
    RemapTests,
    RemapTest,
    ::testing::Combine(
        ::testing::Values(CV_8UC1),
        ::testing::Values(INTER_LINEAR, INTER_NEAREST),
        ::testing::Values(Size(640, 480), Size(1280, 720), Size(1920, 1080))
    )
);

INSTANTIATE_TEST_CASE_P(
    RemapTests,
    RemapTestRGBA,
    ::testing::Combine(
        ::testing::Values(CV_8UC4),
        ::testing::Values(INTER_LINEAR, INTER_NEAREST),
        ::testing::Values(Size(640, 480), Size(1280, 720), Size(1920, 1080))
    )
);

}} // namespaces opencv_test, ::