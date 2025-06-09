/*
 * Copyright (c) 2024-2025 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "test_precomp.hpp"

namespace opencv_test { namespace {

static void getInvertMatrix(Mat& src, Size dstSize, Mat& M)
{
    RNG& rng = cv::theRNG();
    Point2f s[4], d[4];

    s[0] = Point2f(0,0);
    d[0] = Point2f(0,0);
    s[1] = Point2f(src.cols-1.f,0);
    d[1] = Point2f(dstSize.width-1.f,0);
    s[2] = Point2f(src.cols-1.f,src.rows-1.f);
    d[2] = Point2f(dstSize.width-1.f,dstSize.height-1.f);
    s[3] = Point2f(0,src.rows-1.f);
    d[3] = Point2f(0,dstSize.height-1.f);

    float buffer[16];
    Mat tmp( 1, 16, CV_32FC1, buffer );
    rng.fill( tmp, 1, Scalar::all(0.), Scalar::all(0.1) );

    for(int i = 0; i < 4; i++ )
    {
        s[i].x += buffer[i*4]*src.cols/2;
        s[i].y += buffer[i*4+1]*src.rows/2;
        d[i].x += buffer[i*4+2]*dstSize.width/2;
        d[i].y += buffer[i*4+3]*dstSize.height/2;
    }

    cv::getPerspectiveTransform( s, d ).convertTo( M, M.depth() );

    // Invert the perspective matrix
    invert(M,M);
}

static cv::Mat getInverseAffine(const cv::Mat& affine)
{
    // Extract the 2x2 part
    cv::Mat rotationScaling = affine(cv::Rect(0, 0, 2, 2));

    // Invert the 2x2 part
    cv::Mat inverseRotationScaling;
    cv::invert(rotationScaling, inverseRotationScaling);

    // Extract the translation part
    cv::Mat translation = affine(cv::Rect(2, 0, 1, 2));

    // Compute the new translation
    cv::Mat inverseTranslation = -inverseRotationScaling * translation;

    // Construct the inverse affine matrix
    cv::Mat inverseAffine = cv::Mat::zeros(2, 3, CV_32F);
    inverseRotationScaling.copyTo(inverseAffine(cv::Rect(0, 0, 2, 2)));
    inverseTranslation.copyTo(inverseAffine(cv::Rect(2, 0, 1, 2)));

    return inverseAffine;
}

typedef testing::TestWithParam<cv::Size> WarpPerspective2Plane;

TEST_P(WarpPerspective2Plane, accuracy)
{
    cv::Size dstSize = GetParam();
    cv::Mat src = imread(cvtest::findDataFile("cv/shared/baboon.png"), cv::IMREAD_GRAYSCALE);
    EXPECT_FALSE(src.empty());

    cv::Mat dst1, dst2, matrix, ref1, ref2;
    matrix.create(3, 3, CV_32FC1);

    getInvertMatrix(src, dstSize, matrix);

    cv::fastcv::warpPerspective2Plane(src, src, dst1, dst2, matrix, dstSize);
    cv::warpPerspective(src, ref1, matrix, dstSize, (cv::INTER_LINEAR | cv::WARP_INVERSE_MAP),cv::BORDER_CONSTANT,Scalar(0));
    cv::warpPerspective(src, ref2, matrix, dstSize, (cv::INTER_LINEAR | cv::WARP_INVERSE_MAP),cv::BORDER_CONSTANT,Scalar(0));

    cv::Mat difference1, difference2, mask1, mask2;
    cv::absdiff(dst1, ref1, difference1);
    cv::absdiff(dst2, ref2, difference2);

    // There are 1 or 2 difference in pixel value because algorithm is different, ignore those difference
    cv::threshold(difference1, mask1, 5, 255, cv::THRESH_BINARY);
    cv::threshold(difference2, mask2, 5, 255, cv::THRESH_BINARY);
    int num_diff_pixels_1 = cv::countNonZero(mask1);
    int num_diff_pixels_2 = cv::countNonZero(mask2);

    // The border is different
    EXPECT_LT(num_diff_pixels_1, (dstSize.width+dstSize.height)*5);
    EXPECT_LT(num_diff_pixels_2, (dstSize.width+dstSize.height)*5);
}

typedef testing::TestWithParam<tuple<Size, int, int>> WarpPerspective;

TEST_P(WarpPerspective, accuracy)
{
    cv::Size dstSize = get<0>(GetParam());
    int interplation = get<1>(GetParam());
    int borderType   = get<2>(GetParam());
    cv::Scalar borderValue = Scalar::all(100);

    cv::Mat src = imread(cvtest::findDataFile("cv/shared/baboon.png"), cv::IMREAD_GRAYSCALE);
    EXPECT_FALSE(src.empty());

    cv::Mat dst, matrix, ref;
    matrix.create(3, 3, CV_32FC1);

    getInvertMatrix(src, dstSize, matrix);

    cv::fastcv::warpPerspective(src, dst, matrix, dstSize, interplation, borderType, borderValue);
    cv::warpPerspective(src, ref, matrix, dstSize, (interplation | cv::WARP_INVERSE_MAP), borderType, borderValue);

    cv::Mat difference, mask;
    cv::absdiff(dst, ref, difference);
    cv::threshold(difference, mask, 10, 255, cv::THRESH_BINARY);
    int num_diff_pixels = cv::countNonZero(mask);

    EXPECT_LT(num_diff_pixels, src.size().area()*0.05);
}

INSTANTIATE_TEST_CASE_P(FastCV_Extension, WarpPerspective,Combine(
                   ::testing::Values(perf::szVGA, perf::sz720p, perf::sz1080p),
                   ::testing::Values(INTER_NEAREST, INTER_LINEAR, INTER_AREA),
                   ::testing::Values(BORDER_CONSTANT, BORDER_REPLICATE, BORDER_TRANSPARENT)
));
INSTANTIATE_TEST_CASE_P(FastCV_Extension, WarpPerspective2Plane, Values(perf::szVGA, perf::sz720p, perf::sz1080p));

TEST(WarpAffine3ChannelTest, accuracy)
{
    cv::Mat src = imread(cvtest::findDataFile("cv/shared/baboon.png"));

    // Define the transformation matrix
    cv::Mat M = (cv::Mat_<float>(2, 3) << 2.0, 0, -50.0, 0, 2.0, -50.0);

    cv::Size dsize(src.cols, src.rows);

    cv::Mat dst;

    cv::fastcv::warpAffine(src, dst, M, dsize);

    EXPECT_FALSE(dst.empty());
}

TEST(WarpAffineROITest, accuracy)
{
    cv::Mat src = cv::imread(cvtest::findDataFile("cv/shared/baboon.png"), cv::IMREAD_GRAYSCALE);

    // Define the position and affine matrix
    cv::Point2f position(src.cols / 2.0f, src.rows / 2.0f);

    float angle = 180.0; // Rotation angle in degrees
    float radians = angle * CV_PI / 180.0;
    cv::Mat affine = (cv::Mat_<float>(2, 2) << cos(radians), -sin(radians), sin(radians), cos(radians));

    cv::Mat patch;
    cv::Mat roi = src(cv::Rect(0, 0, 100, 100));
    cv::fastcv::warpAffine(roi, patch, affine, cv::Size(100, 100));

    EXPECT_FALSE(patch.empty());
    EXPECT_EQ(patch.size(), cv::Size(100, 100));
    EXPECT_EQ(patch.type(), CV_8UC1);
}

typedef testing::TestWithParam<tuple<int, int>> WarpAffineTest;

TEST_P(WarpAffineTest, accuracy)
{
    // Load the source image
    cv::Mat src = cv::imread(cvtest::findDataFile("cv/shared/baboon.png"), cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(src.empty());

    float angle = 30.0;// Rotation angle in degrees
    float scale = 0.5;// Scale factor
    cv::Mat affine = cv::getRotationMatrix2D(cv::Point2f(100, 100), angle, scale);

    // Compute the inverse affine matrix
    cv::Mat inverseAffine = getInverseAffine(affine);

    // Define the destination size
    cv::Size dsize(src.cols, src.rows);

    // Define the output matrix
    cv::Mat dst;

    // Get the parameters
    int interpolation = std::get<0>(GetParam());
    int borderValue = std::get<1>(GetParam());

    // Perform the affine transformation
    cv::fastcv::warpAffine(src, dst, inverseAffine, dsize, interpolation, borderValue);

    // Check that the output is not empty
    EXPECT_FALSE(dst.empty());
}

INSTANTIATE_TEST_CASE_P(
    FastCV_Extension,
    WarpAffineTest,
    ::testing::Combine(
        ::testing::Values(INTER_NEAREST, INTER_LINEAR, INTER_AREA),
        ::testing::Values(0, 255) // Black and white borders
    )
);

}
}