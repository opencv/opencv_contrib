/*
 * Copyright (c) 2024-2025 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "perf_precomp.hpp"

namespace opencv_test {

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

typedef perf::TestBaseWithParam<Size> WarpPerspective2PlanePerfTest;

PERF_TEST_P(WarpPerspective2PlanePerfTest, run,
    ::testing::Values(perf::szVGA, perf::sz720p, perf::sz1080p))
{
    cv::Size dstSize = GetParam();
    cv::Mat img = imread(cvtest::findDataFile("cv/shared/baboon.png"));
    Mat src(img.rows, img.cols, CV_8UC1);
    cvtColor(img,src,cv::COLOR_BGR2GRAY);
    cv::Mat dst1, dst2, matrix;
    matrix.create(3,3,CV_32FC1);

    getInvertMatrix(src, dstSize, matrix);

    while (next())
    {
        startTimer();
        cv::fastcv::warpPerspective2Plane(src, src, dst1, dst2, matrix, dstSize);
        stopTimer();
    }

    SANITY_CHECK_NOTHING();
}

typedef perf::TestBaseWithParam<tuple<Size, int, int>> WarpPerspectivePerfTest;

PERF_TEST_P(WarpPerspectivePerfTest, run,
    ::testing::Combine( ::testing::Values(perf::szVGA, perf::sz720p, perf::sz1080p),
                        ::testing::Values(INTER_NEAREST, INTER_LINEAR, INTER_AREA),
                        ::testing::Values(BORDER_CONSTANT, BORDER_REPLICATE, BORDER_TRANSPARENT)))
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

    while (next())
    {
        startTimer();
        cv::fastcv::warpPerspective(src, dst, matrix, dstSize, interplation, borderType, borderValue);
        stopTimer();
    }

    SANITY_CHECK_NOTHING();
}

typedef TestBaseWithParam< tuple<MatType, Size> > WarpAffine3ChannelPerf;

PERF_TEST_P(WarpAffine3ChannelPerf, run, Combine(
            Values(CV_8UC3),
            Values( szVGA, sz720p, sz1080p)
))
{
    Size sz, szSrc(512, 512);
    int dataType;
    dataType   = get<0>(GetParam());
    sz         = get<1>(GetParam());

    cv::Mat src(szSrc, dataType), dst(sz, dataType);

    cvtest::fillGradient(src);

    //Affine matrix
    float angle = 30.0; // Rotation angle in degrees
    float scale = 2.2;  // Scale factor
    cv::Mat affine = cv::getRotationMatrix2D(cv::Point2f(100, 100), angle, scale);

    // Compute the inverse affine matrix
    cv::Mat inverseAffine = getInverseAffine(affine);

    // Create the dstBorder array
    Mat dstBorder;

    declare.in(src).out(dst);

    while (next())
    {
        startTimer();
        cv::fastcv::warpAffine(src, dst, inverseAffine, sz);
        stopTimer();
    }

    SANITY_CHECK_NOTHING();
}

typedef perf::TestBaseWithParam<std::tuple<cv::Size, cv::Point2f, cv::Mat>> WarpAffineROIPerfTest;

PERF_TEST_P(WarpAffineROIPerfTest, run, ::testing::Combine(
    ::testing::Values(cv::Size(50, 50), cv::Size(100, 100)), // patch size
    ::testing::Values(cv::Point2f(50.0f, 50.0f), cv::Point2f(100.0f, 100.0f)), // position
    ::testing::Values((cv::Mat_<float>(2, 2) << 1, 0, 0, 1), // identity matrix
                      (cv::Mat_<float>(2, 2) << cos(CV_PI), -sin(CV_PI), sin(CV_PI), cos(CV_PI))) // rotation matrix
))
{
    cv::Size patchSize = std::get<0>(GetParam());
    cv::Point2f position = std::get<1>(GetParam());
    cv::Mat affine = std::get<2>(GetParam());

    cv::Mat src = cv::imread(cvtest::findDataFile("cv/shared/baboon.png"), cv::IMREAD_GRAYSCALE);
    
    // Create ROI with top-left at the specified position
    cv::Rect roiRect(static_cast<int>(position.x), static_cast<int>(position.y), patchSize.width, patchSize.height);

    // Ensure ROI is within image bounds
    roiRect = roiRect & cv::Rect(0, 0, src.cols, src.rows);
    cv::Mat roi = src(roiRect);

    cv::Mat patch;

    while (next())
    {
        startTimer();
        cv::fastcv::warpAffine(roi, patch, affine, patchSize);
        stopTimer();
    }

    SANITY_CHECK_NOTHING();
}

typedef TestBaseWithParam<tuple<int, int> > WarpAffinePerfTest;

PERF_TEST_P(WarpAffinePerfTest, run, ::testing::Combine(
    ::testing::Values(cv::InterpolationFlags::INTER_NEAREST, cv::InterpolationFlags::INTER_LINEAR, cv::InterpolationFlags::INTER_AREA),
    ::testing::Values(0, 255) // Black and white borders
))
{
    // Load the source image
    cv::Mat src = cv::imread(cvtest::findDataFile("cv/shared/baboon.png"), cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(src.empty());

    // Generate random values for the affine matrix
    std::srand(std::time(0));
    float angle = static_cast<float>(std::rand() % 360); // Random angle between 0 and 360 degrees
    float scale = static_cast<float>(std::rand() % 200) / 100.0f + 0.5f; // Random scale between 0.5 and 2.5
    float tx = static_cast<float>(std::rand() % 100) - 50; // Random translation between -50 and 50
    float ty = static_cast<float>(std::rand() % 100) - 50; // Random translation between -50 and 50
    float radians = angle * CV_PI / 180.0;
    cv::Mat affine = (cv::Mat_<float>(2, 3) << scale * cos(radians), -scale * sin(radians), tx,
                                               scale * sin(radians),  scale * cos(radians), ty);

    // Compute the inverse affine matrix
    cv::Mat inverseAffine = getInverseAffine(affine);

    // Define the destination size
    cv::Size dsize(src.cols, src.rows);

    // Define the output matrix
    cv::Mat dst;

    // Get the parameters
    int interpolation = std::get<0>(GetParam());
    int borderValue = std::get<1>(GetParam());

    while (next())
    {
        startTimer();
        cv::fastcv::warpAffine(src, dst, inverseAffine, dsize, interpolation, borderValue);
        stopTimer();
    }

    SANITY_CHECK_NOTHING();
}

} //namespace