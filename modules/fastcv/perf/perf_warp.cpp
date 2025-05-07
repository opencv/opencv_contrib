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

} //namespace