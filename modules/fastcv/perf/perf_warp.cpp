/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "perf_precomp.hpp"

namespace opencv_test {

typedef perf::TestBaseWithParam<Size> WarpPerspective2PlanePerfTest;

PERF_TEST_P(WarpPerspective2PlanePerfTest, run,
    ::testing::Values(perf::szVGA, perf::sz720p, perf::sz1080p))
{
    cv::Size dstSize = GetParam();
    cv::Mat img = imread(cvtest::findDataFile("cv/shared/baboon.png"));
    Mat src(img.rows, img.cols, CV_8UC1);
    cvtColor(img,src,cv::COLOR_BGR2GRAY);
    cv::Mat dst1, dst2, mat;
    mat.create(3,3,CV_32FC1);
    dst1.create(dstSize,CV_8UC1);
    dst2.create(dstSize,CV_8UC1);

    RNG& rng = cv::theRNG();
    Point2f s[4], d[4];

    s[0] = Point2f(0,0);
    d[0] = Point2f(0,0);
    s[1] = Point2f(src.cols-1.f,0);
    d[1] = Point2f(dst1.cols-1.f,0);
    s[2] = Point2f(src.cols-1.f,src.rows-1.f);
    d[2] = Point2f(dst1.cols-1.f,dst1.rows-1.f);
    s[3] = Point2f(0,src.rows-1.f);
    d[3] = Point2f(0,dst1.rows-1.f);

    float buffer[16];
    Mat tmp( 1, 16, CV_32FC1, buffer );
    rng.fill( tmp, 1, Scalar::all(0.), Scalar::all(0.1) );

    for(int i = 0; i < 4; i++ )
    {
        s[i].x += buffer[i*4]*src.cols/2;
        s[i].y += buffer[i*4+1]*src.rows/2;
        d[i].x += buffer[i*4+2]*dst1.cols/2;
        d[i].y += buffer[i*4+3]*dst1.rows/2;
    }

    cv::getPerspectiveTransform( s, d ).convertTo( mat, mat.depth() );
    // Invert the perspective matrix
    invert(mat,mat);

    while (next())
    {
        startTimer();
        cv::fastcv::warpPerspective2Plane(src, src, dst1, dst2, mat, dstSize);
        stopTimer();
    }

    SANITY_CHECK_NOTHING();
}

} //namespace