/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "test_precomp.hpp"

namespace opencv_test { namespace {

typedef testing::TestWithParam<cv::Size> WarpPerspective2Plane;

TEST_P(WarpPerspective2Plane, accuracy)
{
    cv::Size dstSize = GetParam();
    cv::Mat img = imread(cvtest::findDataFile("cv/shared/baboon.png"));
    Mat src(img.rows, img.cols, CV_8UC1);
    cvtColor(img,src,cv::COLOR_BGR2GRAY);
    cv::Mat dst1, dst2, mat, ref1, ref2;
    mat.create(3,3,CV_32FC1);
    dst1.create(dstSize,CV_8UC1);
    dst2.create(dstSize,CV_8UC1);

    RNG rng = RNG((uint64)-1);
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

    cv::fastcv::warpPerspective2Plane(src, src, dst1, dst2, mat, dstSize);
    cv::warpPerspective(src,ref1,mat,dstSize,(cv::INTER_LINEAR | cv::WARP_INVERSE_MAP));
    cv::warpPerspective(src,ref2,mat,dstSize,(cv::INTER_LINEAR | cv::WARP_INVERSE_MAP));

    cv::Mat difference1, difference2, mask1,mask2;
    cv::absdiff(dst1, ref1, difference1);
    cv::absdiff(dst2, ref2, difference2);
    cv::threshold(difference1, mask1, 5, 255, cv::THRESH_BINARY);
    cv::threshold(difference2, mask2, 5, 255, cv::THRESH_BINARY);
    int num_diff_pixels_1 = cv::countNonZero(mask1);
    int num_diff_pixels_2 = cv::countNonZero(mask2);

    EXPECT_LT(num_diff_pixels_1, src.size().area()*0.02);
    EXPECT_LT(num_diff_pixels_2, src.size().area()*0.02);
}

INSTANTIATE_TEST_CASE_P(FastCV_Extension, WarpPerspective2Plane, Values(perf::szVGA, perf::sz720p, perf::sz1080p));

}
}