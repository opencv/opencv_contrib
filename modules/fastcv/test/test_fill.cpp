/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "test_precomp.hpp"

namespace opencv_test { namespace {

typedef tuple<cv::Size /*imgSize*/, int /*nPts*/, int /*channels*/> FillConvexTestParams;
class FillConvexTest : public ::testing::TestWithParam<FillConvexTestParams> {};

TEST_P(FillConvexTest, randomDraw)
{
    auto p = GetParam();

    Size imgSize = std::get<0>(p);
    int nPts     = std::get<1>(p);
    int channels = std::get<2>(p);

    cv::RNG rng = cv::theRNG();

    std::vector<Point> allPts, contour;
    for (int i = 0; i < nPts; i++)
    {
        allPts.push_back(Point(rng() % imgSize.width, rng() % imgSize.height));
    }
    cv::convexHull(allPts, contour);

    Scalar color(rng() % 256, rng() % 256, rng() % 256);

    Mat imgRef(imgSize, CV_MAKE_TYPE(CV_8U, channels), Scalar(0));
    Mat imgFast = imgRef.clone();

    cv::fillConvexPoly(imgRef, contour, color);
    cv::fastcv::fillConvexPoly(imgFast, contour, color);

    double normInf = cvtest::norm(imgRef, imgFast, cv::NORM_INF);
    double normL2  = cvtest::norm(imgRef, imgFast, cv::NORM_L2);

    EXPECT_EQ(normInf, 0);
    EXPECT_EQ(normL2, 0);
}

TEST_P(FillConvexTest, circle)
{
    auto p = GetParam();

    Size imgSize = std::get<0>(p);
    int nPts     = std::get<1>(p);
    int channels = std::get<2>(p);

    cv::RNG rng = cv::theRNG();

    float r = std::min(imgSize.width, imgSize.height) / 2 * 0.9f;
    float angle = CV_PI * 2.0f / (float)nPts;
    std::vector<Point> contour;
    for (int i = 0; i < nPts; i++)
    {
        Point2f pt(r * cos((float)i * angle),
                   r * sin((float)i * angle));
        contour.push_back({ imgSize.width  / 2 + int(pt.x),
                            imgSize.height / 2 + int(pt.y)});
    }
    Scalar color(rng() % 256, rng() % 256, rng() % 256);

    Mat imgRef(imgSize, CV_MAKE_TYPE(CV_8U, channels), Scalar(0));
    Mat imgFast = imgRef.clone();

    cv::fillConvexPoly(imgRef, contour, color);
    cv::fastcv::fillConvexPoly(imgFast, contour, color);

    double normInf = cvtest::norm(imgRef, imgFast, cv::NORM_INF);
    double normL2  = cvtest::norm(imgRef, imgFast, cv::NORM_L2);

    EXPECT_EQ(normInf, 0);
    EXPECT_EQ(normL2, 0);
}

INSTANTIATE_TEST_CASE_P(FastCV_Extension, FillConvexTest,
                        ::testing::Combine(testing::Values(Size(640, 480), Size(512, 512), Size(1920, 1080)), // imgSize
                                           testing::Values(4, 64, 1024), // nPts
                                           testing::Values(1, 2, 3, 4))); // channels

}} // namespaces opencv_test, ::