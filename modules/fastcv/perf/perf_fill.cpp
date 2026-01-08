/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "perf_precomp.hpp"

namespace opencv_test {

typedef tuple<cv::Size /*imgSize*/, int /*nPts*/, int /*channels*/> FillConvexPerfParams;
typedef perf::TestBaseWithParam<FillConvexPerfParams> FillConvexPerfTest;

PERF_TEST_P(FillConvexPerfTest, randomDraw, Combine(
                testing::Values(Size(640, 480), Size(512, 512), Size(1920, 1080)),
                testing::Values(4, 64, 1024),
                testing::Values(1, 2, 3, 4)
            ))
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

    Mat img(imgSize, CV_MAKE_TYPE(CV_8U, channels), Scalar(0));

    while(next())
    {
        img = 0;
        startTimer();
        cv::fastcv::fillConvexPoly(img, contour, color);
        stopTimer();
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(FillConvexPerfTest, circle, Combine(
                testing::Values(Size(640, 480), Size(512, 512), Size(1920, 1080)),
                testing::Values(4, 64, 1024),
                testing::Values(1, 2, 3, 4)
            ))
{
    auto p = GetParam();

    Size imgSize = std::get<0>(p);
    int nPts     = std::get<1>(p);
    int channels = std::get<2>(p);

    cv::RNG rng = cv::theRNG();

    float r = std::min(imgSize.width, imgSize.height) / 2 * 0.9f;
    float angle = CV_PI * 2.0f / (float)nPts;
    std::vector<Point2i> contour;
    for (int i = 0; i < nPts; i++)
    {
        Point2f pt(r * cos((float)i * angle),
                   r * sin((float)i * angle));
        contour.push_back({ imgSize.width  / 2 + int(pt.x),
                            imgSize.height / 2 + int(pt.y)});
    }
    Scalar color(rng() % 256, rng() % 256, rng() % 256);

    Mat img(imgSize, CV_MAKE_TYPE(CV_8U, channels), Scalar(0));

    while(next())
    {
        img = 0;
        startTimer();
        cv::fastcv::fillConvexPoly(img, contour, color);
        stopTimer();
    }

    SANITY_CHECK_NOTHING();
}


} // namespace
