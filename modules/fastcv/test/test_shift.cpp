/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "test_precomp.hpp"

namespace opencv_test { namespace {

typedef std::tuple<cv::Size, MatType, int /*iterations*/, float /*epsilon*/, Size /*winSize*/> MeanShiftTestParams;
class MeanShiftTest : public ::testing::TestWithParam<MeanShiftTestParams> {};

TEST_P(MeanShiftTest, accuracy)
{
    auto p = GetParam();
    cv::Size size = std::get<0>(p);
    MatType type  = std::get<1>(p);
    int iters     = std::get<2>(p);
    float eps     = std::get<3>(p);
    Size winSize  = std::get<4>(p);

    RNG& rng = cv::theRNG();

    const int nPts = 20;
    Mat ptsMap(size, CV_8UC1, Scalar(255));
    for(size_t i = 0; i < nPts; ++i)
    {
        ptsMap.at<uchar>(rng() % size.height, rng() % size.width) = 0;
    }
    Mat distTrans(size, CV_8UC1);
    cv::distanceTransform(ptsMap, distTrans, DIST_L2, DIST_MASK_PRECISE);
    Mat vsrc = 255 - distTrans;
    Mat src;
    vsrc.convertTo(src, type);

    Point startPt(rng() % (size.width  - winSize.width),
                  rng() % (size.height - winSize.height));
    Rect startRect(startPt, winSize);

    cv::TermCriteria termCrit( TermCriteria::EPS + TermCriteria::MAX_ITER, iters, eps);

    Rect window = startRect;
    cv::fastcv::meanShift(src, window, termCrit);

    Rect windowRef = startRect;
    cv::meanShift(vsrc, windowRef, termCrit);

    if (cvtest::debugLevel > 0)
    {
        Mat draw;
        cvtColor(vsrc, draw, COLOR_GRAY2RGB);
        cv::rectangle(draw, startRect, Scalar(0, 0, 255));
        cv::rectangle(draw, window, Scalar(255, 255, 0));
        cv::rectangle(draw, windowRef, Scalar(0, 255, 0));
        std::string stype = (type == CV_8U ? "8U" : (type == CV_32S ? "32S" : (type == CV_32F ? "F" : "?")));
        cv::imwrite(cv::format("src_%dx%d_%s_%dit_%feps_%dx%d.png", size.width, size.height, stype.c_str(),
                                                                    iters, eps, winSize.width, winSize.height),
                    draw);
    }

    cv::Point diff = (window.tl() - windowRef.tl());
    double dist = std::sqrt(diff.ddot(diff));

    EXPECT_LE(dist, 3.0);
}

INSTANTIATE_TEST_CASE_P(FastCV_Extension, MeanShiftTest,
                         ::testing::Combine(::testing::Values(Size(128, 128), Size(640, 480), Size(800, 600)),
                                            ::testing::Values(CV_8U, CV_32S, CV_32F), // type
                                            ::testing::Values(2, 10, 100), // nIterations
                                            ::testing::Values(0.01f, 0.1f, 1.f, 10.f), // epsilon
                                            ::testing::Values(Size(8, 8), Size(13, 48), Size(64, 64)) // window size
                                            ));

}} // namespaces opencv_test, ::
