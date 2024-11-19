/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "perf_precomp.hpp"

namespace opencv_test {

typedef std::tuple<cv::Size, MatType, int /*iterations*/, float /*epsilon*/, Size /*winSize*/> MeanShiftPerfParams;
typedef perf::TestBaseWithParam<MeanShiftPerfParams> MeanShiftPerfTest;

PERF_TEST_P(MeanShiftPerfTest, run,
    ::testing::Combine(::testing::Values(Size(128, 128), Size(640, 480), Size(800, 600)),
                       ::testing::Values(CV_8U, CV_32S, CV_32F), // type
                       ::testing::Values(2, 10, 100), // nIterations
                       ::testing::Values(0.01f, 0.1f, 1.f, 10.f), // epsilon
                       ::testing::Values(Size(8, 8), Size(13, 48), Size(64, 64)) // window size
                       )
           )
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
    while(next())
    {
        startTimer();
        cv::fastcv::meanShift(src, window, termCrit);
        stopTimer();
    }

    SANITY_CHECK_NOTHING();
}

} // namespace
