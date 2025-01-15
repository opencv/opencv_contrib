/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "perf_precomp.hpp"

namespace opencv_test {

typedef std::tuple<int /*winSize*/, bool /*useSobelPyramid*/, bool /*useInitialEstimate*/ > TrackingTestParams;
class TrackingTest : public ::perf::TestBaseWithParam<TrackingTestParams> {};

PERF_TEST_P(TrackingTest, checkAllVersions,
    ::testing::Combine(::testing::Values(5, 7, 9), // window size
                       ::testing::Bool(),          // useSobelPyramid
                       ::testing::Bool()           // useInitialEstimate
                      ))
{
    auto par = GetParam();

    int winSz               = std::get<0>(par);
    bool useSobelPyramid    = std::get<1>(par);
    bool useInitialEstimate = std::get<2>(par);

    cv::Mat src = imread(cvtest::findDataFile("cv/shared/baboon.png"), cv::IMREAD_GRAYSCALE);

    double ang = 5.0 * CV_PI / 180.0;
    cv::Matx33d tr = {
        cos(ang), -sin(ang), 1,
        sin(ang),  cos(ang), 2,
               0,         0, 1
    };
    cv::Matx33d orig {
        1, 0, -(double)src.cols / 2,
        0, 1, -(double)src.rows / 2,
        0, 0, 1
    };
    cv::Matx33d back {
        1, 0, (double)src.cols / 2,
        0, 1, (double)src.rows / 2,
        0, 0, 1
    };
    cv::Matx23d trans = (back * tr * orig).get_minor<2, 3>(0, 0);

    cv::Mat dst;
    cv::warpAffine(src, dst, trans, src.size());

    int nLevels = 4;
    std::vector<cv::Mat> srcPyr, dstPyr;

    cv::buildPyramid(src, srcPyr, nLevels - 1);
    cv::buildPyramid(dst, dstPyr, nLevels - 1);

    cv::Matx23f transf = trans;
    int nPts = 32;
    std::vector<cv::Point2f> ptsIn, ptsEst, ptsExpected;
    for (int i = 0; i < nPts; i++)
    {
        cv::Point2f p { (((float)cv::theRNG())*0.5f + 0.25f) * src.cols,
                        (((float)cv::theRNG())*0.5f + 0.25f) * src.rows };
        ptsIn.push_back(p);
        ptsExpected.push_back(transf * cv::Vec3f(p.x, p.y, 1.0));
        ptsEst.push_back(p);
    }

    cv::TermCriteria termCrit;
    termCrit.type = cv::TermCriteria::COUNT | cv::TermCriteria::EPS;
    termCrit.maxCount = 7;
    termCrit.epsilon = 0.03f * 0.03f;

    std::vector<cv::Mat> srcDxPyr, srcDyPyr;
    if (useSobelPyramid)
    {
        cv::fastcv::sobelPyramid(srcPyr, srcDxPyr, srcDyPyr, CV_8S);
    }

    while(next())
    {
        std::vector<int32_t> statusVec(nPts);
        std::vector<cv::Point2f> ptsOut(nPts);
        startTimer();
        if (useSobelPyramid)
        {
            cv::fastcv::trackOpticalFlowLK(src, dst, srcPyr, dstPyr, srcDxPyr, srcDyPyr,
                                           ptsIn, ptsOut, statusVec, {winSz, winSz});
        }
        else
        {
            cv::fastcv::trackOpticalFlowLK(src, dst, srcPyr, dstPyr, ptsIn, ptsOut, (useInitialEstimate ? ptsEst : noArray()),
                                           statusVec, {winSz, winSz}, termCrit);
        }
        stopTimer();
    }

    SANITY_CHECK_NOTHING();
}

} // namespace
