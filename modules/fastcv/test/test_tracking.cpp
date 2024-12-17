/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "test_precomp.hpp"

namespace opencv_test { namespace {

typedef std::tuple<int /*winSize*/, bool /*useSobelPyramid*/, bool /*useFastCvPyramids*/, bool /*useInitialEstimate*/ > TrackingTestParams;
class TrackingTest : public ::testing::TestWithParam<TrackingTestParams> {};

TEST_P(TrackingTest, accuracy)
{
    auto par = GetParam();

    int winSz               = std::get<0>(par);
    bool useSobelPyramid    = std::get<1>(par);
    bool useFastCvPyramids  = std::get<2>(par);
    bool useInitialEstimate = std::get<3>(par);

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

    if (useFastCvPyramids)
    {
        cv::fastcv::buildPyramid(src, srcPyr, nLevels);
        cv::fastcv::buildPyramid(dst, dstPyr, nLevels);
    }
    else
    {
        cv::buildPyramid(src, srcPyr, nLevels - 1);
        cv::buildPyramid(dst, dstPyr, nLevels - 1);
    }

    cv::Matx23f transf = trans;
    int nPts = 32;
    std::vector<cv::Point2f> ptsIn, ptsOut, ptsEst, ptsExpected;
    for (int i = 0; i < nPts; i++)
    {
        cv::Point2f p { (((float)cv::theRNG())*0.5f + 0.25f) * src.cols,
                        (((float)cv::theRNG())*0.5f + 0.25f) * src.rows };
        ptsIn.push_back(p);
        ptsExpected.push_back(transf * cv::Vec3f(p.x, p.y, 1.0));
        ptsOut.push_back({ });
        ptsEst.push_back(p);
    }

    std::vector<int32_t> statusVec(nPts);

    cv::TermCriteria termCrit;
    termCrit.type = cv::TermCriteria::COUNT | cv::TermCriteria::EPS;
    termCrit.maxCount = 7;
    termCrit.epsilon = 0.03f * 0.03f;

    if (useSobelPyramid)
    {
        std::vector<cv::Mat> srcDxPyr, srcDyPyr;
        cv::fastcv::sobelPyramid(srcPyr, srcDxPyr, srcDyPyr, CV_8S);
        cv::fastcv::trackOpticalFlowLK(src, dst, srcPyr, dstPyr, srcDxPyr, srcDyPyr,
                                       ptsIn, ptsOut, statusVec, {winSz, winSz});
    }
    else
    {
        cv::fastcv::trackOpticalFlowLK(src, dst, srcPyr, dstPyr, ptsIn, ptsOut, (useInitialEstimate ? ptsEst : noArray()),
                                        statusVec, {winSz, winSz}, termCrit);
    }

    std::vector<cv::Point2f> ocvPtsOut;
    std::vector<uint8_t> ocvStatusVec;
    std::vector<float> ocvErrVec;
    cv::calcOpticalFlowPyrLK(src, dst, ptsIn, ocvPtsOut, ocvStatusVec, ocvErrVec, {winSz, winSz}, nLevels - 1, termCrit);

    cv::Mat refStatusVec(nPts, 1, CV_32S, Scalar::all(1));
    cv::Mat ocvStatusVecInt;
    cv::Mat(ocvStatusVec).convertTo(ocvStatusVecInt, CV_32S);

    double statusNormOcv = cv::norm(ocvStatusVecInt, refStatusVec, NORM_INF);
    double statusNorm = cv::norm(cv::Mat(statusVec), refStatusVec, NORM_INF);

    EXPECT_EQ(statusNormOcv, 0);
    EXPECT_EQ(statusNorm, 0);

    double diffNormOcv = cv::norm(ocvPtsOut, ptsExpected, NORM_L2);
    double diffNorm = cv::norm(ptsOut, ptsExpected, NORM_L2);

    EXPECT_LT(diffNormOcv, 31.92);
    EXPECT_LT(diffNorm, 6.69);

    if (cvtest::debugLevel > 0)
    {
        auto drawPts = [ptsIn, dst](const std::vector<cv::Point2f>& ptsRes, const std::string fname)
        {
            cv::Mat draw = dst.clone();
            for (size_t i = 0; i < ptsIn.size(); i++)
            {
                cv::line(draw, ptsIn[i], ptsRes[i], Scalar::all(255));
                cv::circle(draw, ptsIn[i], 1, Scalar::all(255));
                cv::circle(draw, ptsRes[i], 3, Scalar::all(255));
            }
            cv::imwrite(fname, draw);
        };

        drawPts(ptsOut, "track_w"+std::to_string(winSz)+"_warped.png");
        drawPts(ocvPtsOut, "track_ocv_warped.png");

        std::cout << "status vec:"   << std::endl << cv::Mat(statusVec).t()   << std::endl;
        std::cout << "status vec ocv:" << std::endl << cv::Mat(ocvStatusVec).t() << std::endl;
    }
}

INSTANTIATE_TEST_CASE_P(FastCV_Extension, TrackingTest,
                        ::testing::Combine(::testing::Values(5, 7, 9), // window size
                                           ::testing::Bool(),          // useSobelPyramid
                                           ::testing::Bool(),          // useFastCvPyramids
                                           ::testing::Bool()           // useInitialEstimate
                        ));

}} // namespaces opencv_test, ::
