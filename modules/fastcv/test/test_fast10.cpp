/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "test_precomp.hpp"

namespace opencv_test { namespace {

typedef std::tuple<bool /*useScores*/, int /*barrier*/, int /*border*/, bool /*nmsEnabled*/> Fast10TestParams;
class Fast10Test : public ::testing::TestWithParam<Fast10TestParams> {};

TEST_P(Fast10Test, accuracy)
{
    auto p = GetParam();
    bool useScores  = std::get<0>(p);
    int barrier     = std::get<1>(p);
    int border      = std::get<2>(p);
    bool nmsEnabled = std::get<3>(p);

    cv::Mat src = imread(cvtest::findDataFile("cv/shared/baboon.png"), cv::IMREAD_GRAYSCALE);

    std::vector<int> coords, scores;
    cv::fastcv::FAST10(src, noArray(), coords, useScores ? scores : noArray(), barrier, border, nmsEnabled);

    std::vector<KeyPoint> ocvKeypoints;
    int thresh = barrier;
    cv::FAST(src, ocvKeypoints, thresh, nmsEnabled, FastFeatureDetector::DetectorType::TYPE_9_16 );

    if (useScores)
    {
        ASSERT_EQ(scores.size() * 2, coords.size());
    }

    Mat ptsMap(src.size(), CV_8U, Scalar(255));
    for(size_t i = 0; i < coords.size() / 2; ++i)
    {
        ptsMap.at<uchar>(coords[2*i + 1], coords[2*i + 0]) = 0;
    }
    Mat distTrans(src.size(), CV_8U);
    cv::distanceTransform(ptsMap, distTrans, DIST_L2, DIST_MASK_PRECISE);

    Mat refPtsMap(src.size(), CV_8U, Scalar(255));
    for(size_t i = 0; i < ocvKeypoints.size(); ++i)
    {
        refPtsMap.at<uchar>(ocvKeypoints[i].pt) = 0;
    }
    Mat refDistTrans(src.size(), CV_8U);
    cv::distanceTransform(refPtsMap, refDistTrans, DIST_L2, DIST_MASK_PRECISE);

    double normInf = cvtest::norm(refDistTrans, distTrans, cv::NORM_INF);
    double normL2  = cvtest::norm(refDistTrans, distTrans, cv::NORM_L2)  / src.size().area();

    EXPECT_LT(normInf, 129.7);
    EXPECT_LT(normL2, 0.067);
}

INSTANTIATE_TEST_CASE_P(FastCV_Extension, Fast10Test,
                        ::testing::Combine(::testing::Bool(),   // useScores
                                           ::testing::Values(10, 30, 50), // barrier
                                           ::testing::Values( 4, 10, 32), // border
                                           ::testing::Bool() // nonmax suppression
                                           ));

}} // namespaces opencv_test, ::
