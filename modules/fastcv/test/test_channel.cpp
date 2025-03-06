/*
 * Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "test_precomp.hpp"

namespace opencv_test { namespace {

typedef std::tuple<Size, int> ChannelMergeTestParams;
class ChannelMergeTest : public ::testing::TestWithParam<ChannelMergeTestParams> {};

typedef std::tuple<Size, int> ChannelSplitTestParams;
class ChannelSplitTest : public ::testing::TestWithParam<ChannelSplitTestParams> {};

TEST_P(ChannelMergeTest, accuracy)
{
    int depth = CV_8UC1;
    Size sz = std::get<0>(GetParam());
    int count = std::get<1>(GetParam());
    std::vector<Mat> src_mats;

    RNG& rng = cv::theRNG();

    for(int i = 0; i < count; i++)
    {
        Mat tmp(sz, depth);
        src_mats.push_back(tmp);
        cvtest::randUni(rng, src_mats[i], Scalar::all(0), Scalar::all(127));
    }

    Mat dst;
    cv::fastcv::merge(src_mats, dst);

    Mat ref;
    cv::merge(src_mats, ref);

    double normInf = cvtest::norm(ref, dst, cv::NORM_INF);

    EXPECT_EQ(normInf, 0);
}

TEST_P(ChannelSplitTest, accuracy)
{
    Size sz = std::get<0>(GetParam());
    int cn = std::get<1>(GetParam());
    std::vector<Mat> dst_mats(cn), ref_mats(cn);

    RNG& rng = cv::theRNG();
    Mat src(sz, CV_MAKE_TYPE(CV_8U,cn));
    cvtest::randUni(rng, src, Scalar::all(0), Scalar::all(127));

    cv::fastcv::split(src, dst_mats);

    cv::split(src, ref_mats);

    for(int i=0; i<cn; i++)
    {
        double normInf = cvtest::norm(ref_mats[i], dst_mats[i], cv::NORM_INF);
        EXPECT_EQ(normInf, 0);
    }
}

INSTANTIATE_TEST_CASE_P(FastCV_Extension, ChannelMergeTest,
                         ::testing::Combine(::testing::Values(perf::szODD, perf::szVGA, perf::sz720p, perf::sz1080p),   // sz
                                            ::testing::Values(2,3,4)));  // count

INSTANTIATE_TEST_CASE_P(FastCV_Extension, ChannelSplitTest,
                         ::testing::Combine(::testing::Values(perf::szODD, perf::szVGA, perf::sz720p, perf::sz1080p),   // sz
                                            ::testing::Values(2,3,4)));    // cn

}} // namespaces opencv_test, ::
