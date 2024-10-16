/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "test_precomp.hpp"

namespace opencv_test { namespace {

// nPts, nDims, nClusters
typedef std::tuple<int, int, int> ClusterEuclideanTestParams;
class ClusterEuclideanTest : public ::testing::TestWithParam<ClusterEuclideanTestParams> {};

TEST_P(ClusterEuclideanTest, accuracy)
{
    auto p = GetParam();
    int nPts      = std::get<0>(p);
    int nDims     = std::get<1>(p);
    int nClusters = std::get<2>(p);

    Mat points(nPts, nDims, CV_8U);
    Mat clusterCenters(nClusters, nDims, CV_32F);

    Mat trueMeans(nClusters, nDims, CV_32F);
    Mat stddevs(nClusters, nDims, CV_32F);
    std::vector<int> trueClusterSizes(nClusters, 0);
    std::vector<int> trueClusterBindings(nPts, 0);
    std::vector<float> trueSumDists(nClusters, 0);

    cv::RNG& rng = cv::theRNG();
    for (int i = 0; i < nClusters; i++)
    {
        Mat mean(1, nDims, CV_64F), stdev(1, nDims, CV_64F);
        rng.fill(mean,  cv::RNG::UNIFORM, 0, 256);
        rng.fill(stdev, cv::RNG::UNIFORM, 5.f, 16);
        int lo =    i    * nPts / nClusters;
        int hi = (i + 1) * nPts / nClusters;

        for (int d = 0; d < nDims; d++)
        {
            rng.fill(points.col(d).rowRange(lo, hi), cv::RNG::NORMAL,
                     mean.at<double>(d), stdev.at<double>(d));
        }

        float sd = 0;
        for (int j = lo; j < hi; j++)
        {
            Mat pts64f;
            points.row(j).convertTo(pts64f, CV_64F);
            sd += cv::norm(mean, pts64f, NORM_L2);
            trueClusterBindings.at(j) = i;
            trueClusterSizes.at(i)++;
        }
        trueSumDists.at(i) = sd;

        // let's shift initial cluster center a bit
        Mat(mean + stdev * 0.5).copyTo(clusterCenters.row(i));

        mean.copyTo(trueMeans.row(i));
        stdev.copyTo(stddevs.row(i));
    }

    Mat newClusterCenters;
    std::vector<int> clusterSizes, clusterBindings;
    std::vector<float> clusterSumDists;
    cv::fastcv::clusterEuclidean(points, clusterCenters, newClusterCenters, clusterSizes, clusterBindings, clusterSumDists);

    if (cvtest::debugLevel > 0 && nDims == 2)
    {
        Mat draw(256, 256, CV_8UC3, Scalar(0));
        for (int i = 0; i < nPts; i++)
        {
            int x = std::rint(points.at<uchar>(i, 0));
            int y = std::rint(points.at<uchar>(i, 1));
            draw.at<Vec3b>(y, x) = Vec3b::all(128);
        }
        for (int i = 0; i < nClusters; i++)
        {
            float cx = trueMeans.at<double>(i, 0);
            float cy = trueMeans.at<double>(i, 1);
            draw.at<Vec3b>(cy, cx) = Vec3b(0, 255, 0);

            float sx = stddevs.at<double>(i, 0);
            float sy = stddevs.at<double>(i, 1);
            cv::ellipse(draw, Point(cx, cy), Size(sx, sy), 0, 0, 360, Scalar(255, 0, 0));

            float ox = clusterCenters.at<float>(i, 0);
            float oy = clusterCenters.at<float>(i, 1);
            draw.at<Vec3b>(oy, ox) = Vec3b(0, 0, 255);

            float nx = newClusterCenters.at<float>(i, 0);
            float ny = newClusterCenters.at<float>(i, 1);
            draw.at<Vec3b>(ny, nx) = Vec3b(255, 255, 0);
        }
        cv::imwrite(cv::format("draw_%d_%d_%d.png", nPts, nDims, nClusters), draw);
    }

    {
        std::vector<double> diffs;
        for (int i = 0; i < nClusters; i++)
        {
            double cs = std::abs((trueClusterSizes[i] - clusterSizes[i]) / double(trueClusterSizes[i]));
            diffs.push_back(cs);
        }
        double normL2  = cv::norm(diffs, NORM_L2) / nClusters;

        EXPECT_LT(normL2, 0.392);
    }

    {
        Mat bindings8u, trueBindings8u;
        Mat(clusterBindings).convertTo(bindings8u, CV_8U);
        Mat(trueClusterBindings).convertTo(trueBindings8u, CV_8U);
        double normH = cv::norm(bindings8u, trueBindings8u, NORM_HAMMING) / nPts;
        EXPECT_LT(normH, 0.658);
    }
}

INSTANTIATE_TEST_CASE_P(FastCV_Extension, ClusterEuclideanTest,
                        ::testing::Combine(::testing::Values(100, 1000, 10000), // nPts
                                           ::testing::Values(2, 10, 32),        // nDims
                                           ::testing::Values(5, 10, 16)));      // nClusters

}} // namespaces opencv_test, ::