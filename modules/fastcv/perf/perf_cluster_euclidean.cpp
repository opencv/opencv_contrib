/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "perf_precomp.hpp"

namespace opencv_test {

typedef std::tuple<int /* nPts */, int /*nDims*/, int /*nClusters*/> ClusterEuclideanPerfParams;
typedef perf::TestBaseWithParam<ClusterEuclideanPerfParams> ClusterEuclideanPerfTest;

PERF_TEST_P(ClusterEuclideanPerfTest, run,
            ::testing::Combine(::testing::Values(100, 1000, 10000), // nPts
                               ::testing::Values(2, 10, 32),        // nDims
                               ::testing::Values(5, 10, 16))        // nClusters
           )
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

    while(next())
    {
        Mat newClusterCenters;
        std::vector<int> clusterSizes, clusterBindings;
        std::vector<float> clusterSumDists;
        startTimer();
        cv::fastcv::clusterEuclidean(points, clusterCenters, newClusterCenters, clusterSizes, clusterBindings, clusterSumDists);
        stopTimer();
    }

    SANITY_CHECK_NOTHING();
}

} // namespace
