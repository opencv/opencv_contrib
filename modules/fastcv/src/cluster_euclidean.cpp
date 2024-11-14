/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "precomp.hpp"

namespace cv {
namespace fastcv {

void clusterEuclidean(InputArray _points, InputArray _clusterCenters, OutputArray _newClusterCenters,
                      OutputArray _clusterSizes, OutputArray _clusterBindings, OutputArray _clusterSumDists,
                      int numPointsUsed)
{
    INITIALIZATION_CHECK;

    CV_Assert(!_points.empty() && _points.type() == CV_8UC1);
    int nPts      = _points.rows();
    int nDims     = _points.cols();
    int ptsStride = _points.step();

    CV_Assert(!_clusterCenters.empty() && _clusterCenters.depth() == CV_32F);
    int nClusters           = _clusterCenters.rows();
    int clusterCenterStride = _clusterCenters.step();

    CV_Assert(_clusterCenters.cols() == nDims);

    CV_Assert(numPointsUsed <= nPts);
    if (numPointsUsed < 0)
    {
        numPointsUsed = nPts;
    }

    _newClusterCenters.create(nClusters, nDims, CV_32FC1);
    _clusterSizes.create(1, nClusters, CV_32SC1);
    _clusterBindings.create(1, numPointsUsed, CV_32SC1);
    _clusterSumDists.create(1, nClusters, CV_32FC1);

    Mat points            = _points.getMat();
    Mat clusterCenters    = _clusterCenters.getMat();
    Mat newClusterCenters = _newClusterCenters.getMat();
    Mat clusterSizes      = _clusterSizes.getMat();
    Mat clusterBindings   = _clusterBindings.getMat();
    Mat clusterSumDists   = _clusterSumDists.getMat();

    int result = fcvClusterEuclideanu8(points.data,
                                       nPts,
                                       nDims,
                                       ptsStride,
                                       numPointsUsed,
                                       nClusters,
                                       (float32_t*)clusterCenters.data,
                                       clusterCenterStride,
                                       (float32_t*)newClusterCenters.data,
                                       (uint32_t*)clusterSizes.data,
                                       (uint32_t*)clusterBindings.data,
                                       (float32_t*)clusterSumDists.data);

    if (result)
    {
        CV_Error(cv::Error::StsInternal, cv::format("Failed to clusterize, error code: %d", result));
    }
}

} // fastcv::
} // cv::
