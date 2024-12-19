/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#ifndef OPENCV_FASTCV_CLUSTER_HPP
#define OPENCV_FASTCV_CLUSTER_HPP

#include <opencv2/core.hpp>

namespace cv {
namespace fastcv {

//! @addtogroup fastcv
//! @{

/**
 * @brief Clusterizes N input points in D-dimensional space into K clusters
 *        Accepts 8-bit unsigned integer points
 *        Provides faster execution time than cv::kmeans on Qualcomm's processors
 * @param points            Points array of type 8u, each row represets a point.
 *                          Size is N rows by D columns, can be non-continuous.
 * @param clusterCenters    Initial cluster centers array of type 32f, each row represents a center.
 *                          Size is K rows by D columns, can be non-continuous.
 * @param newClusterCenters Resulting cluster centers array of type 32f, each row represents found center.
 *                          Size is set to be K rows by D columns.
 * @param clusterSizes      Resulting cluster member counts array of type uint32, size is set to be 1 row by K columns.
 * @param clusterBindings   Resulting points indices array of type uint32, each index tells to which cluster the corresponding point belongs to.
 *                          Size is set to be 1 row by numPointsUsed columns.
 * @param clusterSumDists   Resulting distance sums array of type 32f, each number is a sum of distances between each cluster center to its belonging points.
 *                          Size is set to be 1 row by K columns
 * @param numPointsUsed     Number of points to clusterize starting from 0 to numPointsUsed-1 inclusively. Sets to N if negative.
 */
CV_EXPORTS_W void clusterEuclidean(InputArray points, InputArray clusterCenters, OutputArray newClusterCenters,
                                   OutputArray clusterSizes, OutputArray clusterBindings, OutputArray clusterSumDists,
                                   int numPointsUsed = -1);

//! @}

} // fastcv::
} // cv::

#endif // OPENCV_FASTCV_CLUSTER_HPP
