// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.

#ifndef __ZXING_COMMON_KMEANS_HPP__
#define __ZXING_COMMON_KMEANS_HPP__
#include <vector>

namespace zxing {

using namespace std;
typedef unsigned int uint;

struct Cluster {
    vector<double> centroid;
    vector<uint> samples;
};

double cal_distance(vector<double> a, vector<double> b);
vector<Cluster> k_means(vector<vector<double> > trainX, uint k, uint maxepoches, uint minchanged);

}  // namespace zxing
#endif  // __ZXING_COMMON_KMEANS_HPP__
