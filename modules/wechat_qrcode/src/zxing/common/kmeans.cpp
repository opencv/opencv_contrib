// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
#include "../../precomp.hpp"
#include "kmeans.hpp"

typedef unsigned int uint;

namespace zxing {

double cal_distance(vector<double> a, vector<double> b) {
    const float KMEANS_COUNT_FACTOR = 0;
    const float KMEANS_MS_FACTOR = 1;

    uint da = a.size();
    double val = 0.0;
    for (uint i = 0; i < da; i++) {
        if (i == 1)
            val += KMEANS_MS_FACTOR * pow((a[i] - b[i]), 2);
        else if (i == 0)
            val += KMEANS_COUNT_FACTOR * pow((a[i] - b[i]), 2);
        else
            val += pow((a[i] - b[i]), 2);
    }
    return pow(val, 0.5);
}

/*
 * maxepoches max iteration epochs
 * minchanged min central change times
 */
vector<Cluster> k_means(vector<vector<double> > trainX, uint k, uint maxepoches, uint minchanged) {
    const uint row_num = trainX.size();
    const uint col_num = trainX[0].size();

    // initialize the cluster central
    vector<Cluster> clusters(k);
    int step = trainX.size() / k;

    for (uint i = 0; i < k; i++) {
        clusters[i].centroid = trainX[i * step];
    }

    // try max epochs times iteration untill convergence
    for (uint it = 0; it < maxepoches; it++) {
        for (uint i = 0; i < k; i++) {
            clusters[i].samples.clear();
        }
        for (uint j = 0; j < row_num; j++) {
            uint c = 0;
            double min_distance = cal_distance(trainX[j], clusters[c].centroid);
            for (uint i = 1; i < k; i++) {
                double distance = cal_distance(trainX[j], clusters[i].centroid);
                if (distance < min_distance) {
                    min_distance = distance;
                    c = i;
                }
            }
            clusters[c].samples.push_back(j);
        }

        uint changed = 0;
        // update cluster central
        for (uint i = 0; i < k; i++) {
            vector<double> val(col_num, 0.0);
            for (uint j = 0; j < clusters[i].samples.size(); j++) {
                uint sample = clusters[i].samples[j];
                for (uint d = 0; d < col_num; d++) {
                    val[d] += trainX[sample][d];
                    if (j == clusters[i].samples.size() - 1) {
                        double value = val[d] / clusters[i].samples.size();
                        if (clusters[i].centroid[d] != value) {
                            clusters[i].centroid[d] = value;
                            changed++;
                        }
                    }
                }
            }
        }

        if (changed <= minchanged) return clusters;
    }
    return clusters;
}

}  // namespace zxing
