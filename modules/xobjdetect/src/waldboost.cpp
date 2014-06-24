/*

By downloading, copying, installing or using the software you agree to this
license. If you do not agree to this license, do not download, install,
copy or use the software.


                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Copyright (C) 2013, OpenCV Foundation, all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are
disclaimed. In no event shall copyright holders or contributors be liable for
any direct, indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.

*/

#include "precomp.hpp"

#include <cmath>

#include <algorithm>
using std::swap;

#include <opencv2/xobjdetect.hpp>

using std::vector;

namespace cv
{
namespace adas
{

WaldBoost::WaldBoost(const WaldBoostParams& params): params_(params)
{
}

vector<int> WaldBoost::train(const Mat& data, const Mat& labels)
{
    CV_Assert(labels.rows == 1 && labels.cols == data.cols);

    int pos_count = 0, neg_count = 0;
    for( int col = 0; col < labels.cols; ++col )
    {
        if( labels.at<int>(0, col) == +1 )
            pos_count += 1;
        else
            neg_count += 1;
    }

    Mat_<float> weights(labels.rows, labels.cols);
    float pos_weight = 1. / (2 * pos_count);
    float neg_weight = 1. / (2 * neg_count);
    for( int col = 0; col < weights.cols; ++col )
    {
        if( labels.at<int>(0, col) == +1 )
            weights.at<float>(0, col) = pos_weight;
        else
            weights.at<float>(0, col) = neg_weight;
    }


    vector<int> feature_indices;
    Mat_<float> trace = Mat_<float>::zeros(labels.rows, labels.cols);
    stumps_.clear();
    thresholds_.clear();
    for( int i = 0; i < params_.weak_count; ++i)
    {
        Stump s;
        int feature_ind = s.train(data, labels, weights);
        stumps_.push_back(s);
        feature_indices.push_back(feature_ind);

        // Recompute weights
        for( int col = 0; col < weights.cols; ++col )
        {
            float h = s.predict(data.at<int>(feature_ind, col));
            trace(0, col) += h;
            int label = labels.at<int>(0, col);
            weights.at<float>(0, col) *= exp(-label * h);
        }

        // Normalize weights
        float z = sum(weights)[0];
        for( int col = 0; col < weights.cols; ++col)
        {
            weights.at<float>(0, col) /= z;
        }


        // Sort trace
        Mat indices;
        sortIdx(trace, indices, cv::SORT_EVERY_ROW | cv::SORT_ASCENDING);
        Mat new_weights = Mat_<float>::zeros(weights.rows, weights.cols);
        Mat new_labels = Mat_<int>::zeros(labels.rows, labels.cols);
        Mat new_trace;
        for( int col = 0; col < new_weights.cols; ++col )
        {
            new_weights.at<float>(0, col) =
                weights.at<float>(0, indices.at<int>(0, col));
            new_labels.at<int>(0, col) =
                labels.at<int>(0, indices.at<int>(0, col));
        }
        sort(trace, new_trace, cv::SORT_EVERY_ROW | cv::SORT_ASCENDING);


        // Compute threshold for trace
        int col = 0;
        for( int pos_i = 0;
             pos_i < pos_count * params_.alpha && col < weights.cols;
             ++col )
        {
            if( labels.at<int>(0, col) == +1 )
                ++pos_i;
        }

        thresholds_.push_back(new_trace.at<float>(0, col));

        // Drop samples below threshold
        new_trace.colRange(col, new_trace.cols).copyTo(trace);
        new_weights.colRange(col, new_weights.cols).copyTo(weights);
        new_labels.colRange(col, new_labels.cols).copyTo(labels);
    }
    return feature_indices;
}

float WaldBoost::predict(const Ptr<ACFFeatureEvaluator>& feature_evaluator)
{
    float trace = 0;
    for( size_t i = 0; i < stumps_.size(); ++i )
    {
        int value = feature_evaluator->evaluate(i);
        trace += stumps_[i].predict(value);
        if( trace < thresholds_[i] )
            return -1;
    }
    return trace;
}

} /* namespace adas */
} /* namespace cv */
