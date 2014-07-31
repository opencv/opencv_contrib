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

using std::swap;

using std::vector;

#include <iostream>
using std::cout;
using std::endl;


namespace cv
{
namespace xobjdetect
{

class WaldBoostImpl : public WaldBoost
{
public:
    /* Initialize WaldBoost cascade with default of specified parameters */
    WaldBoostImpl(const WaldBoostParams& params):
        params_(params)
    {}

    virtual std::vector<int> train(const Mat& data,
                                   const Mat& labels);

    virtual float predict(
        const Ptr<FeatureEvaluator>& feature_evaluator) const;

    virtual void write(FileStorage& fs) const;

    virtual void read(const FileNode& node);

private:
    /* Parameters for cascade training */
    WaldBoostParams params_;
    /* Stumps in cascade */
    std::vector<Stump> stumps_;
    /* Rejection thresholds for linear combination at every stump evaluation */
    std::vector<float> thresholds_;
};

static int count(const Mat_<int> &m, int elem)
{
    int res = 0;
    for( int row = 0; row < m.rows; ++row )
        for( int col = 0; col < m.cols; ++col )
            if( m(row, col) == elem)
                res += 1;
    return res;
}

void write(FileStorage& fs, String&, const WaldBoost& waldboost)
{
    waldboost.write(fs);
}

void read(const FileNode& node, WaldBoost& w,
    const WaldBoost& default_value)
{
    if( node.empty() )
        w = default_value;
    else
        w.read(node);
}

void WaldBoostImpl::read(const FileNode& node)
{
    FileNode params = node["waldboost_params"];
    params_.weak_count = (int)(params["weak_count"]);
    params_.alpha = (float)(params["alpha"]);

    FileNode stumps = node["waldboost_stumps"];
    stumps_.clear();
    for( FileNodeIterator n = stumps.begin(); n != stumps.end(); ++n )
    {
        Stump s;
        *n >> s;
        stumps_.push_back(s);
    }

    FileNode thresholds = node["waldboost_thresholds"];
    thresholds_.clear();
    for( FileNodeIterator n = thresholds.begin(); n != thresholds.end(); ++n )
    {
        float t;
        *n >> t;
        thresholds_.push_back(t);
    }
}

void WaldBoostImpl::write(FileStorage& fs) const
{
    fs << "{";
    fs << "waldboost_params" << "{"
        << "weak_count" << params_.weak_count
        << "alpha" << params_.alpha
        << "}";

    fs << "waldboost_stumps" << "[";
    for( size_t i = 0; i < stumps_.size(); ++i )
        fs << stumps_[i];
    fs << "]";

    fs << "waldboost_thresholds" << "[";
    for( size_t i = 0; i < thresholds_.size(); ++i )
        fs << thresholds_[i];
    fs << "]";
    fs << "}";

}

vector<int> WaldBoostImpl::train(const Mat& data_, const Mat& labels_)
{
    CV_Assert(labels_.rows == 1 && labels_.cols == data_.cols);
    CV_Assert(data_.rows >= params_.weak_count);

    Mat labels, data;
    data_.copyTo(data);
    labels_.copyTo(labels);

    bool null_data = true;
    for( int row = 0; row < data.rows; ++row )
    {
        for( int col = 0; col < data.cols; ++col )
            if( data.at<int>(row, col) )
                null_data = false;
    }
    CV_Assert(!null_data);

    int pos_count = count(labels, +1);
    int neg_count = count(labels, -1);

    Mat_<float> weights(labels.rows, labels.cols);
    float pos_weight = 1.0f / (2 * pos_count);
    float neg_weight = 1.0f / (2 * neg_count);
    for( int col = 0; col < weights.cols; ++col )
    {
        if( labels.at<int>(0, col) == +1 )
            weights.at<float>(0, col) = pos_weight;
        else
            weights.at<float>(0, col) = neg_weight;
    }

    vector<int> feature_indices_pool;
    for( int ind = 0; ind < data.rows; ++ind )
        feature_indices_pool.push_back(ind);

    vector<int> feature_indices;
    Mat_<float> trace = Mat_<float>::zeros(labels.rows, labels.cols);
    stumps_.clear();
    thresholds_.clear();
    for( int i = 0; i < params_.weak_count; ++i)
    {
        cout << "stage " << i << endl;
        Stump s;
        int feature_ind = s.train(data, labels, weights);
        cout << "feature_ind " << feature_ind << endl;
        stumps_.push_back(s);
        int ind = feature_indices_pool[feature_ind];
        feature_indices_pool.erase(feature_indices_pool.begin() + feature_ind);
        feature_indices.push_back(ind);

        // Recompute weights
        for( int col = 0; col < weights.cols; ++col )
        {
            float h = s.predict(data.at<int>(feature_ind, col));
            trace(0, col) += h;
            int label = labels.at<int>(0, col);
            weights.at<float>(0, col) *= exp(-label * h);
        }

        // Erase row for feature in data
        Mat fixed_data;
        fixed_data.push_back(data.rowRange(0, feature_ind));
        fixed_data.push_back(data.rowRange(feature_ind + 1, data.rows));

        data = fixed_data;


        // Normalize weights
        float z = (float)sum(weights)[0];
        for( int col = 0; col < weights.cols; ++col)
        {
            weights.at<float>(0, col) /= z;
        }

        // Sort trace
        Mat indices;
        sortIdx(trace, indices, cv::SORT_EVERY_ROW | cv::SORT_ASCENDING);
        Mat new_weights = Mat_<float>::zeros(weights.rows, weights.cols);
        Mat new_labels = Mat_<int>::zeros(labels.rows, labels.cols);
        Mat new_data = Mat_<int>::zeros(data.rows, data.cols);
        Mat new_trace;
        for( int col = 0; col < new_weights.cols; ++col )
        {
            new_weights.at<float>(0, col) =
                weights.at<float>(0, indices.at<int>(0, col));
            new_labels.at<int>(0, col) =
                labels.at<int>(0, indices.at<int>(0, col));
            for( int row = 0; row < new_data.rows; ++row )
            {
                new_data.at<int>(row, col) =
                    data.at<int>(row, indices.at<int>(0, col));
            }
        }
        sort(trace, new_trace, cv::SORT_EVERY_ROW | cv::SORT_ASCENDING);


        // Compute threshold for trace
        /*
        int col = 0;
        for( int pos_i = 0;
             pos_i < pos_count * params_.alpha && col < new_labels.cols;
             ++col )
        {
            if( new_labels.at<int>(0, col) == +1 )
                ++pos_i;
        }
        */

        int cur_pos = 0, cur_neg = 0;
        int max_col = 0;
        for( int col = 0; col < new_labels.cols - 1; ++col ) {
            if (new_labels.at<int>(0, col) == +1 )
                ++cur_pos;
            else
                ++cur_neg;

            float p_neg = cur_neg / (float)neg_count;
            float p_pos = cur_pos / (float)pos_count;
            if( params_.alpha * p_neg > p_pos )
                max_col = col;
        }

        thresholds_.push_back(new_trace.at<float>(0, max_col));
        cout << "threshold " << *(thresholds_.end() - 1) << endl;

        cout << "col " << max_col << " size " << data.cols << endl;

        // Drop samples below threshold
        new_data.colRange(max_col, new_data.cols).copyTo(data);
        new_trace.colRange(max_col, new_trace.cols).copyTo(trace);
        new_weights.colRange(max_col, new_weights.cols).copyTo(weights);
        new_labels.colRange(max_col, new_labels.cols).copyTo(labels);

        pos_count = count(labels, +1);
        neg_count = count(labels, -1);
        cout << "pos_count " << pos_count << "; neg_count " << neg_count << endl;

        if( data.cols < 2 || neg_count == 0)
        {
            break;
        }
    }
    return feature_indices;
}

float WaldBoostImpl::predict(
    const Ptr<FeatureEvaluator>& feature_evaluator) const
{
    float trace = 0;
    CV_Assert(stumps_.size() == thresholds_.size());
    for( size_t i = 0; i < stumps_.size(); ++i )
    {
        int value = feature_evaluator->evaluate(i);
        trace += stumps_[i].predict(value);
        if( trace < thresholds_[i] )
            return -1;
    }
    return trace;
}

Ptr<WaldBoost>
createWaldBoost(const WaldBoostParams& params)
{
    return Ptr<WaldBoost>(new WaldBoostImpl(params));
}


} /* namespace xobjdetect */
} /* namespace cv */
