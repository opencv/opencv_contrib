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

namespace cv
{
namespace xobjdetect
{

/* Cumulative sum by rows */
static void cumsum(const Mat_<float>& src, Mat_<float> dst)
{
    CV_Assert(src.cols > 0);

    for( int row = 0; row < src.rows; ++row )
    {
        dst(row, 0) = src(row, 0);
        for( int col = 1; col < src.cols; ++col )
        {
            dst(row, col) = dst(row, col - 1) + src(row, col);
        }
    }
}

//fast log implementation. A bit less accurate but ~5x faster
inline float fast_log2 (float val)
{
   int * const    exp_ptr = reinterpret_cast <int *> (&val);
   int            x = *exp_ptr;
   const int      log_2 = ((x >> 23) & 255) - 128;
   x &= ~(255 << 23);
   x += 127 << 23;
   *exp_ptr = x;

   val = ((-1.0f/3) * val + 2) * val - 2.0f/3;   // (1)

   return (val + log_2);
} 

inline float fast_log (const float &val)
{
   return (fast_log2 (val) * 0.69314718f);
}

int Stump::train(const Mat& data, const Mat& labels, const Mat& weights, const std::vector<int>& visited_features, bool use_fast_log)
{
    CV_Assert(labels.rows == 1 && labels.cols == data.cols);
    CV_Assert(weights.rows == 1 && weights.cols == data.cols);
    CV_Assert(data.cols > 1);
    /* Assert that data and labels have int type */
    /* Assert that weights have float type */

    Mat_<int> d = Mat_<int>::zeros(1, data.cols);
    const Mat_<int>& l = labels;
    const Mat_<float>& w = weights;

    Mat_<int> indices(1, l.cols);

    Mat_<int> sorted_d(1, data.cols);
    Mat_<int> sorted_l(1, l.cols);
    Mat_<float> sorted_w(1, w.cols);


    Mat_<float> pos_c_w = Mat_<float>::zeros(1, w.cols);
    Mat_<float> neg_c_w = Mat_<float>::zeros(1, w.cols);


    float min_err = FLT_MAX;
    int min_row = -1;
    int min_thr = -1;
    int min_pol = -1;
    float min_pos = 1;
    float min_neg = -1;
    float eps = 1.0f / (4 * l.cols);

    /* For every feature */
    for( int row = 0; row < data.rows; ++row )
    {
        if(std::find(visited_features.begin(), visited_features.end(), row) != visited_features.end()) {
              //feature discarded
              continue;
        }
        data.row(row).copyTo(d.row(0));

        sortIdx(d, indices, cv::SORT_EVERY_ROW | cv::SORT_ASCENDING);

        for( int col = 0; col < indices.cols; ++col )
        {
            int ind = indices(0, col);
            sorted_d(0, col) = d(0, ind);
            sorted_l(0, col) = l(0, ind);
            sorted_w(0, col) = w(0, ind);
        }

        Mat_<float> pos_w = Mat_<float>::zeros(1, w.cols);
        Mat_<float> neg_w = Mat_<float>::zeros(1, w.cols);
        for( int col = 0; col < d.cols; ++col )
        {
            float weight = sorted_w(0, col);
            if( sorted_l(0, col) == +1)
                pos_w(0, col) = weight;
            else
                neg_w(0, col) = weight;
        }

        cumsum(pos_w, pos_c_w);
        cumsum(neg_w, neg_c_w);

        float pos_total_w = pos_c_w(0, w.cols - 1);
        float neg_total_w = neg_c_w(0, w.cols - 1);

        for( int col = 0; col < w.cols - 1; ++col )
        {
            float err, h_pos, h_neg;
            float pos_wrong, pos_right;
            float neg_wrong, neg_right;

            /* Direct polarity */

            pos_wrong = pos_c_w(0, col);
            pos_right = pos_total_w - pos_wrong;

            neg_right = neg_c_w(0, col);
            neg_wrong = neg_total_w - neg_right;

            err = sqrt(pos_right * neg_wrong) + sqrt(pos_wrong * neg_right);

            if(use_fast_log)
            {
              h_pos = .5f * fast_log((pos_right + eps) / (pos_wrong + eps));
              h_neg = .5f * fast_log((neg_wrong + eps) / (neg_right + eps));
            }
            else
            {
              h_pos = .5f * log((pos_right + eps) / (pos_wrong + eps));
              h_neg = .5f * log((neg_wrong + eps) / (neg_right + eps));
            }

            if( err < min_err )
            {
                min_err = err;
                min_row = row;
                min_thr = (sorted_d(0, col) + sorted_d(0, col + 1)) / 2;
                min_pol = +1;
                min_pos = h_pos;
                min_neg = h_neg;
            }

            /* Opposite polarity */

            swap(pos_right, pos_wrong);
            swap(neg_right, neg_wrong);

            err = sqrt(pos_right * neg_wrong) + sqrt(pos_wrong * neg_right);

            if( err < min_err )
            {
                min_err = err;
                min_row = row;
                min_thr = (sorted_d(0, col) + sorted_d(0, col + 1)) / 2;
                min_pol = -1;
                min_pos = -h_pos;
                min_neg = -h_neg;
            }

        }
    }

    threshold_ = min_thr;
    polarity_ = min_pol;
    pos_value_ = min_pos;
    neg_value_ = min_neg;

    return min_row;
}

float Stump::predict(int value) const
{
    return polarity_ * (value - threshold_) > 0 ? pos_value_ : neg_value_;
}


void read(const FileNode& node, Stump& s, const Stump& default_value)
{
    if( node.empty() )
        s = default_value;
    else
        s.read(node);
}

void write(FileStorage& fs, String&, const Stump& s)
{
    s.write(fs);
}

} /* namespace xobjdetect */
} /* namespace cv */
