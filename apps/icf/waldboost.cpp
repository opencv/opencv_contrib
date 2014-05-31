#include <cmath>

#include <algorithm>
using std::swap;

#include "waldboost.hpp"

using cv::Mat;
using cv::Mat_;
using cv::sort;
using cv::sortIdx;
using cv::adas::Stump;
using cv::adas::WaldBoost;
using cv::Ptr;

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

int Stump::train(const Mat& data, const Mat& labels, const Mat& weights)
{
    CV_Assert(labels.rows == 1 && labels.cols == data.cols);
    CV_Assert(weights.rows == 1 && weights.cols == data.cols);
    /* Assert that data and labels have int type */
    /* Assert that weights have float type */


    /* Prepare labels for each feature rearranged according to sorted order */
    Mat sorted_labels(data.rows, data.cols, labels.type());
    Mat sorted_weights(data.rows, data.cols, weights.type());
    Mat indices;
    sortIdx(data, indices, cv::SORT_EVERY_ROW | cv::SORT_ASCENDING);
    for( int row = 0; row < indices.rows; ++row )
    {
        for( int col = 0; col < indices.cols; ++col )
        {
            sorted_labels.at<int>(row, col) =
                labels.at<int>(0, indices.at<int>(row, col));
            sorted_weights.at<float>(row, col) =
                weights.at<float>(0, indices.at<float>(row, col));
        }
    }

    /* Sort feature values */
    Mat sorted_data(data.rows, data.cols, data.type());
    sort(data, sorted_data, cv::SORT_EVERY_ROW | cv::SORT_ASCENDING);

    /* Split positive and negative weights */
    Mat pos_weights = Mat::zeros(sorted_weights.rows, sorted_weights.cols,
        sorted_weights.type());
    Mat neg_weights = Mat::zeros(sorted_weights.rows, sorted_weights.cols,
        sorted_weights.type());
    for( int row = 0; row < data.rows; ++row )
    {
        for( int col = 0; col < data.cols; ++col )
        {
            if( sorted_labels.at<int>(row, col) == +1 )
            {
                pos_weights.at<float>(row, col) =
                    sorted_weights.at<float>(row, col);
            }
            else
            {
                neg_weights.at<float>(row, col) =
                    sorted_weights.at<float>(row, col);
            }
        }
    }

    /* Compute cumulative sums for fast stump error computation */
    Mat pos_cum_weights = Mat::zeros(sorted_weights.rows, sorted_weights.cols,
        sorted_weights.type());
    Mat neg_cum_weights = Mat::zeros(sorted_weights.rows, sorted_weights.cols,
        sorted_weights.type());
    cumsum(pos_weights, pos_cum_weights);
    cumsum(neg_weights, neg_cum_weights);

    /* Compute total weights of positive and negative samples */
    float pos_total_weight = pos_cum_weights.at<float>(0, weights.cols - 1);
    float neg_total_weight = neg_cum_weights.at<float>(0, weights.cols - 1);


    float eps = 1. / 4 * labels.cols;

    /* Compute minimal error */
    float min_err = FLT_MAX;
    int min_row = -1;
    int min_col = -1;
    int min_polarity = 0;
    float min_pos_value = 1, min_neg_value = -1;

    for( int row = 0; row < sorted_weights.rows; ++row )
    {
        for( int col = 0; col < sorted_weights.cols - 1; ++col )
        {
            float err, h_pos, h_neg;

            // Direct polarity

            float pos_wrong = pos_cum_weights.at<float>(row, col);
            float pos_right = pos_total_weight - pos_wrong;

            float neg_right = neg_cum_weights.at<float>(row, col);
            float neg_wrong = neg_total_weight - neg_right;

            h_pos = .5 * log((pos_right + eps) / (pos_wrong + eps));
            h_neg = .5 * log((neg_wrong + eps) / (neg_right + eps));

            err = sqrt(pos_right * neg_wrong) + sqrt(pos_wrong * neg_right);

            if( err < min_err )
            {
                min_err = err;
                min_row = row;
                min_col = col;
                min_polarity = +1;
                min_pos_value = h_pos;
                min_neg_value = h_neg;
            }

            // Opposite polarity
            swap(pos_right, pos_wrong);
            swap(neg_right, neg_wrong);

            h_pos = -h_pos;
            h_neg = -h_neg;

            err = sqrt(pos_right * neg_wrong) + sqrt(pos_wrong * neg_right);


            if( err < min_err )
            {
                min_err = err;
                min_row = row;
                min_col = col;
                min_polarity = -1;
                min_pos_value = h_pos;
                min_neg_value = h_neg;
            }
        }
    }

    /* Compute threshold, store found values in fields */
    threshold_ = ( sorted_data.at<int>(min_row, min_col) +
                   sorted_data.at<int>(min_row, min_col + 1) ) / 2;
    polarity_ = min_polarity;
    pos_value_ = min_pos_value;
    neg_value_ = min_neg_value;

    return min_row;
}

float Stump::predict(int value)
{
    return polarity_ * (value - threshold_) > 0 ? pos_value_ : neg_value_;
}
