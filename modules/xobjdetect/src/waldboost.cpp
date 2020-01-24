/*
By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install,
copy or use the software.


                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Copyright (C) 2000-2015, Intel Corporation, all rights reserved.
Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
Copyright (C) 2009-2015, NVIDIA Corporation, all rights reserved.
Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
Copyright (C) 2015, OpenCV Foundation, all rights reserved.
Copyright (C) 2015, Itseez Inc., all rights reserved.
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
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall copyright holders or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/

#include "precomp.hpp"

namespace cv {
namespace xobjdetect {

static void compute_cdf(const Mat1b& data,
                        const Mat1f& weights,
                        Mat1f& cdf)
{
    for (int i = 0; i < cdf.cols; ++i)
        cdf(0, i) = 0;

    for (int i = 0; i < weights.cols; ++i) {
        cdf(0, data(0, i)) += weights(0, i);
    }

    for (int i = 1; i < cdf.cols; ++i) {
        cdf(0, i) += cdf(0, i - 1);
    }
}

static void compute_min_step(const Mat &data_pos, const Mat &data_neg, size_t n_bins,
                      Mat &data_min, Mat &data_step)
{
    // Check that quantized data will fit in unsigned char
    assert(n_bins <= 256);

    assert(data_pos.rows == data_neg.rows);

    Mat reduced_pos, reduced_neg;

    reduce(data_pos, reduced_pos, 1, CV_REDUCE_MIN);
    reduce(data_neg, reduced_neg, 1, CV_REDUCE_MIN);
    min(reduced_pos, reduced_neg, data_min);
    data_min -= 0.01;

    Mat data_max;
    reduce(data_pos, reduced_pos, 1, CV_REDUCE_MAX);
    reduce(data_neg, reduced_neg, 1, CV_REDUCE_MAX);
    max(reduced_pos, reduced_neg, data_max);
    data_max += 0.01;

    data_step = (data_max - data_min) / (double)(n_bins - 1);
}

static void quantize_data(Mat &data, Mat1f &data_min, Mat1f &data_step)
{
//#pragma omp parallel for
    for (int col = 0; col < data.cols; ++col) {
        data.col(col) -= data_min;
        data.col(col) /= data_step;
    }
    data.convertTo(data, CV_8U);
}

WaldBoost::WaldBoost(int weak_count):
    weak_count_(weak_count),
    thresholds_(),
    alphas_(),
    feature_indices_(),
    polarities_(),
    cascade_thresholds_() {}

WaldBoost::WaldBoost():
    weak_count_(),
    thresholds_(),
    alphas_(),
    feature_indices_(),
    polarities_(),
    cascade_thresholds_() {}

std::vector<int> WaldBoost::get_feature_indices()
{
    return feature_indices_;
}

void WaldBoost::detect(Ptr<CvFeatureEvaluator> eval,
            const Mat& img, const std::vector<float>& scales,
            std::vector<Rect>& bboxes, Mat1f& confidences)
{
    bboxes.clear();
    confidences.release();

    Mat resized_img;
    int step = 4;
    float h;
    for (size_t i = 0; i < scales.size(); ++i) {
        float scale = scales[i];
        resize(img, resized_img, Size(), scale, scale, INTER_LINEAR_EXACT);
        eval->setImage(resized_img, 0, 0, feature_indices_);
        int n_rows = (int)(24 / scale);
        int n_cols = (int)(24 / scale);
        for (int r = 0; r + 24 < resized_img.rows; r += step) {
            for (int c = 0; c + 24 < resized_img.cols; c += step) {
                //eval->setImage(resized_img(Rect(c, r, 24, 24)), 0, 0);
                eval->setWindow(Point(c, r));
                if (predict(eval, &h) == +1) {
                    int row = (int)(r / scale);
                    int col = (int)(c / scale);
                    bboxes.push_back(Rect(col, row, n_cols, n_rows));
                    confidences.push_back(h);
                }
            }
        }
    }
    groupRectangles(bboxes, 3, 0.7);
}

void WaldBoost::detect(Ptr<CvFeatureEvaluator> eval,
            const Mat& img, const std::vector<float>& scales,
            std::vector<Rect>& bboxes, std::vector<double>& confidences)
{
    bboxes.clear();
    confidences.clear();

    Mat resized_img;
    int step = 4;
    float h;
    for (size_t i = 0; i < scales.size(); ++i) {
        float scale = scales[i];
        resize(img, resized_img, Size(), scale, scale, INTER_LINEAR_EXACT);
        eval->setImage(resized_img, 0, 0, feature_indices_);
        int n_rows = (int)(24 / scale);
        int n_cols = (int)(24 / scale);
        for (int r = 0; r + 24 < resized_img.rows; r += step) {
            for (int c = 0; c + 24 < resized_img.cols; c += step) {
                eval->setWindow(Point(c, r));
                if (predict(eval, &h) == +1) {
                    int row = (int)(r / scale);
                    int col = (int)(c / scale);
                    bboxes.push_back(Rect(col, row, n_cols, n_rows));
                    confidences.push_back(h);
                }
            }
        }
    }
    std::vector<int> levels(bboxes.size(), 0);
    groupRectangles(bboxes, levels, confidences, 3, 0.7);
}

void WaldBoost::fit(Mat& data_pos, Mat& data_neg)
{
    // data_pos: F x N_pos
    // data_neg: F x N_neg
    // every feature corresponds to row
    // every sample corresponds to column
    assert(data_pos.rows >= weak_count_);
    assert(data_pos.rows == data_neg.rows);

    std::vector<bool> feature_ignore;
    for (int i = 0; i < data_pos.rows; ++i) {
        feature_ignore.push_back(false);
    }

    Mat1f pos_weights(1, data_pos.cols, 1.0f / (2 * data_pos.cols));
    Mat1f neg_weights(1, data_neg.cols, 1.0f / (2 * data_neg.cols));
    Mat1f pos_trace(1, data_pos.cols, 0.0f);
    Mat1f neg_trace(1, data_neg.cols, 0.0f);

    bool quantize = false;
    if (data_pos.type() != CV_8U) {
        std::cerr << "quantize" << std::endl;
        quantize = true;
    }

    Mat1f data_min, data_step;
    int n_bins = 256;
    if (quantize) {
        compute_min_step(data_pos, data_neg, n_bins, data_min, data_step);
        quantize_data(data_pos, data_min, data_step);
        quantize_data(data_neg, data_min, data_step);
    }

    std::cerr << "pos=" << data_pos.cols << " neg=" << data_neg.cols << std::endl;
    for (int i = 0; i < weak_count_; ++i) {
        // Train weak learner with lowest error using weights
        double min_err = DBL_MAX;
        int min_feature_ind = -1;
        int min_polarity = 0;
        int threshold_q = 0;
        float min_threshold = 0;
//#pragma omp parallel for
        for (int feat_i = 0; feat_i < data_pos.rows; ++feat_i) {
            if (feature_ignore[feat_i])
                continue;

            // Construct cdf
            Mat1f pos_cdf(1, n_bins), neg_cdf(1, n_bins);
            compute_cdf(data_pos.row(feat_i), pos_weights, pos_cdf);
            compute_cdf(data_neg.row(feat_i), neg_weights, neg_cdf);

            float neg_total = (float)sum(neg_weights)[0];
            Mat1f err_direct = pos_cdf + neg_total - neg_cdf;
            Mat1f err_backward = 1.0f - err_direct;

            int idx1[2], idx2[2];
            double err1, err2;
            minMaxIdx(err_direct, &err1, NULL, idx1);
            minMaxIdx(err_backward, &err2, NULL, idx2);
//#pragma omp critical
            {
            if (min(err1, err2) < min_err) {
                if (err1 < err2) {
                    min_err = err1;
                    min_polarity = +1;
                    threshold_q = idx1[1];
                } else {
                    min_err = err2;
                    min_polarity = -1;
                    threshold_q = idx2[1];
                }
                min_feature_ind = feat_i;
                if (quantize) {
                    min_threshold = data_min(feat_i, 0) + data_step(feat_i, 0) *
                        (threshold_q + .5f);
                } else {
                    min_threshold = threshold_q + .5f;
                }
            }
            }
        }


        float alpha = .5f * (float)log((1 - min_err) / min_err);
        alphas_.push_back(alpha);
        feature_indices_.push_back(min_feature_ind);
        thresholds_.push_back(min_threshold);
        polarities_.push_back(min_polarity);
        feature_ignore[min_feature_ind] = true;

        double loss = 0;
        // Update positive weights
        for (int j = 0; j < data_pos.cols; ++j) {
            int val = data_pos.at<unsigned char>(min_feature_ind, j);
            int label = min_polarity * (val - threshold_q) >= 0 ? +1 : -1;
            pos_weights(0, j) *= exp(-alpha * label);
            pos_trace(0, j) += alpha * label;
            loss += exp(-pos_trace(0, j)) / (2.0f * data_pos.cols);
        }

        // Update negative weights
        for (int j = 0; j < data_neg.cols; ++j) {
            int val = data_neg.at<unsigned char>(min_feature_ind, j);
            int label = min_polarity * (val - threshold_q) >= 0 ? +1 : -1;
            neg_weights(0, j) *= exp(alpha * label);
            neg_trace(0, j) += alpha * label;
            loss += exp(+neg_trace(0, j)) / (2.0f * data_neg.cols);
        }
        double cascade_threshold = -1;
        minMaxIdx(pos_trace, &cascade_threshold);
        cascade_thresholds_.push_back((float)cascade_threshold);

        std::cerr << "i=" << std::setw(4) << i;
        std::cerr << " feat=" << std::setw(5) << min_feature_ind;
        std::cerr << " thr=" << std::setw(3) << threshold_q;
        std::cerr << " casthr=" << std::fixed << std::setprecision(3)
             << cascade_threshold;
        std::cerr <<  " alpha=" << std::fixed << std::setprecision(3)
             << alpha << " err=" << std::fixed << std::setprecision(3) << min_err
             << " loss=" << std::scientific << loss << std::endl;

        //int pos = 0;
        //for (int j = 0; j < data_pos.cols; ++j) {
        //    if (pos_trace(0, j) > cascade_threshold - 0.5) {
        //        pos_trace(0, pos) = pos_trace(0, j);
        //        data_pos.col(j).copyTo(data_pos.col(pos));
        //        pos_weights(0, pos) = pos_weights(0, j);
        //        pos += 1;
        //    }
        //}
        //std::cerr << "pos " << data_pos.cols << "/" << pos << std::endl;
        //pos_trace = pos_trace.colRange(0, pos);
        //data_pos = data_pos.colRange(0, pos);
        //pos_weights = pos_weights.colRange(0, pos);

        int pos = 0;
        for (int j = 0; j < data_neg.cols; ++j) {
            if (neg_trace(0, j) > cascade_threshold - 0.5) {
                neg_trace(0, pos) = neg_trace(0, j);
                data_neg.col(j).copyTo(data_neg.col(pos));
                neg_weights(0, pos) = neg_weights(0, j);
                pos += 1;
            }
        }
        std::cerr << "neg " << data_neg.cols << "/" << pos << std::endl;
        neg_trace = neg_trace.colRange(0, pos);
        data_neg = data_neg.colRange(0, pos);
        neg_weights = neg_weights.colRange(0, pos);


        if (loss < 1e-50 || min_err > 0.5) {
            std::cerr << "Stopping early. loss=" << loss << " min_err=" << min_err << std::endl;
            weak_count_ = i + 1;
            break;
        }

        // Avoid crashing on next Mat creation
        if (pos <= 1) {
            std::cerr << "Stopping early. pos=" << pos << std::endl;
            weak_count_ = i + 1;
            break;
        }

        // Normalize weights
        double z = (sum(pos_weights) + sum(neg_weights))[0];
        pos_weights /= z;
        neg_weights /= z;
    }
}

int WaldBoost::predict(Ptr<CvFeatureEvaluator> eval, float *h) const
{
    assert(feature_indices_.size() == size_t(weak_count_));
    assert(cascade_thresholds_.size() == size_t(weak_count_));
    float res = 0;
    int count = weak_count_;
    for (int i = 0; i < count; ++i) {
        float val = (*eval)(feature_indices_[i]);
        int label = polarities_[i] * (val - thresholds_[i]) > 0 ? +1: -1;
        res += alphas_[i] * label;
        if (res < cascade_thresholds_[i]) {
            return -1;
        }
    }
    *h = res;
    return res > cascade_thresholds_[count - 1] ? +1 : -1;
}

void WaldBoost::write(FileStorage &fs) const
{
    fs << "{";
    fs << "waldboost_params"
       << "{" << "weak_count" << weak_count_ << "}";

    fs << "thresholds" << "[";
    for (size_t i = 0; i < thresholds_.size(); ++i)
        fs << thresholds_[i];
    fs << "]";

    fs << "alphas" << "[";
    for (size_t i = 0; i < alphas_.size(); ++i)
        fs << alphas_[i];
    fs << "]";

    fs << "polarities" << "[";
    for (size_t i = 0; i < polarities_.size(); ++i)
        fs << polarities_[i];
    fs << "]";

    fs << "cascade_thresholds" << "[";
    for (size_t i = 0; i < cascade_thresholds_.size(); ++i)
        fs << cascade_thresholds_[i];
    fs << "]";

    fs << "feature_indices" << "[";
    for (size_t i = 0; i < feature_indices_.size(); ++i)
        fs << feature_indices_[i];
    fs << "]";

    fs << "}";
}

void WaldBoost::read(const FileNode &node)
{
    weak_count_ = (int)(node["waldboost_params"]["weak_count"]);
    thresholds_.resize(weak_count_);
    alphas_.resize(weak_count_);
    polarities_.resize(weak_count_);
    cascade_thresholds_.resize(weak_count_);
    feature_indices_.resize(weak_count_);

    FileNodeIterator n;

    n = node["thresholds"].begin();
    for (int i = 0; i < weak_count_; ++i, ++n)
        *n >> thresholds_[i];

    n = node["alphas"].begin();
    for (int i = 0; i < weak_count_; ++i, ++n)
        *n >> alphas_[i];

    n = node["polarities"].begin();
    for (int i = 0; i < weak_count_; ++i, ++n)
        *n >> polarities_[i];

    n = node["cascade_thresholds"].begin();
    for (int i = 0; i < weak_count_; ++i, ++n)
        *n >> cascade_thresholds_[i];

    n = node["feature_indices"].begin();
    for (int i = 0; i < weak_count_; ++i, ++n)
        *n >> feature_indices_[i];
}

void WaldBoost::reset(int weak_count)
{
    weak_count_ = weak_count;
    thresholds_.clear();
    alphas_.clear();
    feature_indices_.clear();
    polarities_.clear();
    cascade_thresholds_.clear();
}

WaldBoost::~WaldBoost()
{
}

}
}
