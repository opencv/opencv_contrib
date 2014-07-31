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

using std::vector;

using std::min;

#include <iostream>
using std::cout;
using std::endl;

namespace cv
{
namespace xobjdetect
{

static bool isNull(const Mat_<int> &m)
{
    bool null_data = true;
    for( int row = 0; row < m.rows; ++row )
    {
        for( int col = 0; col < m.cols; ++col )
            if( m.at<int>(row, col) )
                null_data = false;
    }
    return null_data;
}

class FeatureEvaluatorImpl : public FeatureEvaluator
{
public:
    FeatureEvaluatorImpl(const vector<vector<int> >& features):
        features_(features), channels_(), position_()
    {
        CV_Assert(features.size() > 0);
    }

    virtual void assertChannels()
    {
        bool null_data = true;
        for( size_t i = 0; i < channels_.size(); ++i )
            null_data &= isNull(channels_[i]);
        CV_Assert(!null_data);
    }

    virtual void evaluateAll(OutputArray feature_values) const
    {
        Mat_<int> feature_vals(1, (int)features_.size());
        for( int i = 0; i < (int)features_.size(); ++i )
        {
            feature_vals(0, i) = evaluate(i);
        }
        feature_values.assign(feature_vals);
    }

protected:
    /* Features to evaluate */
    vector<vector<int> > features_;
    /* Channels for feature evaluation */
    std::vector<Mat> channels_;
    /* Channels window position */
    Size position_;
};

class ICFFeatureEvaluatorImpl : public FeatureEvaluatorImpl
{
public:
    ICFFeatureEvaluatorImpl(const vector<vector<int> >& features):
        FeatureEvaluatorImpl(features)
    {
    }

    virtual void setChannels(InputArrayOfArrays channels);
    virtual void setPosition(Size position);
    virtual int evaluate(size_t feature_ind) const;
};

void ICFFeatureEvaluatorImpl::setChannels(InputArrayOfArrays channels)
{
    channels_.clear();
    vector<Mat> ch;
    channels.getMatVector(ch);
    CV_Assert(ch.size() == 10);

    for( size_t i = 0; i < ch.size(); ++i )
    {
        const Mat &channel = ch[i];
        Mat integral_channel;
        integral(channel, integral_channel, CV_32F);
        Mat_<int> chan(integral_channel.rows, integral_channel.cols);
        for( int row = 0; row < integral_channel.rows; ++row )
            for( int col = 0; col < integral_channel.cols; ++col )
                chan(row, col) = (int)integral_channel.at<float>(row, col);
        channels_.push_back(chan.clone());
    }
}

void ICFFeatureEvaluatorImpl::setPosition(Size position)
{
    position_ = position;
}

int ICFFeatureEvaluatorImpl::evaluate(size_t feature_ind) const
{
    CV_Assert(channels_.size() == 10);
    CV_Assert(feature_ind < features_.size());

    const vector<int>& feature = features_[feature_ind];
    int x = feature[0] + position_.height;
    int y = feature[1] + position_.width;
    int x_to = feature[2] + position_.height;
    int y_to = feature[3] + position_.width;
    int n = feature[4];
    const Mat_<int>& ch = channels_[n];
    return ch(y_to + 1, x_to + 1) - ch(y, x_to + 1) - ch(y_to + 1, x) + ch(y, x);
}

class ACFFeatureEvaluatorImpl : public FeatureEvaluatorImpl
{
public:
    ACFFeatureEvaluatorImpl(const vector<vector<int> >& features):
        FeatureEvaluatorImpl(features)
    {
    }

    virtual void setChannels(InputArrayOfArrays channels);
    virtual void setPosition(Size position);
    virtual int evaluate(size_t feature_ind) const;
};

void ACFFeatureEvaluatorImpl::setChannels(InputArrayOfArrays channels)
{
    channels_.clear();
    vector<Mat> ch;
    channels.getMatVector(ch);
    CV_Assert(ch.size() == 10);

    for( size_t i = 0; i < ch.size(); ++i )
    {
        const Mat &channel = ch[i];
        Mat_<int> acf_channel = Mat_<int>::zeros(channel.rows / 4, channel.cols / 4);
        for( int row = 0; row < channel.rows; row += 4 )
        {
            for( int col = 0; col < channel.cols; col += 4 )
            {
                int sum = 0;
                for( int cell_row = row; cell_row < row + 4; ++cell_row )
                    for( int cell_col = col; cell_col < col + 4; ++cell_col )
                        sum += (int)channel.at<float>(cell_row, cell_col);

                acf_channel(row / 4, col / 4) = sum;
            }
        }

        channels_.push_back(acf_channel.clone());
    }
}

void ACFFeatureEvaluatorImpl::setPosition(Size position)
{
    position_ = Size(position.width / 4, position.height / 4);
}

int ACFFeatureEvaluatorImpl::evaluate(size_t feature_ind) const
{
    CV_Assert(channels_.size() == 10);
    CV_Assert(feature_ind < features_.size());

    const vector<int>& feature = features_[feature_ind];
    int x = feature[0];
    int y = feature[1];
    int n = feature[2];
    return channels_[n].at<int>(y + position_.width, x + position_.height);
}

Ptr<FeatureEvaluator> createFeatureEvaluator(
    const vector<vector<int> >& features, const std::string& type)
{
    FeatureEvaluator *evaluator = NULL;
    if( type == "acf" )
        evaluator = new ACFFeatureEvaluatorImpl(features);
    else if( type == "icf" )
        evaluator = new ICFFeatureEvaluatorImpl(features);
    else
        CV_Assert(false);

    return Ptr<FeatureEvaluator>(evaluator);
}

vector<vector<int> > generateFeatures(Size window_size, const std::string& type,
                                      int count, int channel_count)
{
    CV_Assert(count > 0);
    vector<vector<int> > features;
    if( type == "acf" )
    {
        int cur_count = 0;
        int max_count = window_size.width * window_size.height / 16;
        count = min(count, max_count);
        for( int x = 0; x < window_size.width / 4; ++x )
            for( int y = 0; y < window_size.height / 4; ++y )
                for( int n = 0; n < channel_count; ++n )
                {
                    int f[] = {x, y, n};
                    vector<int> feature(f, f + sizeof(f) / sizeof(*f));
                    features.push_back(feature);
                    if( (cur_count += 1) == count )
                        break;
                }
    }
    else if( type == "icf" )
    {
        RNG rng;
        for( int i = 0; i < count; ++i )
        {
            int x = rng.uniform(0, window_size.width - 1);
            int y = rng.uniform(0, window_size.height - 1);
            int x_to = rng.uniform(x, window_size.width - 1);
            int y_to = rng.uniform(y, window_size.height - 1);
            int n = rng.uniform(0, channel_count - 1);
            int f[] = {x, y, x_to, y_to, n};
            vector<int> feature(f, f + sizeof(f) / sizeof(*f));
            features.push_back(feature);
        }
    }
    else
        CV_Assert(false);

    return features;
}

void computeChannels(InputArray image, vector<Mat>& channels)
{
    Mat src(image.getMat().rows, image.getMat().cols, CV_32FC3);
    image.getMat().convertTo(src, CV_32FC3, 1./255);

    Mat_<float> grad;
    Mat luv, gray;
    cvtColor(src, gray, CV_RGB2GRAY);
    cvtColor(src, luv, CV_RGB2Luv);

    Mat_<float> row_der, col_der;
    Sobel(gray, row_der, CV_32F, 0, 1);
    Sobel(gray, col_der, CV_32F, 1, 0);

    magnitude(row_der, col_der, grad);

    Mat_<Vec6f> hist = Mat_<Vec6f>::zeros(grad.rows, grad.cols);
    const float to_deg = 180 / 3.1415926f;
    for (int row = 0; row < grad.rows; ++row) {
        for (int col = 0; col < grad.cols; ++col) {
            float angle = atan2(row_der(row, col), col_der(row, col)) * to_deg;
            if (angle < 0)
                angle += 180;
            int ind = (int)(angle / 30);

            // If angle == 180, prevent index overflow
            if (ind == 6)
                ind = 5;

            hist(row, col)[ind] = grad(row, col) * 255;
        }
    }

    channels.clear();

    Mat luv_channels[3];
    split(luv, luv_channels);
    for( int i = 0; i < 3; ++i )
        channels.push_back(luv_channels[i]);

    channels.push_back(grad);

    vector<Mat> hist_channels;
    split(hist, hist_channels);

    for( size_t i = 0; i < hist_channels.size(); ++i )
        channels.push_back(hist_channels[i]);
}

} /* namespace xobjdetect */
} /* namespace cv */
