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
using std::string;

using std::min;
using std::max;

namespace cv
{
namespace xobjdetect
{

static bool overlap(const Rect& r, const vector<Rect>& gt)
{
    for( size_t i = 0; i < gt.size(); ++i )
        if( (r & gt[i]).area() )
            return true;
    return false;
}

void ICFDetector::train(const vector<string>& image_filenames,
                        const vector< vector<Rect> >& labelling,
                        ICFDetectorParams params)
{
    Size model_size(params.model_n_cols, params.model_n_rows);

    vector<Mat> samples; /* positive samples + negative samples */
    Mat sample, resized_sample;
    size_t pos_count = 0;
    for( size_t i = 0; i < image_filenames.size(); ++i, ++pos_count )
    {
        Mat img = imread(String(image_filenames[i].c_str()));
        for( size_t j = 0; j < labelling[i].size(); ++j )
        {
            Rect r = labelling[i][j];
            if( r.x > img.cols || r.y > img.rows )
                continue;

            sample = img.colRange(max(r.x, 0), min(r.width, img.cols))
                        .rowRange(max(r.y, 0), min(r.height, img.rows));

            resize(sample, resized_sample, model_size);

            samples.push_back(resized_sample);
        }
    }

    int neg_count = 0;
    RNG rng;
    for( size_t i = 0; i < image_filenames.size(); ++i )
    {
        Mat img = imread(String(image_filenames[i].c_str()));
        for( size_t j = 0; j < pos_count / image_filenames.size() + 1; )
        {
            Rect r;
            r.x = rng.uniform(0, img.cols);
            r.width = rng.uniform(r.x + 1, img.cols);
            r.y = rng.uniform(0, img.rows);
            r.height = rng.uniform(r.y + 1, img.rows);

            if( !overlap(r, labelling[i]) )
            {
                sample = img.colRange(r.x, r.width).rowRange(r.y, r.height);
                //resize(sample, resized_sample);
                samples.push_back(resized_sample);
                ++neg_count;
                ++j;
            }
        }
    }

    Mat_<int> labels(1, pos_count + neg_count);
    for( size_t i = 0; i < pos_count; ++i)
        labels(0, i) = 1;
    for( size_t i = pos_count; i < pos_count + neg_count; ++i )
        labels(0, i) = -1;

    vector<Point3i> features = generateFeatures(model_size);
    ACFFeatureEvaluator feature_evaluator(features);

    Mat_<int> data(features.size(), samples.size());
    Mat_<int> feature_col;

    vector<Mat> channels;
    for( size_t i = 0; i < samples.size(); ++i )
    {
        computeChannels(samples[i], channels);
        feature_evaluator.setChannels(channels);
        feature_evaluator.evaluateAll(feature_col);
        for( int j = 0; j < feature_col.rows; ++j )
            data(i, j) = feature_col(0, j);
    }

    WaldBoostParams wparams;
    wparams.weak_count = params.weak_count;
    wparams.alpha = 0.001;

    WaldBoost waldboost(wparams);
    waldboost.train(data, labels);
}

bool ICFDetector::save(const string&)
{
    return true;
}

} /* namespace xobjdetect */
} /* namespace cv */
