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

#include <sstream>
using std::ostringstream;

#include <iomanip>
using std::setfill;
using std::setw;

#include <iostream>
using std::cout;
using std::endl;

#include "precomp.hpp"

using std::vector;
using std::string;

using std::min;
using std::max;


namespace cv
{

namespace xobjdetect
{


void ICFDetector::train(const vector<String>& pos_filenames,
                        const vector<String>& bg_filenames,
                        ICFDetectorParams params)
{
  
    int color;
    if(params.is_grayscale == false)
      color = IMREAD_COLOR;
    else
      color = IMREAD_GRAYSCALE;

    model_n_rows_ = params.model_n_rows;
    model_n_cols_ = params.model_n_cols;
    ftype_ = params.features_type;

    Size model_size(params.model_n_cols, params.model_n_rows);

    vector<Mat> samples; /* positive samples + negative samples */
    Mat sample, resized_sample;
    int pos_count = 0;
  
    for( size_t i = 0; i < pos_filenames.size(); ++i, ++pos_count )
    {
        cout << setw(6) << (i + 1) << "/" << pos_filenames.size() << "\r";
        Mat img = imread(pos_filenames[i], color);
        resize(img, resized_sample, model_size);
        samples.push_back(resized_sample.clone());
    }
    cout << "\n";

    int neg_count = 0;
    RNG rng;
    for( size_t i = 0; i < bg_filenames.size(); ++i )
    {
        cout << setw(6) << (i + 1) << "/" << bg_filenames.size() << "\r";
        Mat img = imread(bg_filenames[i], color);
        for( int j = 0; j < params.bg_per_image; ++j, ++neg_count)
        {
            Rect r;
            r.x = rng.uniform(0, img.cols-model_size.width);
            r.width = model_size.width;
            r.y = rng.uniform(0, img.rows-model_size.height);
            r.height = model_size.height;
            sample = img.colRange(r.x, r.x + r.width).rowRange(r.y, r.y + r.height);
            samples.push_back(sample.clone());
        }
    }
    cout << "\n";

    Mat_<int> labels(1, pos_count + neg_count);
    for( int i = 0; i < pos_count; ++i)
        labels(0, i) = 1;
    for( int i = pos_count; i < pos_count + neg_count; ++i )
        labels(0, i) = -1;

    
    vector<vector<int> > features;
    if(params.is_grayscale == false)
      features = generateFeatures(model_size, params.features_type,  params.feature_count, 10);
    else
      features = generateFeatures(model_size, params.features_type,  params.feature_count, 7);
    
    Ptr<FeatureEvaluator> evaluator = createFeatureEvaluator(features, params.features_type);


    Mat_<int> data = Mat_<int>::zeros((int)features.size(), (int)samples.size());
    Mat_<int> feature_col(1, (int)samples.size());

    vector<Mat> channels;
    for( int i = 0; i < (int)samples.size(); ++i )
    {
        cout << setw(6) << i << "/" << samples.size() << "\r";
        computeChannels(samples[i], channels);
        evaluator->setChannels(channels);
        //evaluator->assertChannels();
        evaluator->evaluateAll(feature_col);

        CV_Assert(feature_col.cols == (int)features.size());

        for( int j = 0; j < feature_col.cols; ++j )
            data(j, i) = feature_col(0, j);
    }
    cout << "\n";
    samples.clear();
        
    WaldBoostParams wparams;
    wparams.weak_count = params.weak_count;
    wparams.alpha = params.alpha;

    waldboost_ = createWaldBoost(wparams);
    vector<int> indices = waldboost_->train(data, labels, params.use_fast_log);
    cout << "indices: ";
    for( size_t i = 0; i < indices.size(); ++i )
        cout << indices[i] << " ";
    cout << endl;

    features_.clear();
    for( size_t i = 0; i < indices.size(); ++i )
        features_.push_back(features[indices[i]]);
}

void ICFDetector::write(FileStorage& fs) const
{
    fs << "{";
    fs << "model_n_rows" << model_n_rows_;
    fs << "model_n_cols" << model_n_cols_;
    fs << "ftype" << String(ftype_.c_str());
    fs << "waldboost";
    waldboost_->write(fs);
    fs << "features" << "[";
    for( size_t i = 0; i < features_.size(); ++i )
    {
        fs << features_[i];
    }
    fs << "]";
    fs << "}";
}

void ICFDetector::read(const FileNode& node)
{
    waldboost_ = Ptr<WaldBoost>(createWaldBoost(WaldBoostParams()));
    String f_temp;
    node["model_n_rows"] >> model_n_rows_;
    node["model_n_cols"] >> model_n_cols_;
    f_temp = (String)node["ftype"];    
    this->ftype_ = (string)f_temp.c_str();
    waldboost_->read(node["waldboost"]);
    FileNode features = node["features"];
    features_.clear();
    vector<int> p;
    for( FileNodeIterator n = features.begin(); n != features.end(); ++n )
    {
        (*n) >> p;
        features_.push_back(p);
    }
}

void ICFDetector::detect(const Mat& img, vector<Rect>& objects,
    float scaleFactor, Size minSize, Size maxSize, float threshold, int slidingStep, std::vector<float>& values)
{
    
    
    float scale_from = min(model_n_cols_ / (float)maxSize.width,
                           model_n_rows_ / (float)maxSize.height);
    float scale_to = max(model_n_cols_ / (float)minSize.width,
                         model_n_rows_ / (float)minSize.height);
    objects.clear();
    Ptr<FeatureEvaluator> evaluator = createFeatureEvaluator(features_, ftype_);
    Mat rescaled_image;
    vector<Mat> channels;
    
    for( float scale = scale_from; scale < scale_to + 0.001; scale *= scaleFactor )
    {
        int new_width = int(img.cols * scale);
        new_width -= new_width % 4;
        int new_height = int(img.rows * scale);
        new_height -= new_height % 4;
        
        resize(img, rescaled_image, Size(new_width, new_height));
        computeChannels(rescaled_image, channels);
        evaluator->setChannels(channels);
        for( int row = 0; row <= rescaled_image.rows - model_n_rows_; row += slidingStep)
        {
            for( int col = 0; col <= rescaled_image.cols - model_n_cols_;
                col += slidingStep )
            {
                evaluator->setPosition(Size(row, col));
                float value = waldboost_->predict(evaluator);
                if( value > threshold )
                {
                    values.push_back(value);
                    int x = (int)(col / scale);
                    int y = (int)(row / scale);
                    int width = (int)(model_n_cols_ / scale);
                    int height = (int)(model_n_rows_ / scale);
                    objects.push_back(Rect(x, y, width, height));
                }
            }
        }

    }
    
}

void ICFDetector::detect(const Mat& img, vector<Rect>& objects,
    float minScaleFactor, float maxScaleFactor, float factorStep, float threshold, int slidingStep, std::vector<float>& values)
{

    if(factorStep <= 0)
    {
      CV_Error_(CV_StsBadArg, ("factorStep must be > 0"));
    }
    
    objects.clear();
    Ptr<FeatureEvaluator> evaluator = createFeatureEvaluator(features_, ftype_);
    Mat rescaled_image;
    vector<Mat> channels;
    
    for( float scale = minScaleFactor; scale < maxScaleFactor + 0.001; scale += factorStep )
    {
        if(scale < 1.0)
          resize(img, rescaled_image, Size(),scale, scale, INTER_AREA);
        else if (scale > 1.0)
          resize(img, rescaled_image, Size(),scale, scale, INTER_CUBIC);
        else //scale == 1.0
          img.copyTo(rescaled_image);
          
        computeChannels(rescaled_image, channels);
        evaluator->setChannels(channels);
        for( int row = 0; row <= rescaled_image.rows - model_n_rows_; row += slidingStep)
        {
            for( int col = 0; col <= rescaled_image.cols - model_n_cols_;
                col += slidingStep )
            {
                evaluator->setPosition(Size(row, col));
                float value = waldboost_->predict(evaluator);
                if( value > threshold )
                {
                    values.push_back(value);
                    int x = (int)(col / scale);
                    int y = (int)(row / scale);
                    int width = (int)(model_n_cols_ / scale);
                    int height = (int)(model_n_rows_ / scale);
                    objects.push_back(Rect(x, y, width, height));
                }
            }
        }

    }
    
}

void write(FileStorage& fs, String&, const ICFDetector& detector)
{
    detector.write(fs);
}

void read(const FileNode& node, ICFDetector& d,
    const ICFDetector& default_value)
{
    if( node.empty() )
        d = default_value;
    else
        d.read(node);
}

} /* namespace xobjdetect */
} /* namespace cv */
