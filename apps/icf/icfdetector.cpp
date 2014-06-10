#include "icfdetector.hpp"
#include "waldboost.hpp"

#include <iostream>

#include <sstream>
using std::ostringstream;

using std::vector;
using std::string;

#include <algorithm>
using std::min;
using std::max;

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

namespace cv
{
namespace adas
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
    int pos_count = 0;
    for( size_t i = 0; i < image_filenames.size(); ++i, ++pos_count )
    {
        Mat img = imread(image_filenames[i]);
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
    for( size_t i = 0; i < image_filenames.size(); ++i, ++neg_count )
    {
        Mat img = imread(image_filenames[i]);
        for( size_t j = 0; j < pos_count / image_filenames.size() + 1; ++j )
        {
            Rect r;
            r.x = rng.uniform(0, img.cols);
            r.width = rng.uniform(r.x + 1, img.cols);
            r.y = rng.uniform(0, img.rows);
            r.height = rng.uniform(r.y + 1, img.rows);

            if( !overlap(r, labelling[i]) )
            {
                sample = img.colRange(r.x, r.width).rowRange(r.y, r.height);
                resize(sample, resized_sample);
                samples.push_back(resized_sample);
                ++neg_count;
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

bool ICFDetector::save(const string& filename)
{
    return true;
}

} /* namespace adas */
} /* namespace cv */
