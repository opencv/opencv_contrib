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

#ifndef __OPENCV_XOBJDETECT_XOBJDETECT_HPP__
#define __OPENCV_XOBJDETECT_XOBJDETECT_HPP__

#include <opencv2/core.hpp>
#include <vector>
#include <string>

namespace cv
{
namespace xobjdetect
{

/* Compute channel pyramid for acf features

    image — image, for which channels should be computed

    channels — output array for computed channels

*/
void computeChannels(InputArray image, OutputArrayOfArrays channels);

class CV_EXPORTS ACFFeatureEvaluator
{
public:
    /* Construct evaluator, set features to evaluate */
    ACFFeatureEvaluator(const std::vector<Point3i>& features);

    /* Set channels for feature evaluation */
    void setChannels(InputArrayOfArrays channels);

    /* Set window position */
    void setPosition(Size position);

    /* Evaluate feature with given index for current channels
        and window position */
    int evaluate(size_t feature_ind) const;

    /* Evaluate all features for current channels and window position

    Returns matrix-column of features
    */
    void evaluateAll(OutputArray feature_values) const;

private:
    /* Features to evaluate */
    std::vector<Point3i> features_;
    /* Channels for feature evaluation */
    std::vector<Mat> channels_;
    /* Channels window position */
    Size position_;
};

/* Generate acf features

    window_size — size of window in which features should be evaluated

    count — number of features to generate.
    Max number of features is min(count, # possible distinct features)

Returns vector of distinct acf features
*/
std::vector<Point3i>
generateFeatures(Size window_size, int count = INT_MAX);


struct CV_EXPORTS WaldBoostParams
{
    int weak_count;
    float alpha;

    WaldBoostParams(): weak_count(100), alpha(0.01)
    {}
};


class CV_EXPORTS Stump
{
public:

    /* Initialize zero stump */
    Stump(): threshold_(0), polarity_(1), pos_value_(1), neg_value_(-1) {}

    /* Initialize stump with given threshold, polarity
        and classification values */
    Stump(int threshold, int polarity, float pos_value, float neg_value):
        threshold_(threshold), polarity_(polarity),
        pos_value_(pos_value), neg_value_(neg_value) {}

    /* Train stump for given data

        data — matrix of feature values, size M x N, one feature per row

        labels — matrix of sample class labels, size 1 x N. Labels can be from
            {-1, +1}

        weights — matrix of sample weights, size 1 x N

    Returns chosen feature index. Feature enumeration starts from 0
    */
    int train(const Mat& data, const Mat& labels, const Mat& weights);

    /* Predict object class given

        value — feature value. Feature must be the same as was chosen
        during training stump

    Returns real value, sign(value) means class
    */
    float predict(int value) const;

private:
    /* Stump decision threshold */
    int threshold_;
    /* Stump polarity, can be from {-1, +1} */
    int polarity_;
    /* Classification values for positive and negative classes  */
    float pos_value_, neg_value_;
};

class CV_EXPORTS WaldBoost
{
public:
    /* Initialize WaldBoost cascade with default of specified parameters */
    WaldBoost(const WaldBoostParams& params = WaldBoostParams());

    /* Train WaldBoost cascade for given data

        data — matrix of feature values, size M x N, one feature per row

        labels — matrix of sample class labels, size 1 x N. Labels can be from
            {-1, +1}

    Returns feature indices chosen for cascade.
    Feature enumeration starts from 0
    */
    std::vector<int> train(const Mat& data,
                           const Mat& labels);

    /* Predict object class given object that can compute object features

       feature_evaluator — object that can compute features by demand

    Returns confidence_value — measure of confidense that object
    is from class +1
    */
    float predict(const Ptr<ACFFeatureEvaluator>& feature_evaluator);

private:
    /* Parameters for cascade training */
    WaldBoostParams params_;
    /* Stumps in cascade */
    std::vector<Stump> stumps_;
    /* Rejection thresholds for linear combination at every stump evaluation */
    std::vector<float> thresholds_;
};

struct CV_EXPORTS ICFDetectorParams
{
    int feature_count;
    int weak_count;
    int model_n_rows;
    int model_n_cols;
    double overlap;

    ICFDetectorParams(): feature_count(UINT_MAX), weak_count(100),
        model_n_rows(40), model_n_cols(40), overlap(0.0)
    {}
};

class CV_EXPORTS ICFDetector
{
public:
    /* Train detector

        image_filenames — filenames of images for training

        labelling — vector of object bounding boxes per every image

        params — parameters for detector training
    */
    void train(const std::vector<std::string>& image_filenames,
               const std::vector<std::vector<cv::Rect> >& labelling,
               ICFDetectorParams params = ICFDetectorParams());

    /* Save detector in file, return true on success, false otherwise */
    bool save(const std::string& filename);
};

} /* namespace xobjdetect */
} /* namespace cv */


#endif /* __OPENCV_XOBJDETECT_XOBJDETECT_HPP__ */
