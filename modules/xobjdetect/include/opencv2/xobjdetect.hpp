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
CV_EXPORTS void computeChannels(InputArray image, std::vector<Mat>& channels);

class CV_EXPORTS FeatureEvaluator : public Algorithm
{
public:
    /* Set channels for feature evaluation */
    virtual void setChannels(InputArrayOfArrays channels) = 0;

    /* Set window position */
    virtual void setPosition(Size position) = 0;

    /* Evaluate feature with given index for current channels
        and window position */
    virtual int evaluate(size_t feature_ind) const = 0;

    /* Evaluate all features for current channels and window position

    Returns matrix-column of features
    */
    virtual void evaluateAll(OutputArray feature_values) const = 0;

    virtual void assertChannels() = 0;
};

/* Construct feature evaluator, set features to evaluate
   type can "icf" or "acf" */
CV_EXPORTS Ptr<FeatureEvaluator>
createFeatureEvaluator(const std::vector<std::vector<int> >& features,
                       const std::string& type);

/* Generate acf features

    window_size — size of window in which features should be evaluated

    type — type of features, can be "icf" or "acf"
    count — number of features to generate.
    Max number of features is min(count, # possible distinct features)

Returns vector of distinct acf features
*/
std::vector<std::vector<int> >
generateFeatures(Size window_size, const std::string& type,
                 int count = INT_MAX, int channel_count = 10);


struct CV_EXPORTS WaldBoostParams
{
    int weak_count;
    float alpha;

    WaldBoostParams(): weak_count(100), alpha(0.02f)
    {}
};



class CV_EXPORTS WaldBoost : public Algorithm
{
public:
    /* Train WaldBoost cascade for given data

        data — matrix of feature values, size M x N, one feature per row

        labels — matrix of sample class labels, size 1 x N. Labels can be from
            {-1, +1}

    Returns feature indices chosen for cascade.
    Feature enumeration starts from 0
    */
    virtual std::vector<int> train(const Mat& /*data*/,
                                   const Mat& /*labels*/) = 0;

    /* Predict object class given object that can compute object features

       feature_evaluator — object that can compute features by demand

    Returns confidence_value — measure of confidense that object
    is from class +1
    */
    virtual float predict(
        const Ptr<FeatureEvaluator>& /*feature_evaluator*/) const = 0;

    /* Write WaldBoost to FileStorage */
    virtual void write(FileStorage& /*fs*/) const = 0;

    /* Read WaldBoost */
    virtual void read(const FileNode& /*node*/) = 0;
};

CV_EXPORTS Ptr<WaldBoost>
createWaldBoost(const WaldBoostParams& params = WaldBoostParams());

struct CV_EXPORTS ICFDetectorParams
{
    int feature_count;
    int weak_count;
    int model_n_rows;
    int model_n_cols;
    int bg_per_image;

    ICFDetectorParams(): feature_count(UINT_MAX), weak_count(100),
        model_n_rows(56), model_n_cols(56), bg_per_image(5)
    {}
};

class CV_EXPORTS ICFDetector
{
public:

    ICFDetector(): waldboost_(), features_() {}

    /* Train detector

        pos_path — path to folder with images of objects

        bg_path — path to folder with background images

        params — parameters for detector training
    */
    void train(const String& pos_path,
               const String& bg_path,
               ICFDetectorParams params = ICFDetectorParams());

    /* Detect object on image

        image — image for detection

        object — output array of bounding boxes

        scaleFactor — scale between layers in detection pyramid

        minSize — min size of objects in pixels

        maxSize — max size of objects in pixels
    */
    void detect(const Mat& image, std::vector<Rect>& objects,
        float scaleFactor, Size minSize, Size maxSize, float threshold);

    /* Write detector to FileStorage */
    void write(FileStorage &fs) const;

    /* Read detector */
    void read(const FileNode &node);

private:
    Ptr<WaldBoost> waldboost_;
    std::vector<std::vector<int> > features_;
    int model_n_rows_;
    int model_n_cols_;
};

CV_EXPORTS void write(FileStorage& fs, String&, const ICFDetector& detector);

CV_EXPORTS void read(const FileNode& node, ICFDetector& d,
    const ICFDetector& default_value = ICFDetector());

} /* namespace xobjdetect */
} /* namespace cv */

#endif /* __OPENCV_XOBJDETECT_XOBJDETECT_HPP__ */
