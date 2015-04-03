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
#include <opencv2/highgui.hpp>
#include <vector>
#include <string>

/** @defgroup xobjdetect Extended object detection
*/

namespace cv
{
namespace xobjdetect
{

//! @addtogroup xobjdetect
//! @{

/** @brief Compute channels for integral channel features evaluation

@param image image for which channels should be computed
@param channels output array for computed channels
 */
CV_EXPORTS void computeChannels(InputArray image, std::vector<Mat>& channels);

/** @brief Feature evaluation interface
 */
class CV_EXPORTS FeatureEvaluator : public Algorithm
{
public:
    /** @brief Set channels for feature evaluation

    @param channels array of channels to be set
     */
    virtual void setChannels(InputArrayOfArrays channels) = 0;

    /** @brief Set window position to sample features with shift. By default position is (0, 0).

    @param position position to be set
     */
    virtual void setPosition(Size position) = 0;

    /** @brief Evaluate feature value with given index for current channels and window position.

    @param feature_ind index of feature to be evaluated
     */
    virtual int evaluate(size_t feature_ind) const = 0;

    /** @brief Evaluate all features for current channels and window position.

    @param feature_values matrix-column of evaluated feature values
     */
    virtual void evaluateAll(OutputArray feature_values) const = 0;

    virtual void assertChannels() = 0;
};

/** @brief Construct feature evaluator.

@param features features for evaluation
@param type feature type. Can be "icf" or "acf"
 */
CV_EXPORTS Ptr<FeatureEvaluator>
createFeatureEvaluator(const std::vector<std::vector<int> >& features,
                       const std::string& type);

/** @brief Generate integral features. Returns vector of features.

@param window_size size of window in which features should be evaluated
@param type feature type. Can be "icf" or "acf"
@param count number of features to generate.
@param channel_count number of feature channels
 */
std::vector<std::vector<int> >
generateFeatures(Size window_size, const std::string& type,
                 int count = INT_MAX, int channel_count = 10);

//sort in-place of columns of the input matrix
void sort_columns_without_copy(Mat& m, Mat indices = Mat());

/** @brief Parameters for WaldBoost. weak_count — number of weak learners, alpha — cascade thresholding param.
 */
struct CV_EXPORTS WaldBoostParams
{
    int weak_count;
    float alpha;

    WaldBoostParams(): weak_count(100), alpha(0.02f)
    {}
};

/** @brief WaldBoost object detector from @cite Sochman05 .
*/
class CV_EXPORTS WaldBoost : public Algorithm
{
public:
    /** @brief Train WaldBoost cascade for given data.

    Returns feature indices chosen for cascade. Feature enumeration starts from 0.
    @param data matrix of feature values, size M x N, one feature per row
    @param labels matrix of samples class labels, size 1 x N. Labels can be from {-1, +1}
    @param use_fast_log
     */
    virtual std::vector<int> train(Mat& data,
                                   const Mat& labels, bool use_fast_log=false) = 0;

    /** @brief Predict objects class given object that can compute object features.

    Returns unnormed confidence value — measure of confidence that object is from class +1.
    @param feature_evaluator object that can compute features by demand
     */
    virtual float predict(
        const Ptr<FeatureEvaluator>& feature_evaluator) const = 0;
};

/** @brief Construct WaldBoost object.
 */
CV_EXPORTS Ptr<WaldBoost>
createWaldBoost(const WaldBoostParams& params = WaldBoostParams());

/** @brief Params for ICFDetector training.
 */
struct CV_EXPORTS ICFDetectorParams
{
    int feature_count;
    int weak_count;
    int model_n_rows;
    int model_n_cols;
    int bg_per_image;
    std::string features_type;
    float alpha;
    bool is_grayscale;
    bool use_fast_log;

    ICFDetectorParams(): feature_count(UINT_MAX), weak_count(100),
        model_n_rows(56), model_n_cols(56), bg_per_image(5), alpha(0.02f), is_grayscale(false), use_fast_log(false)
    {}
};

/** @brief Integral Channel Features from @cite Dollar09 .
*/
class CV_EXPORTS ICFDetector
{
public:

    ICFDetector(): waldboost_(), features_(), ftype_() {}

    /** @brief Train detector.

    @param pos_filenames path to folder with images of objects (wildcards like /my/path/\*.png are allowed)
    @param bg_filenames path to folder with background images
    @param params parameters for detector training
     */
    void train(const std::vector<String>& pos_filenames,
               const std::vector<String>& bg_filenames,
               ICFDetectorParams params = ICFDetectorParams());

    /** @brief Detect objects on image.
    @param image image for detection
    @param objects output array of bounding boxes
    @param scaleFactor scale between layers in detection pyramid
    @param minSize min size of objects in pixels
    @param maxSize max size of objects in pixels
    @param threshold
    @param slidingStep sliding window step
    @param values output vector with values of positive samples
     */
    void detect(const Mat& image, std::vector<Rect>& objects,
        float scaleFactor, Size minSize, Size maxSize, float threshold, int slidingStep, std::vector<float>& values);
    
    /** @brief Detect objects on image.
    @param img image for detection
    @param objects output array of bounding boxes
    @param minScaleFactor min factor by which the image will be resized
    @param maxScaleFactor max factor by which the image will be resized
    @param factorStep scaling factor is incremented each pyramid layer according to this parameter
    @param threshold
    @param slidingStep sliding window step
    @param values output vector with values of positive samples
     */
    void detect(const Mat& img, std::vector<Rect>& objects, float minScaleFactor, float maxScaleFactor, float factorStep, float threshold, int slidingStep, std::vector<float>& values);

    /** @brief Write detector to FileStorage.
    @param fs FileStorage for output
     */
    void write(FileStorage &fs) const;

    /** @brief Write ICFDetector to FileNode
    @param node FileNode for reading
     */
    void read(const FileNode &node);

private:
    Ptr<WaldBoost> waldboost_;
    std::vector<std::vector<int> > features_;
    int model_n_rows_;
    int model_n_cols_;
    std::string ftype_;
};

CV_EXPORTS void write(FileStorage& fs, String&, const ICFDetector& detector);

CV_EXPORTS void read(const FileNode& node, ICFDetector& d,
    const ICFDetector& default_value = ICFDetector());

//! @}

} /* namespace xobjdetect */
} /* namespace cv */

#endif /* __OPENCV_XOBJDETECT_XOBJDETECT_HPP__ */
