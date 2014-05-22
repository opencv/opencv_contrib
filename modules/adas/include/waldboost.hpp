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

class Stump
{
public:
    /* Train stump for given data

        data — matrix of feature values, size M x N, one feature per row

        labels — matrix of sample class labels, size 1 x N. Labels can be from
            {-1, +1}

    Returns chosen feature index. Feature enumeration starts from 0
    */
    int train(const cv::Mat_<int>& data, const cv::Mat_<int>& labels);

    /* Predict object class given

        value — feature value. Feature must be the same as chose during training
        stump

    Returns object class from {-1, +1}
    */
    int predict(int value);

private:
    /* Stump decision threshold */
    int threshold_;
    /* Stump polarity, can be from {-1, +1} */
    int polarity_;
    /* Stump decision rule:
        h(value) = polarity * sign(value - threshold)
    */
};

/* Save Stump to FileStorage */
cv::FileStorage& operator<< (cv::FileStorage& out, const Stump& classifier);
/* Load Stump from FileStorage */
cv::FileStorage& operator>> (cv::FileStorage& in, Stump& classifier);

class WaldBoost
{
public:
    /* Initialize WaldBoost cascade with default of specified parameters */
    WaldBoost(const WaldBoostParams& = WaldBoostParams());

    /* Train WaldBoost cascade for given data

        data — matrix of feature values, size M x N, one feature per row

        labels — matrix of sample class labels, size 1 x N. Labels can be from
            {-1, +1}

    Returns feature indices chosen for cascade.
    Feature enumeration starts from 0
    */
    std::vector<int> train(const cv::Mat_<int>& data,
                           const cv::Mat_<int>& labels);

    /* Predict object class given object that can compute object features

       feature_evaluator — object that can compute features by demand

    Returns confidence_value — measure of confidense that object
    is from class +1
    */
    float predict(const ACFFeatureEvaluator& feature_evaluator);

private:
    /* Parameters for cascade training */
    WaldBoostParams params_;
    /* Stumps in cascade */
    std::vector<Stump> stumps_;
    /* Weight for stumps in cascade linear combination */
    std::vector<float> stump_weights_;
    /* Rejection thresholds for linear combination at every stump evaluation */
    std::vector<float> thresholds_;
};

/* Save WaldBoost to FileStorage */
cv::FileStorage& operator<< (cv::FileStorage& out, const WaldBoost& classifier);
/* Load WaldBoost from FileStorage */
cv::FileStorage& operator>> (cv::FileStorage& in, WaldBoost& classifier);