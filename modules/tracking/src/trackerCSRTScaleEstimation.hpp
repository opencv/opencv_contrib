/*///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                           License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2013, OpenCV Foundation, all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/

#ifndef OPENCV_TRACKER_CSRT_SCALE_ESTIMATION
#define OPENCV_TRACKER_CSRT_SCALE_ESTIMATION

#include "precomp.hpp"

namespace cv
{

class DSST {
public:
    DSST() {};
    DSST(const Mat &image, Rect2f bounding_box, Size2f template_size, int numberOfScales,
            float scaleStep, float maxModelArea, float sigmaFactor, float scaleLearnRate);
    ~DSST();
    void update(const Mat &image, const Point2f objectCenter);
    float getScale(const Mat &image, const Point2f objecCenter);
private:
    Mat get_scale_features( Mat img, Point2f pos, Size2f base_target_sz, float current_scale,
            std::vector<float> &scale_factors, Mat scale_window, Size scale_model_sz);

    Size scale_model_sz;
    Mat ys;
    Mat ysf;
    Mat scale_window;
    std::vector<float> scale_factors;
    Mat sf_num;
    Mat sf_den;
    float scale_sigma;
    float min_scale_factor;
    float max_scale_factor;
    float current_scale_factor;
    int scales_count;
    float scale_step;
    float max_model_area;
    float sigma_factor;
    float learn_rate;

    Size original_targ_sz;
};

} /* namespace cv */

#endif
