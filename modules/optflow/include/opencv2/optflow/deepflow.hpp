#pragma once
/*
By downloading, copying, installing or using the software you agree to this
license. If you do not agree to this license, do not download, install,
copy or use the software.


                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Copyright (C) 2016, OpenCV Foundation, all rights reserved.
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

#ifndef __OPENCV_OPTFLOW_DEEPFLOW_HPP__
#define __OPENCV_OPTFLOW_DEEPFLOW_HPP__

#include "opencv2/core.hpp"
#include "opencv2/video.hpp"

namespace cv
{
namespace optflow
{

class CV_EXPORTS_W OpticalFlowDeepFlow: public DenseOpticalFlow
{
public:
    OpticalFlowDeepFlow();

    void calc( InputArray I0, InputArray I1, InputOutputArray flow ) CV_OVERRIDE;
    void collectGarbage() CV_OVERRIDE;
public:
    CV_WRAP void setSigma(float val) {sigma = val;}
    CV_WRAP float getSigma() const {return sigma;}
    CV_WRAP void setMinSize(int val) {minSize = val;}
    CV_WRAP int getMinSize() const {return minSize;}
    CV_WRAP void setDownscaleFactor(float val) {downscaleFactor = val;}
    CV_WRAP float getDownscaleFactor() const {return downscaleFactor;}
    CV_WRAP void setFixedPointIterations(int val) {fixedPointIterations = val;}
    CV_WRAP int getFixedPointIterations() const {return fixedPointIterations;}
    CV_WRAP void setSorIterations(int val) {sorIterations = val;}
    CV_WRAP int getSorIterations() const {return sorIterations;}
    CV_WRAP void setAlpha(float val) {alpha = val;}
    CV_WRAP float getAlpha() const {return alpha;}
    CV_WRAP void setDelta(float val) {delta = val;}
    CV_WRAP float getDelta() const {return delta;}
    CV_WRAP void setGamma(float val) {gamma = val;}
    CV_WRAP float getGamma() const {return gamma;}
    CV_WRAP void setOmega(float val) {omega = val;}
    CV_WRAP float getOmega() const {return omega;}
    CV_WRAP void setMaxLayers(int val) {maxLayers = val;}
    CV_WRAP int getMaxLayers() const {return maxLayers;}
    CV_WRAP void setInterpolationType(int val) {interpolationType = val;}
    CV_WRAP int getInterpolationType() const {return interpolationType;}

protected:
    float sigma; // Gaussian smoothing parameter
    int minSize; // minimal dimension of an image in the pyramid
    float downscaleFactor; // scaling factor in the pyramid
    int fixedPointIterations; // during each level of the pyramid
    int sorIterations; // iterations of SOR
    float alpha; // smoothness assumption weight
    float delta; // color constancy weight
    float gamma; // gradient constancy weight
    float omega; // relaxation factor in SOR

    int maxLayers; // max amount of layers in the pyramid
    int interpolationType;

private:
    std::vector<Mat> buildPyramid( const Mat& src );

};

CV_EXPORTS_W Ptr<DenseOpticalFlow> createOptFlow_DeepFlow() { return makePtr<OpticalFlowDeepFlow>(); }

}
}

#endif
