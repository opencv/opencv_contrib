/*M///////////////////////////////////////////////////////////////////////////////////////
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
 // Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
 // Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

#include "precomp.hpp"

#include "opencv2/optflow/deepflow.hpp"

namespace cv
{
namespace optflow
{

class CV_EXPORTS_W OpticalFlowDeepFlowImpl: public OpticalFlowDeepFlow
{
public:
    OpticalFlowDeepFlowImpl(float sigma = 0.6f,
                            int minSize = 25,
                            float downscaleFactor = 0.95f,
                            int fixedPointIterations = 5,
                            int sorIterations = 25,
                            float alpha = 1.0f,
                            float delta = 0.5f,
                            float gamma = 5.0f,
                            float omega = 1.6f,
                            int maxLayers = 200,
                            int interpolationType = INTER_LINEAR);

    void calc( InputArray I0, InputArray I1, InputOutputArray flow ) CV_OVERRIDE;
    void collectGarbage() CV_OVERRIDE;
public:
    CV_WRAP virtual void setSigma(float val) CV_OVERRIDE {sigma = val;}
    CV_WRAP virtual float getSigma() const CV_OVERRIDE {return sigma;}
    CV_WRAP virtual void setMinSize(int val) CV_OVERRIDE {minSize = val;}
    CV_WRAP virtual int getMinSize() const CV_OVERRIDE {return minSize;}
    CV_WRAP virtual void setDownscaleFactor(float val) CV_OVERRIDE {downscaleFactor = val;}
    CV_WRAP virtual float getDownscaleFactor() const CV_OVERRIDE {return downscaleFactor;}
    CV_WRAP virtual void setFixedPointIterations(int val) CV_OVERRIDE {fixedPointIterations = val;}
    CV_WRAP virtual int getFixedPointIterations() const CV_OVERRIDE {return fixedPointIterations;}
    CV_WRAP virtual void setSorIterations(int val) CV_OVERRIDE {sorIterations = val;}
    CV_WRAP virtual int getSorIterations() const CV_OVERRIDE {return sorIterations;}
    CV_WRAP virtual void setAlpha(float val) CV_OVERRIDE {alpha = val;}
    CV_WRAP virtual float getAlpha() const CV_OVERRIDE {return alpha;}
    CV_WRAP virtual void setDelta(float val) CV_OVERRIDE {delta = val;}
    CV_WRAP virtual float getDelta() const CV_OVERRIDE {return delta;}
    CV_WRAP virtual void setGamma(float val) CV_OVERRIDE {gamma = val;}
    CV_WRAP virtual float getGamma() const CV_OVERRIDE {return gamma;}
    CV_WRAP virtual void setOmega(float val) CV_OVERRIDE {omega = val;}
    CV_WRAP virtual float getOmega() const CV_OVERRIDE {return omega;}
    CV_WRAP virtual void setMaxLayers(int val) CV_OVERRIDE {maxLayers = val;}
    CV_WRAP virtual int getMaxLayers() const CV_OVERRIDE {return maxLayers;}
    CV_WRAP virtual void setInterpolationType(int val) CV_OVERRIDE {interpolationType = val;}
    CV_WRAP virtual int getInterpolationType() const CV_OVERRIDE {return interpolationType;}
public:
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


OpticalFlowDeepFlowImpl::OpticalFlowDeepFlowImpl(float sigma,
                                                 int minSize,
                                                 float downscaleFactor,
                                                 int fixedPointIterations,
                                                 int sorIterations,
                                                 float alpha,
                                                 float delta,
                                                 float gamma,
                                                 float omega,
                                                 int maxLayers,
                                                 int interpolationType)
                       :sigma(sigma),minSize(minSize),downscaleFactor(downscaleFactor),fixedPointIterations(fixedPointIterations),
                        sorIterations(sorIterations),alpha(alpha),delta(delta),gamma(gamma),omega(omega),
                        maxLayers(maxLayers),interpolationType(interpolationType)
{
}

std::vector<Mat> OpticalFlowDeepFlowImpl::buildPyramid( const Mat& src )
{
    std::vector<Mat> pyramid;
    pyramid.push_back(src);
    Mat prev = pyramid[0];
    for( int i = 0; i < this->maxLayers; ++i)
    {
        Mat next; //TODO: filtering at each level?
        Size nextSize((int) (prev.cols * downscaleFactor + 0.5f),
                        (int) (prev.rows * downscaleFactor + 0.5f));
        if( nextSize.height <= minSize || nextSize.width <= minSize)
            break;
        resize(prev, next,
                nextSize, 0, 0,
                interpolationType);
        pyramid.push_back(next);
        prev = next;
    }
    return pyramid;
}

void OpticalFlowDeepFlowImpl::calc( InputArray _I0, InputArray _I1, InputOutputArray _flow )
{
    Mat I0temp = _I0.getMat();
    Mat I1temp = _I1.getMat();

    CV_Assert(I0temp.size() == I1temp.size());
    CV_Assert(I0temp.type() == I1temp.type());
    CV_Assert(I0temp.channels() == 1);
    // TODO: currently only grayscale - data term could be computed in color version as well...

    Mat I0, I1;

    I0temp.convertTo(I0, CV_32F);
    I1temp.convertTo(I1, CV_32F);

    _flow.create(I0.size(), CV_32FC2);
    Mat W = _flow.getMat(); // if any data present - will be discarded

    // pre-smooth images
    int kernelLen = ((int)floor(3 * sigma) * 2) + 1;
    Size kernelSize(kernelLen, kernelLen);
    GaussianBlur(I0, I0, kernelSize, sigma);
    GaussianBlur(I1, I1, kernelSize, sigma);
    // build down-sized pyramids
    std::vector<Mat> pyramid_I0 = buildPyramid(I0);
    std::vector<Mat> pyramid_I1 = buildPyramid(I1);
    int levelCount = (int) pyramid_I0.size();

    // initialize the first version of flow estimate to zeros
    Size smallestSize = pyramid_I0[levelCount - 1].size();
    W = Mat::zeros(smallestSize, CV_32FC2);

    for ( int level = levelCount - 1; level >= 0; --level )
    { //iterate through  all levels, beginning with the most coarse
        Ptr<VariationalRefinement> var = VariationalRefinement::create();

        var->setAlpha(4 * alpha);
        var->setDelta(delta / 3);
        var->setGamma(gamma / 3);
        var->setFixedPointIterations(fixedPointIterations);
        var->setSorIterations(sorIterations);
        var->setOmega(omega);

        var->calc(pyramid_I0[level], pyramid_I1[level], W);
        if ( level > 0 ) //not the last level
        {
            Mat temp;
            Size newSize = pyramid_I0[level - 1].size();
            resize(W, temp, newSize, 0, 0, interpolationType); //resize calculated flow
            W = temp * (1.0f / downscaleFactor); //scale values
        }
    }
    W.copyTo(_flow);
}

void OpticalFlowDeepFlowImpl::collectGarbage() {}

Ptr<DenseOpticalFlow> createOptFlow_DeepFlow() { return OpticalFlowDeepFlow::create(); }

Ptr<OpticalFlowDeepFlow> OpticalFlowDeepFlow::create(float sigma,
                                                     int minSize,
                                                     float downscaleFactor,
                                                     int fixedPointIterations,
                                                     int sorIterations,
                                                     float alpha,
                                                     float delta,
                                                     float gamma,
                                                     float omega,
                                                     int maxLayers,
                                                     int interpolationType)
{
  Ptr<OpticalFlowDeepFlow> result = makePtr<OpticalFlowDeepFlowImpl>(sigma, minSize, downscaleFactor, fixedPointIterations, sorIterations,
                                                                     alpha, delta, gamma, omega, maxLayers, interpolationType);
  return result;
}

}//optflow
}//cv
