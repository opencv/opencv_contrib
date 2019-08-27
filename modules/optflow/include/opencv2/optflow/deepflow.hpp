// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef __OPENCV_OPTFLOW_DEEPFLOW_HPP__
#define __OPENCV_OPTFLOW_DEEPFLOW_HPP__

#include "opencv2/core.hpp"
#include "opencv2/video.hpp"

namespace cv
{
namespace optflow
{

//! @addtogroup optflow
//! @{

/** @brief DeepFlow algorithm.
 */
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

/** @brief Creates an instance of PCAFlow
*/
CV_EXPORTS_W Ptr<DenseOpticalFlow> createOptFlow_DeepFlow();

//! @}

}
}

#endif
