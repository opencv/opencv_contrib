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
    CV_WRAP virtual void setSigma(float val) = 0;
    CV_WRAP virtual float getSigma() const = 0;
    CV_WRAP virtual void setMinSize(int val) = 0;
    CV_WRAP virtual int getMinSize() const = 0;
    CV_WRAP virtual void setDownscaleFactor(float val) = 0;
    CV_WRAP virtual float getDownscaleFactor() const = 0;
    CV_WRAP virtual void setFixedPointIterations(int val) = 0;
    CV_WRAP virtual int getFixedPointIterations() const = 0;
    CV_WRAP virtual void setSorIterations(int val) = 0;
    CV_WRAP virtual int getSorIterations() const = 0;
    CV_WRAP virtual void setAlpha(float val) = 0;
    CV_WRAP virtual float getAlpha() const = 0;
    CV_WRAP virtual void setDelta(float val) = 0;
    CV_WRAP virtual float getDelta() const = 0;
    CV_WRAP virtual void setGamma(float val) = 0;
    CV_WRAP virtual float getGamma() const = 0;
    CV_WRAP virtual void setOmega(float val) = 0;
    CV_WRAP virtual float getOmega() const = 0;
    CV_WRAP virtual void setMaxLayers(int val) = 0;
    CV_WRAP virtual int getMaxLayers() const = 0;
    CV_WRAP virtual void setInterpolationType(int val) = 0;
    CV_WRAP virtual int getInterpolationType() const = 0;

    /** @brief Creates instance of cv::optflow::OpticalFlowDeepFlow*/
    CV_WRAP static Ptr<OpticalFlowDeepFlow> create(
                                            float sigma = 0.6f,
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
};

//! @}

}
}

#endif
