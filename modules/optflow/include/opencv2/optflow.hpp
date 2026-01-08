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

#ifndef __OPENCV_OPTFLOW_HPP__
#define __OPENCV_OPTFLOW_HPP__

#include "opencv2/core.hpp"
#include "opencv2/video.hpp"

/**
@defgroup optflow Optical Flow Algorithms

Dense optical flow algorithms compute motion for each point:

- cv::optflow::calcOpticalFlowSF
- cv::optflow::createOptFlow_DeepFlow

Motion templates is alternative technique for detecting motion and computing its direction.
See samples/motempl.py.

- cv::motempl::updateMotionHistory
- cv::motempl::calcMotionGradient
- cv::motempl::calcGlobalOrientation
- cv::motempl::segmentMotion

Functions reading and writing .flo files in "Middlebury" format, see: <http://vision.middlebury.edu/flow/code/flow-code/README.txt>

- cv::optflow::readOpticalFlow
- cv::optflow::writeOpticalFlow

 */

#include "opencv2/optflow/pcaflow.hpp"
#include "opencv2/optflow/sparse_matching_gpc.hpp"
#include "opencv2/optflow/rlofflow.hpp"
namespace cv
{
namespace optflow
{

//! @addtogroup optflow
//! @{

/** @overload */
CV_EXPORTS_W void calcOpticalFlowSF( InputArray from, InputArray to, OutputArray flow,
                                     int layers, int averaging_block_size, int max_flow);

/** @brief Calculate an optical flow using "SimpleFlow" algorithm.

@param from First 8-bit 3-channel image.
@param to Second 8-bit 3-channel image of the same size as prev
@param flow computed flow image that has the same size as prev and type CV_32FC2
@param layers Number of layers
@param averaging_block_size Size of block through which we sum up when calculate cost function
for pixel
@param max_flow maximal flow that we search at each level
@param sigma_dist vector smooth spatial sigma parameter
@param sigma_color vector smooth color sigma parameter
@param postprocess_window window size for postprocess cross bilateral filter
@param sigma_dist_fix spatial sigma for postprocess cross bilateralf filter
@param sigma_color_fix color sigma for postprocess cross bilateral filter
@param occ_thr threshold for detecting occlusions
@param upscale_averaging_radius window size for bilateral upscale operation
@param upscale_sigma_dist spatial sigma for bilateral upscale operation
@param upscale_sigma_color color sigma for bilateral upscale operation
@param speed_up_thr threshold to detect point with irregular flow - where flow should be
recalculated after upscale

See @cite Tao2012 . And site of project - <http://graphics.berkeley.edu/papers/Tao-SAN-2012-05/>.

@note
   -   An example using the simpleFlow algorithm can be found at samples/simpleflow_demo.cpp
 */
CV_EXPORTS_W void calcOpticalFlowSF( InputArray from, InputArray to, OutputArray flow, int layers,
                                     int averaging_block_size, int max_flow,
                                     double sigma_dist, double sigma_color, int postprocess_window,
                                     double sigma_dist_fix, double sigma_color_fix, double occ_thr,
                                     int upscale_averaging_radius, double upscale_sigma_dist,
                                     double upscale_sigma_color, double speed_up_thr );

/** @brief Fast dense optical flow based on PyrLK sparse matches interpolation.

@param from first 8-bit 3-channel or 1-channel image.
@param to  second 8-bit 3-channel or 1-channel image of the same size as from
@param flow computed flow image that has the same size as from and CV_32FC2 type
@param grid_step stride used in sparse match computation. Lower values usually
       result in higher quality but slow down the algorithm.
@param k number of nearest-neighbor matches considered, when fitting a locally affine
       model. Lower values can make the algorithm noticeably faster at the cost of
       some quality degradation.
@param sigma parameter defining how fast the weights decrease in the locally-weighted affine
       fitting. Higher values can help preserve fine details, lower values can help to get rid
       of the noise in the output flow.
@param use_post_proc defines whether the ximgproc::fastGlobalSmootherFilter() is used
       for post-processing after interpolation
@param fgs_lambda see the respective parameter of the ximgproc::fastGlobalSmootherFilter()
@param fgs_sigma  see the respective parameter of the ximgproc::fastGlobalSmootherFilter()
 */
CV_EXPORTS_W void calcOpticalFlowSparseToDense ( InputArray from, InputArray to, OutputArray flow,
                                                 int grid_step = 8, int k = 128, float sigma = 0.05f,
                                                 bool use_post_proc = true, float fgs_lambda = 500.0f,
                                                 float fgs_sigma = 1.5f );

/** @brief DeepFlow optical flow algorithm implementation.

The class implements the DeepFlow optical flow algorithm described in @cite Weinzaepfel2013 . See
also <http://lear.inrialpes.fr/src/deepmatching/> .
Parameters - class fields - that may be modified after creating a class instance:
-   member float alpha
Smoothness assumption weight
-   member float delta
Color constancy assumption weight
-   member float gamma
Gradient constancy weight
-   member float sigma
Gaussian smoothing parameter
-   member int minSize
Minimal dimension of an image in the pyramid (next, smaller images in the pyramid are generated
until one of the dimensions reaches this size)
-   member float downscaleFactor
Scaling factor in the image pyramid (must be \< 1)
-   member int fixedPointIterations
How many iterations on each level of the pyramid
-   member int sorIterations
Iterations of Succesive Over-Relaxation (solver)
-   member float omega
Relaxation factor in SOR
 */
CV_EXPORTS_W Ptr<DenseOpticalFlow> createOptFlow_DeepFlow();

//! Additional interface to the SimpleFlow algorithm - calcOpticalFlowSF()
CV_EXPORTS_W Ptr<DenseOpticalFlow> createOptFlow_SimpleFlow();

//! Additional interface to the Farneback's algorithm - calcOpticalFlowFarneback()
CV_EXPORTS_W Ptr<DenseOpticalFlow> createOptFlow_Farneback();

//! Additional interface to the SparseToDenseFlow algorithm - calcOpticalFlowSparseToDense()
CV_EXPORTS_W Ptr<DenseOpticalFlow> createOptFlow_SparseToDense();

/** @brief "Dual TV L1" Optical Flow Algorithm.

The class implements the "Dual TV L1" optical flow algorithm described in @cite Zach2007 and
@cite Javier2012 .
Here are important members of the class that control the algorithm, which you can set after
constructing the class instance:

-   member double tau
    Time step of the numerical scheme.

-   member double lambda
    Weight parameter for the data term, attachment parameter. This is the most relevant
    parameter, which determines the smoothness of the output. The smaller this parameter is,
    the smoother the solutions we obtain. It depends on the range of motions of the images, so
    its value should be adapted to each image sequence.

-   member double theta
    Weight parameter for (u - v)\^2, tightness parameter. It serves as a link between the
    attachment and the regularization terms. In theory, it should have a small value in order
    to maintain both parts in correspondence. The method is stable for a large range of values
    of this parameter.

-   member int nscales
    Number of scales used to create the pyramid of images.

-   member int warps
    Number of warpings per scale. Represents the number of times that I1(x+u0) and grad(
    I1(x+u0) ) are computed per scale. This is a parameter that assures the stability of the
    method. It also affects the running time, so it is a compromise between speed and
    accuracy.

-   member double epsilon
    Stopping criterion threshold used in the numerical scheme, which is a trade-off between
    precision and running time. A small value will yield more accurate solutions at the
    expense of a slower convergence.

-   member int iterations
    Stopping criterion iterations number used in the numerical scheme.

C. Zach, T. Pock and H. Bischof, "A Duality Based Approach for Realtime TV-L1 Optical Flow".
Javier Sanchez, Enric Meinhardt-Llopis and Gabriele Facciolo. "TV-L1 Optical Flow Estimation".
*/
class CV_EXPORTS_W DualTVL1OpticalFlow : public DenseOpticalFlow
{
public:
    //! @brief Time step of the numerical scheme
    /** @see setTau */
    CV_WRAP virtual double getTau() const = 0;
    /** @copybrief getTau @see getTau */
    CV_WRAP virtual void setTau(double val) = 0;
    //! @brief Weight parameter for the data term, attachment parameter
    /** @see setLambda */
    CV_WRAP virtual double getLambda() const = 0;
    /** @copybrief getLambda @see getLambda */
    CV_WRAP virtual void setLambda(double val) = 0;
    //! @brief Weight parameter for (u - v)^2, tightness parameter
    /** @see setTheta */
    CV_WRAP virtual double getTheta() const = 0;
    /** @copybrief getTheta @see getTheta */
    CV_WRAP virtual void setTheta(double val) = 0;
    //! @brief coefficient for additional illumination variation term
    /** @see setGamma */
    CV_WRAP virtual double getGamma() const = 0;
    /** @copybrief getGamma @see getGamma */
    CV_WRAP virtual void setGamma(double val) = 0;
    //! @brief Number of scales used to create the pyramid of images
    /** @see setScalesNumber */
    CV_WRAP virtual int getScalesNumber() const = 0;
    /** @copybrief getScalesNumber @see getScalesNumber */
    CV_WRAP virtual void setScalesNumber(int val) = 0;
    //! @brief Number of warpings per scale
    /** @see setWarpingsNumber */
    CV_WRAP virtual int getWarpingsNumber() const = 0;
    /** @copybrief getWarpingsNumber @see getWarpingsNumber */
    CV_WRAP virtual void setWarpingsNumber(int val) = 0;
    //! @brief Stopping criterion threshold used in the numerical scheme, which is a trade-off between precision and running time
    /** @see setEpsilon */
    CV_WRAP virtual double getEpsilon() const = 0;
    /** @copybrief getEpsilon @see getEpsilon */
    CV_WRAP virtual void setEpsilon(double val) = 0;
    //! @brief Inner iterations (between outlier filtering) used in the numerical scheme
    /** @see setInnerIterations */
    CV_WRAP virtual int getInnerIterations() const = 0;
    /** @copybrief getInnerIterations @see getInnerIterations */
    CV_WRAP virtual void setInnerIterations(int val) = 0;
    //! @brief Outer iterations (number of inner loops) used in the numerical scheme
    /** @see setOuterIterations */
    CV_WRAP virtual int getOuterIterations() const = 0;
    /** @copybrief getOuterIterations @see getOuterIterations */
    CV_WRAP virtual void setOuterIterations(int val) = 0;
    //! @brief Use initial flow
    /** @see setUseInitialFlow */
    CV_WRAP virtual bool getUseInitialFlow() const = 0;
    /** @copybrief getUseInitialFlow @see getUseInitialFlow */
    CV_WRAP virtual void setUseInitialFlow(bool val) = 0;
    //! @brief Step between scales (<1)
    /** @see setScaleStep */
    CV_WRAP virtual double getScaleStep() const = 0;
    /** @copybrief getScaleStep @see getScaleStep */
    CV_WRAP virtual void setScaleStep(double val) = 0;
    //! @brief Median filter kernel size (1 = no filter) (3 or 5)
    /** @see setMedianFiltering */
    CV_WRAP virtual int getMedianFiltering() const = 0;
    /** @copybrief getMedianFiltering @see getMedianFiltering */
    CV_WRAP virtual void setMedianFiltering(int val) = 0;

    /** @brief Creates instance of cv::DualTVL1OpticalFlow*/
    CV_WRAP static Ptr<DualTVL1OpticalFlow> create(
                                            double tau = 0.25,
                                            double lambda = 0.15,
                                            double theta = 0.3,
                                            int nscales = 5,
                                            int warps = 5,
                                            double epsilon = 0.01,
                                            int innnerIterations = 30,
                                            int outerIterations = 10,
                                            double scaleStep = 0.8,
                                            double gamma = 0.0,
                                            int medianFiltering = 5,
                                            bool useInitialFlow = false);
};

/** @brief Creates instance of cv::DenseOpticalFlow
*/
CV_EXPORTS_W Ptr<DualTVL1OpticalFlow> createOptFlow_DualTVL1();

//! @}

} //optflow
}

#include "opencv2/optflow/motempl.hpp"

#endif
