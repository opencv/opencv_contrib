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

/** @brief Read a .flo file

@param path Path to the file to be loaded

The function readOpticalFlow loads a flow field from a file and returns it as a single matrix.
Resulting Mat has a type CV_32FC2 - floating-point, 2-channel. First channel corresponds to the
flow in the horizontal direction (u), second - vertical (v).
 */
CV_EXPORTS_W Mat readOpticalFlow( const String& path );
/** @brief Write a .flo to disk

@param path Path to the file to be written
@param flow Flow field to be stored

The function stores a flow field in a file, returns true on success, false otherwise.
The flow field must be a 2-channel, floating-point matrix (CV_32FC2). First channel corresponds
to the flow in the horizontal direction (u), second - vertical (v).
 */
CV_EXPORTS_W bool writeOpticalFlow( const String& path, InputArray flow );


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

//! @}

} //optflow
}

#include "opencv2/optflow/motempl.hpp"

#endif
