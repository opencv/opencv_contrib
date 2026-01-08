/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2009, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef __OPENCV_SFM_ROBUST_HPP__
#define __OPENCV_SFM_ROBUST_HPP__

#ifdef __cplusplus

#include <opencv2/core.hpp>

namespace cv
{
namespace sfm
{

//! @addtogroup robust
//! @{

/** @brief Estimate robustly the fundamental matrix between two dataset of 2D point (image coords space).
  @param x1 Input 2xN Array of 2D points in view 1.
  @param x2 Input 2xN Array of 2D points in view 2.
  @param max_error maximum error (in pixels).
  @param F Output 3x3 fundamental matrix such that \f$x_2^T F x_1=0\f$.
  @param inliers Output 1xN vector that contains the indexes of the detected inliers.
  @param outliers_probability outliers probability (in ]0,1[).
         The number of iterations is controlled using the following equation:
         \f$k = \frac{log(1-p)}{log(1.0 - w^n )}\f$ where \f$k\f$, \f$w\f$ and \f$n\f$ are the number of
         iterations, the inliers ratio and minimun number of selected independent samples.
         The more this value is high, the less the function selects ramdom samples.

The fundamental solver relies on the 8 point solution. Returns the best error (in pixels), associated to the solution F.
 */
CV_EXPORTS_W
double
fundamentalFromCorrespondences8PointRobust( InputArray x1,
                                            InputArray x2,
                                            double max_error,
                                            OutputArray F,
                                            OutputArray inliers,
                                            double outliers_probability = 1e-2 );

/** @brief Estimate robustly the fundamental matrix between two dataset of 2D point (image coords space).
  @param x1 Input 2xN Array of 2D points in view 1.
  @param x2 Input 2xN Array of 2D points in view 2.
  @param max_error maximum error (in pixels).
  @param F Output 3x3 fundamental matrix such that \f$x_2^T F x_1=0\f$.
  @param inliers Output 1xN vector that contains the indexes of the detected inliers.
  @param outliers_probability outliers probability (in ]0,1[).
         The number of iterations is controlled using the following equation:
         \f$k = \frac{log(1-p)}{log(1.0 - w^n )}\f$ where \f$k\f$, \f$w\f$ and \f$n\f$ are the number of
         iterations, the inliers ratio and minimun number of selected independent samples.
         The more this value is high, the less the function selects ramdom samples.

The fundamental solver relies on the 7 point solution. Returns the best error (in pixels), associated to the solution F.
 */
CV_EXPORTS_W
double
fundamentalFromCorrespondences7PointRobust( InputArray x1,
                                            InputArray x2,
                                            double max_error,
                                            OutputArray F,
                                            OutputArray inliers,
                                            double outliers_probability = 1e-2 );

//! @} sfm

} /* namespace cv */
} /* namespace sfm */

#endif /* __cplusplus */

#endif

/* End of file. */