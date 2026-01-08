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

#ifndef __OPENCV_CONDITIONING_HPP__
#define __OPENCV_CONDITIONING_HPP__

#include <opencv2/core.hpp>

namespace cv
{
namespace sfm
{

//! @addtogroup conditioning
//! @{

/** Point conditioning (non isotropic).
  @param points Input vector of N-dimensional points.
  @param T Output 3x3 transformation matrix.

  Computes the transformation matrix such that the two principal moments of the set of points are equal to unity,
  forming an approximately symmetric circular cloud of points of radius 1 about the origin.\n
  Reference: @cite HartleyZ00 4.4.4 pag.109
*/
CV_EXPORTS_W
void
preconditionerFromPoints( InputArray points,
                          OutputArray T );

/** @brief Point conditioning (isotropic).
  @param points Input vector of N-dimensional points.
  @param T Output 3x3 transformation matrix.

  Computes the transformation matrix such that each coordinate direction will be scaled equally,
  bringing the centroid to the origin with an average centroid \f$(1,1,1)^T\f$.\n
  Reference: @cite HartleyZ00 4.4.4 pag.107.
*/
CV_EXPORTS_W
void
isotropicPreconditionerFromPoints( InputArray points,
                                   OutputArray T );

/** @brief Apply Transformation to points.
  @param points Input vector of N-dimensional points.
  @param T Input 3x3 transformation matrix such that \f$x = T*X\f$, where \f$X\f$ are the points to transform and \f$x\f$ the transformed points.
  @param transformed_points Output vector of N-dimensional transformed points.
*/
CV_EXPORTS_W
void
applyTransformationToPoints( InputArray points,
                             InputArray T,
                             OutputArray transformed_points );

/** @brief This function normalizes points (non isotropic).
  @param points Input vector of N-dimensional points.
  @param normalized_points Output vector of the same N-dimensional points but with mean 0 and average norm \f$\sqrt{2}\f$.
  @param T Output 3x3 transform matrix such that \f$x = T*X\f$, where \f$X\f$ are the points to normalize and \f$x\f$ the normalized points.

  Internally calls @ref preconditionerFromPoints in order to get the scaling matrix before applying @ref applyTransformationToPoints.
  This operation is an essential step before applying the DLT algorithm in order to consider the result as optimal.\n
  Reference: @cite HartleyZ00 4.4.4 pag.109
*/
CV_EXPORTS_W
void
normalizePoints( InputArray points,
                 OutputArray normalized_points,
                 OutputArray T );

/** @brief This function normalizes points. (isotropic).
  @param points Input vector of N-dimensional points.
  @param normalized_points Output vector of the same N-dimensional points but with mean 0 and average norm \f$\sqrt{2}\f$.
  @param T Output 3x3 transform matrix such that \f$x = T*X\f$, where \f$X\f$ are the points to normalize and \f$x\f$ the normalized points.

  Internally calls @ref preconditionerFromPoints in order to get the scaling matrix before applying @ref applyTransformationToPoints.
  This operation is an essential step before applying the DLT algorithm in order to consider the result as optimal.\n
  Reference: @cite HartleyZ00 4.4.4 pag.107.
*/
CV_EXPORTS_W
void
normalizeIsotropicPoints( InputArray points,
                          OutputArray normalized_points,
                          OutputArray T );

//! @} sfm

} /* namespace sfm */
} /* namespace cv */

#endif

/* End of file. */