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

#ifndef __OPENCV_PROJECTION_HPP__
#define __OPENCV_PROJECTION_HPP__

#include <opencv2/core.hpp>

namespace cv
{
namespace sfm
{

//! @addtogroup projection
//! @{

/** @brief Converts point coordinates from homogeneous to euclidean pixel coordinates. E.g., ((x,y,z)->(x/z, y/z))
  @param src Input vector of N-dimensional points.
  @param dst Output vector of N-1-dimensional points.
*/
CV_EXPORTS_W
void
homogeneousToEuclidean(InputArray src, OutputArray dst);

/** @brief Converts points from Euclidean to homogeneous space. E.g., ((x,y)->(x,y,1))
  @param src Input vector of N-dimensional points.
  @param dst Output vector of N+1-dimensional points.
*/
CV_EXPORTS_W
void
euclideanToHomogeneous(InputArray src, OutputArray dst);

/** @brief Get projection matrix P from K, R and t.
  @param K Input 3x3 camera matrix \f$K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$.
  @param R Input 3x3 rotation matrix.
  @param t Input 3x1 translation vector.
  @param P Output 3x4 projection matrix.

  This function estimate the projection matrix by solving the following equation: \f$P = K * [R|t]\f$

 */
CV_EXPORTS_W
void
projectionFromKRt(InputArray K, InputArray R, InputArray t, OutputArray P);

/** @brief Get K, R and t from projection matrix P, decompose using the RQ decomposition.
  @param P Input 3x4 projection matrix.
  @param K Output 3x3 camera matrix \f$K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$.
  @param R Output 3x3 rotation matrix.
  @param t Output 3x1 translation vector.

  Reference: @cite HartleyZ00 A4.1.1 pag.579
 */
CV_EXPORTS_W
void
KRtFromProjection( InputArray P, OutputArray K, OutputArray R, OutputArray t );

/** @brief Returns the depth of a point transformed by a rigid transform.
  @param R Input 3x3 rotation matrix.
  @param t Input 3x1 translation vector.
  @param X Input 3x1 or 4x1 vector with the 3d point.
 */
CV_EXPORTS_W
double
depth( InputArray R, InputArray t, InputArray X);

//! @} sfm

} /* namespace sfm */
} /* namespace cv */

#endif

/* End of file. */
