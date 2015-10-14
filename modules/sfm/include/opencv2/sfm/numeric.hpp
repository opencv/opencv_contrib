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

#ifndef __OPENCV_SFM_NUMERIC_HPP__
#define __OPENCV_SFM_NUMERIC_HPP__

#include <opencv2/core.hpp>

#include <Eigen/Core>

namespace cv
{
namespace sfm
{

//! @addtogroup numeric
//! @{

/** @brief Computes the mean and variance of a given matrix along its rows.
  @param A Input NxN matrix.
  @param mean Output Nx1 matrix with computed mean.
  @param variance Output Nx1 matrix with computed variance.

  It computes in the same way as woud do @ref reduce but with \a Variance function.
*/
CV_EXPORTS_W
void
meanAndVarianceAlongRows( InputArray A,
                          OutputArray mean,
                          OutputArray variance );

/** @brief Returns the 3x3 skew symmetric matrix of a vector.
  @param x Input 3x1 vector.

  Reference: @cite HartleyZ00, p581, equation (A4.5).
*/
CV_EXPORTS_W
Mat
skew( InputArray x );

///** @brief Returns the skew anti-symmetric matrix of a vector.
//  @param x Input 3x3 matrix.
//*/
//CV_EXPORTS
//Matx33d
//skewMat( const Vec3d &x );
//
///** @brief Returns the skew anti-symmetric matrix of a vector with only the first two (independent) lines.
//  @param x Input 3x3 matrix.
//*/
//CV_EXPORTS
//Matx33d
//skewMatMinimal( const Vec3d &x );

//! @} numeric

} /* namespace sfm */
} /* namespace cv */

#endif

/* End of file. */
