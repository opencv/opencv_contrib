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

#ifndef __OPENCV_SFM_FUNDAMENTAL_HPP__
#define __OPENCV_SFM_FUNDAMENTAL_HPP__

#include <vector>

#include <opencv2/core.hpp>

namespace cv
{
namespace sfm
{

//! @addtogroup fundamental
//! @{

/** @brief Get projection matrices from Fundamental matrix
  @param F Input 3x3 fundamental matrix.
  @param P1 Output 3x4 one possible projection matrix.
  @param P2 Output 3x4 another possible projection matrix.
 */
CV_EXPORTS_W
void
projectionsFromFundamental( InputArray F,
                            OutputArray P1,
                            OutputArray P2 );

/** @brief Get Fundamental matrix from Projection matrices.
  @param P1 Input 3x4 first projection matrix.
  @param P2 Input 3x4 second projection matrix.
  @param F Output 3x3 fundamental matrix.
 */
CV_EXPORTS_W
void
fundamentalFromProjections( InputArray P1,
                            InputArray P2,
                            OutputArray F );

/** @brief Estimate the fundamental matrix between two dataset of 2D point (image coords space).
  @param x1 Input 2xN Array of 2D points in view 1.
  @param x2 Input 2xN Array of 2D points in view 2.
  @param F Output 3x3 fundamental matrix.

  Uses the normalized 8-point fundamental matrix solver.
  Reference: @cite HartleyZ00 11.2 pag.281 (x1 = x, x2 = x')
 */
CV_EXPORTS_W
void
normalizedEightPointSolver( InputArray x1,
                            InputArray x2,
                            OutputArray F );

/** @brief Computes the relative camera motion between two cameras.
  @param R1 Input 3x3 first camera rotation matrix.
  @param t1 Input 3x1 first camera translation vector.
  @param R2 Input 3x3 second camera rotation matrix.
  @param t2 Input 3x1 second camera translation vector.
  @param R Output 3x3 relative rotation matrix.
  @param t Output 3x1 relative translation vector.

  Given the motion parameters of two cameras, computes the motion parameters
  of the second one assuming the first one to be at the origin.
  If T1 and T2 are the camera motions, the computed relative motion is \f$T = T_2 T_1^{-1}\f$
 */
CV_EXPORTS_W
void
relativeCameraMotion( InputArray R1,
                      InputArray t1,
                      InputArray R2,
                      InputArray t2,
                      OutputArray R,
                      OutputArray t );

/** Get Motion (R's and t's ) from Essential matrix.
  @param E Input 3x3 essential matrix.
  @param Rs Output vector of 3x3 rotation matrices.
  @param ts Output vector of 3x1 translation vectors.

  Reference: @cite HartleyZ00 9.6 pag 259 (Result 9.19)
 */
CV_EXPORTS_W
void
motionFromEssential( InputArray E,
                     OutputArrayOfArrays Rs,
                     OutputArrayOfArrays ts );

/** Choose one of the four possible motion solutions from an essential matrix.
  @param Rs Input vector of 3x3 rotation matrices.
  @param ts Input vector of 3x1 translation vectors.
  @param K1 Input 3x3 first camera matrix \f$K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$.
  @param x1 Input 2x1 vector with first 2d point.
  @param K2 Input 3x3 second camera matrix. The parameters are similar to K1.
  @param x2 Input 2x1 vector with second 2d point.

  Decides the right solution by checking that the triangulation of a match
  x1--x2 lies in front of the cameras. Return index of the right solution or -1 if no solution.

  Reference: See @cite HartleyZ00 9.6 pag 259 (9.6.3 Geometrical interpretation of the 4 solutions).
 */
CV_EXPORTS_W
int motionFromEssentialChooseSolution( InputArrayOfArrays Rs,
                                       InputArrayOfArrays ts,
                                       InputArray K1,
                                       InputArray x1,
                                       InputArray K2,
                                       InputArray x2 );

/** @brief Get Essential matrix from Fundamental and Camera matrices.
  @param E Input 3x3 essential matrix.
  @param K1 Input 3x3 first camera matrix \f$K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$.
  @param K2 Input 3x3 second camera matrix. The parameters are similar to K1.
  @param F Output 3x3 fundamental matrix.

  Reference: @cite HartleyZ00 9.6 pag 257 (formula 9.12) or http://ai.stanford.edu/~birch/projective/node20.html
 */
CV_EXPORTS_W
void
fundamentalFromEssential( InputArray E,
                          InputArray K1,
                          InputArray K2,
                          OutputArray F );

/** @brief Get Essential matrix from Fundamental and Camera matrices.
  @param F Input 3x3 fundamental matrix.
  @param K1 Input 3x3 first camera matrix \f$K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$.
  @param K2 Input 3x3 second camera matrix. The parameters are similar to K1.
  @param E Output 3x3 essential matrix.

  Reference: @cite HartleyZ00 9.6 pag 257 (formula 9.12)
 */
CV_EXPORTS_W
void
essentialFromFundamental( InputArray F,
                          InputArray K1,
                          InputArray K2,
                          OutputArray E );

/** @brief Get Essential matrix from Motion (R's and t's ).
  @param R1 Input 3x3 first camera rotation matrix.
  @param t1 Input 3x1 first camera translation vector.
  @param R2 Input 3x3 second camera rotation matrix.
  @param t2 Input 3x1 second camera translation vector.
  @param E Output 3x3 essential matrix.

  Reference: @cite HartleyZ00 9.6 pag 257 (formula 9.12)
 */
CV_EXPORTS_W
void
essentialFromRt( InputArray R1,
                 InputArray t1,
                 InputArray R2,
                 InputArray t2,
                 OutputArray E );

/** @brief Normalizes the Fundamental matrix.
  @param F Input 3x3 fundamental matrix.
  @param F_normalized Output 3x3 normalized fundamental matrix.

  By default divides the fundamental matrix by its L2 norm.
 */
CV_EXPORTS_W
void
normalizeFundamental( InputArray F,
                      OutputArray F_normalized );

/** @brief Computes Absolute or Exterior Orientation (Pose Estimation) between 2 sets of 3D point.
  @param x1 Input first 3xN or 2xN array of points.
  @param x2 Input second 3xN or 2xN array of points.
  @param R Output 3x3 computed rotation matrix.
  @param t Output 3x1 computed translation vector.
  @param s Output computed scale factor.

  Find the best transformation such that xp=projection*(s*R*x+t) (same as Pose Estimation, ePNP).
  The routines below are only for the orthographic case for now.
 */
CV_EXPORTS_W
void
computeOrientation( InputArrayOfArrays x1,
                    InputArrayOfArrays x2,
                    OutputArray R,
                    OutputArray t,
                    double s );

//! @} sfm

} /* namespace sfm */
} /* namespace cv */

#endif

/* End of file. */
