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

#ifdef __cplusplus

#include <vector>

#include <opencv2/core.hpp>

namespace cv
{

CV_EXPORTS
void
projectionsFromFundamental( const Matx33d &F,
                            Matx34d &P1,
                            Matx34d &P2 );

CV_EXPORTS
void
fundamentalFromProjections( const Matx34d &P1,
                            const Matx34d &P2,
                            Matx33d &F );

/**
 * The normalized 8-point fundamental matrix solver.
 * Reference: HZ2 11.2 pag.281 (x1 = x, x2 = x')
 */
CV_EXPORTS
void
normalizedEightPointSolver( const cv::Mat_<double> &x1,
                            const cv::Mat_<double> &x2,
                            Matx33d &F );

/**
 * Compute the relative camera motion between two cameras.
 *
 * Given the motion parameters of two cameras, computes the motion parameters
 * of the second one assuming the first one to be at the origin.
 * If T1 and T2 are the camera motions, the computed relative motion is
 *    T = T2 T1^{-1}
 */
CV_EXPORTS
void
relativeCameraMotion( const Matx33d &R1,
                      const Vec3d &t1,
                      const Matx33d &R2,
                      const Vec3d &t2,
                      Matx33d &R,
                      Vec3d &t );

/** Get Motion (R's and t's ) from Essential matrix.
 *  HZ 9.6 pag 259 (Result 9.19)
 */
CV_EXPORTS
void
motionFromEssential(const Matx33d &E, std::vector<Matx33d> &Rs, std::vector<Vec3d> &ts);

/**
 * Choose one of the four possible motion solutions from an essential matrix.
 *
 * Decides the right solution by checking that the triangulation of a match
 * x1--x2 lies in front of the cameras.  See HZ 9.6 pag 259 (9.6.3 Geometrical
 * interpretation of the 4 solutions)
 *
 * \return index of the right solution or -1 if no solution.
 */
CV_EXPORTS
int motionFromEssentialChooseSolution( const std::vector<Matx33d> &Rs,
                                       const std::vector<Vec3d> &ts,
                                       const Matx33d &K1,
                                       const Vec2d &x1,
                                       const Matx33d &K2,
                                       const Vec2d &x2 );

/** Get Essential matrix from Fundamental and Camera matrices
 *  HZ 9.6 pag 257 (formula 9.12)
 *  Or http://ai.stanford.edu/~birch/projective/node20.html
 */
CV_EXPORTS
void
fundamentalFromEssential(const Matx33d &E, const Matx33d &K1, const Matx33d &K2, Matx33d &F);

/** Get Essential matrix from Fundamental and Camera matrices
 *  HZ 9.6 pag 257 (formula 9.12)
 */
CV_EXPORTS
void
essentialFromFundamental(const Matx33d &F, const Matx33d &K1, const Matx33d &K2, Matx33d &E);

/** Get Essential matrix from Motion (R's and t's )
 *  HZ 9.6 pag 257 (formula 9.12)
 */
CV_EXPORTS
void
essentialFromRt( const Matx33d &R1,
                 const Vec3d &t1,
                 const Matx33d &R2,
                 const Vec3d &t2,
                 Matx33d &E );



CV_EXPORTS
void
normalizeFundamental( const Matx33d &F,
                      Matx33d &F_normalized );

} /* namespace cv */

#endif /* __cplusplus */

#endif

/* End of file. */
