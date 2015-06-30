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

#ifdef __cplusplus

#include <opencv2/core.hpp>

namespace cv
{

/** Converts point coordinates from homogeneous to euclidean pixel coordinates. E.g., ((x,y,z)->(x/z, y/z))
* @param src Input vector of N-dimensional points
* @param dst Output vector of N-1-dimensional points.
*/
CV_EXPORTS
void
homogeneousToEuclidean(InputArray src, OutputArray dst);

/** Converts points from Euclidean to homogeneous space. E.g., ((x,y)->(x,y,1))
* @param src Input vector of N-dimensional points
* @param dst Output vector of N+1-dimensional points.
*/
CV_EXPORTS
void
euclideanToHomogeneous(InputArray src, OutputArray dst);

/** Get projection matrix P from K, R and t.
 *  P = K * [R|t]
 */
CV_EXPORTS
void
P_From_KRt(const Matx33d &K, const Matx33d &R, const Vec3d &t, Matx34d &P);

/** Decompose using the RQ decomposition HZ A4.1.1 pag.579 */
CV_EXPORTS
void
KRt_From_P( const Matx34d &P, Matx33d &K, Matx33d &R, Vec3d &t );

CV_EXPORTS
double
depth(const Matx33d &R, const Vec3d &t, const Vec3d &X);

CV_EXPORTS
double
depth(const Matx33d &R, const Vec3d &t, const Vec4d &X);

} /* namespace cv */

#endif /* __cplusplus */

#endif

/* End of file. */
