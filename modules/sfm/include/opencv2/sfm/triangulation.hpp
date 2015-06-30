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

#ifndef __OPENCV_SFM_TRIANGULATION_HPP__
#define __OPENCV_SFM_TRIANGULATION_HPP__

#ifdef __cplusplus

#include <opencv2/core.hpp>

namespace cv
{

// /** Triangulates enum */
// enum
// {
//     CV_TRIANG_DLT = 0,         /*!< HZ 12.2 pag.312 */
//     CV_TRIANG_ALGEBRAIC = 1,   /*!< ... */
//     CV_TRIANG_BY_PLANE = 2,    /*!< Minimises the reprojection error */
// };


/** Triangulates the 3d position of 2d correspondences between two images, using the DLT
 * Reference: HZ 12.2 pag.312
 * @param xl vectors of 2d points (left camera). Has to be 2 x N
 * @param xr vectors of 2d points (right camera). Has to be 2 x N
 * @param Pl The 3 x 4 projection matrix of left camera
 * @param Pr The 3 x 4 projection matrix of right camera
 * @param points3d (output) 3d points. Is 3 x N
 */
CV_EXPORTS
void
triangulateDLT( const Vec2d &xl, const Vec2d &xr,
                const Matx34d &Pl, const Matx34d &Pr,
                Vec3d &points3d );


/** Triangulates the 3d position of 2d correspondences between n images, using the DLT
 * Reference: it is the standard DLT; for derivation see appendix of Keir's thesis
 * @param x  vectors of 2d points (n camera). Has to be 2 x N
 * @param Ps The 3 x 4 projections matrices of each image
 * @param points3d (output) 3d points. Is 3 x N
 */
CV_EXPORTS
void
nViewTriangulate( const Mat_<double> &x,
                  const std::vector<Matx34d> &Ps,
                  Vec3d &points3d );


} /* namespace cv */

#endif /* __cplusplus */

#endif

/* End of file. */
