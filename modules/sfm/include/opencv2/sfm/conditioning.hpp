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

#ifdef __cplusplus

#include <opencv2/core.hpp>

namespace cv
{

/** Point conditioning (non isotropic)
    Reference: HZ2 4.4.4 pag.109
*/
CV_EXPORTS
void
preconditionerFromPoints( const Mat &points,
                          Mat &T );

/** Point conditioning (isotropic)
    Reference: HZ2 4.4.4 pag.107
*/
CV_EXPORTS
void
isotropicPreconditionerFromPoints( const Mat &points,
                                    Mat &T );

/** Apply Transformation to points such that transformed_points = T * points
*/
CV_EXPORTS
void
applyTransformationToPoints( const Mat &points,
                              const Mat &T,
                              Mat &transformed_points );

/** This function normalizes points (non isotropic)
* @param X Input vector of N-dimensional points
* @param x Output vector of the same N-dimensional points but with mean 0 and average norm sqrt(2)
* @param T Output transform matrix such that x = T*X
* Reference: HZ2 4.4.4 pag.109
*/
CV_EXPORTS
void
normalizePoints( const Mat &X,
                  Mat &x,
                  Mat &T );

/** This function normalizes points (isotropic)
* @param X Input vector of N-dimensional points
* @param x Output vector of the same N-dimensional points but with mean 0 and average norm sqrt(2)
* @param T Output transform matrix such that x = T*X
* Reference: HZ2 4.4.4 pag.107
*/
CV_EXPORTS
void
normalizeIsotropicPoints( const Mat &X,
                          Mat &x,
                          Mat &T );

} /* namespace cv */

#endif /* __cplusplus */

#endif

/* End of file. */