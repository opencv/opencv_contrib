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

#include "precomp.hpp"

// Eigen
#include <Eigen/Core>

// OpenCV
#include <opencv2/core/eigen.hpp>
#include <opencv2/sfm/conditioning.hpp>

// libmv headers
#include "libmv/multiview/conditioning.h"

namespace cv
{
namespace sfm
{

template<typename T>
void
preconditionerFromPoints( const Mat_<T> &_points,
                          Mat_<T> _Tr )
{
  libmv::Mat points;
  libmv::Mat3 Tr;

  cv2eigen( _points, points );

  libmv::PreconditionerFromPoints( points, &Tr );

  eigen2cv( Tr, _Tr );
}


void
preconditionerFromPoints( InputArray _points,
                          OutputArray _T )
{
  const Mat points = _points.getMat();
  const int depth = points.depth();
  CV_Assert((points.dims == 2 || points.dims == 3) && (depth == CV_32F || depth == CV_64F));

  _T.create(3, 3, depth);

  Mat T = _T.getMat();

  if ( depth == CV_32F )
  {
    preconditionerFromPoints<float>(points, T);
  }
  else
  {
    preconditionerFromPoints<double>(points, T);
  }
}

template<typename T>
void
isotropicPreconditionerFromPoints( const Mat_<T> &_points,
                                   Mat_<T> _T )
{
  libmv::Mat points;
  libmv::Mat3 Tr;

  cv2eigen( _points, points );

  libmv::IsotropicPreconditionerFromPoints( points, &Tr );

  eigen2cv( Tr, _T );
}

void
isotropicPreconditionerFromPoints( InputArray _points,
                                   OutputArray _T )
{
  const Mat points = _points.getMat();
  const int depth = points.depth();
  CV_Assert((points.dims == 2 || points.dims == 3) && (depth == CV_32F || depth == CV_64F));

  _T.create(3, 3, depth);

  Mat T = _T.getMat();

  if ( depth == CV_32F )
  {
    isotropicPreconditionerFromPoints<float>(points, T);
  }
  else
  {
    isotropicPreconditionerFromPoints<double>(points, T);
  }
}

template<typename T>
void
applyTransformationToPoints( const Mat_<T> &_points,
                             const Mat_<T> &_T,
                             Mat_<T> _transformed_points )
{
  libmv::Mat points, transformed_points;
  libmv::Mat3 Tr;

  cv2eigen( _points, points );
  cv2eigen( _T, Tr );

  libmv::ApplyTransformationToPoints( points, Tr, &transformed_points );

  eigen2cv( transformed_points, _transformed_points );
}

void
applyTransformationToPoints( InputArray _points,
                             InputArray _T,
                             OutputArray _transformed_points )
{
  const Mat points = _points.getMat(), T = _T.getMat();
  const int depth = points.depth();
  CV_Assert((points.dims == 2 || points.dims == 3) && T.size() == Size(3,3) && (depth == CV_32F || depth == CV_64F));

  _transformed_points.create(points.size(), depth);

  Mat transformed_points = _transformed_points.getMat();

  if ( depth == CV_32F )
  {
    applyTransformationToPoints<float>(points, T, transformed_points);
  }
  else
  {
    applyTransformationToPoints<double>(points, T, transformed_points);
  }
}

void
normalizePoints( InputArray points,
                 OutputArray normalized_points,
                 OutputArray T )
{
  preconditionerFromPoints(points, T);
  applyTransformationToPoints(points, T, normalized_points);
}

void
normalizeIsotropicPoints( InputArray points,
                          OutputArray normalized_points,
                          OutputArray T )
{
  isotropicPreconditionerFromPoints(points, T);
  applyTransformationToPoints(points, T, normalized_points);
}

} /* namespace sfm */
} /* namespace cv */
