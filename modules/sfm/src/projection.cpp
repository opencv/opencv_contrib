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
#include <opencv2/sfm/numeric.hpp>
#include <opencv2/sfm/projection.hpp>

// libmv headers
#include "libmv/multiview/projection.h"

#include <iostream>

namespace cv
{
namespace sfm
{

template<typename T>
void
homogeneousToEuclidean(const Mat & X_, Mat & x_)
{
  int d = X_.rows - 1;

  const Mat_<T> & X_rows = X_.rowRange(0,d);
  const Mat_<T> h = X_.row(d);

  const T * h_ptr = h[0], *h_ptr_end = h_ptr + h.cols;
  const T * X_ptr = X_rows[0];
  T * x_ptr = x_.ptr<T>(0);
  for (; h_ptr != h_ptr_end; ++h_ptr, ++X_ptr, ++x_ptr)
  {
    const T * X_col_ptr = X_ptr;
    T * x_col_ptr = x_ptr, *x_col_ptr_end = x_col_ptr + d * x_.step1();
    for (; x_col_ptr != x_col_ptr_end; X_col_ptr+=X_rows.step1(), x_col_ptr+=x_.step1() )
      *x_col_ptr = (*X_col_ptr) / (*h_ptr);
  }
}

void
homogeneousToEuclidean(InputArray X_, OutputArray x_)
{
  // src
  const Mat X = X_.getMat();

  // dst
   x_.create(X.rows-1, X.cols, X.type());
  Mat x = x_.getMat();

  // type
  if( X.depth() == CV_32F )
  {
    homogeneousToEuclidean<float>(X,x);
  }
  else
  {
    homogeneousToEuclidean<double>(X,x);
  }
}

void
euclideanToHomogeneous(InputArray x_, OutputArray X_)
{
  const Mat x = x_.getMat();
  const Mat last_row = Mat::ones(1, x.cols, x.type());
  vconcat(x, last_row, X_);
}

template<typename T>
void
projectionFromKRt(const Mat_<T> &K, const Mat_<T> &R, const Mat_<T> &t, Mat_<T> P)
{
  hconcat( K*R, K*t, P );
}

void
projectionFromKRt(InputArray K_, InputArray R_, InputArray t_, OutputArray P_)
{
  const Mat K = K_.getMat(), R = R_.getMat(), t = t_.getMat();
  const int depth = K.depth();
  CV_Assert((K.cols == 3 && K.rows == 3) && (t.cols == 1 && t.rows == 3) && (K.size() == R.size()));
  CV_Assert((depth == CV_32F || depth == CV_64F) && depth == R.depth() && depth == t.depth());

  P_.create(3, 4, depth);

  Mat P = P_.getMat();

  // type
  if( depth == CV_32F )
  {
    projectionFromKRt<float>(K, R, t, P);
  }
  else
  {
    projectionFromKRt<double>(K, R, t, P);
  }

}

template<typename T>
void
KRtFromProjection( const Mat_<T> &P_, Mat_<T> K_, Mat_<T> R_, Mat_<T> t_ )
{
  libmv::Mat34 P;
  libmv::Mat3 K, R;
  libmv::Vec3 t;

  cv2eigen( P_, P );

  libmv::KRt_From_P( P, &K, &R, &t );

  eigen2cv( K, K_ );
  eigen2cv( R, R_ );
  eigen2cv( t, t_ );
}

void
KRtFromProjection( InputArray P_, OutputArray K_, OutputArray R_, OutputArray t_ )
{
  const Mat P = P_.getMat();
  const int depth = P.depth();
  CV_Assert((P.cols == 4 && P.rows == 3) && (depth == CV_32F || depth == CV_64F));

  K_.create(3, 3, depth);
  R_.create(3, 3, depth);
  t_.create(3, 1, depth);

  Mat K = K_.getMat(), R = R_.getMat(), t = t_.getMat();

  // type
  if( depth == CV_32F )
  {
    KRtFromProjection<float>(P, K, R, t);
  }
  else
  {
    KRtFromProjection<double>(P, K, R, t);
  }
}

template<typename T>
T
depthValue( const Mat_<T> &R_, const Mat_<T> &t_, const Mat_<T> &X_ )
{
  Matx<T,3,3> R(R_);
  Vec<T,3> t(t_);

  if ( X_.rows == 3)
  {
    Vec<T,3> X(X_);
    return (R*X)(2) + t(2);
  }
  else
  {
    Vec<T,4> X(X_);
    Vec<T,3> Xe;
    homogeneousToEuclidean(X,Xe);
    return depthValue<T>( Mat(R), Mat(t), Mat(Xe) );
  }
}

double
depth( InputArray R_, InputArray t_, InputArray X_)
{
  const Mat R = R_.getMat(), t = t_.getMat(), X = X_.getMat();
  const int depth = R.depth();
  CV_Assert( R.rows == 3 && R.cols == 3 && t.rows == 3 && t.cols == 1 );
  CV_Assert( (X.rows == 3 && X.cols == 1) || (X.rows == 4 && X.cols == 1) );
  CV_Assert( depth == CV_32F || depth == CV_64F );

  double depth_value = 0.0;

  if ( depth == CV_32F )
  {
    depth_value = static_cast<double>(depthValue<float>(R, t, X));
  }
  else
  {
    depth_value = depthValue<double>(R, t, X);
  }

  return depth_value;
}

} /* namespace sfm */
} /* namespace cv */
