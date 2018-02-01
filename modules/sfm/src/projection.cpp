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
homogeneousToEuclidean(const Mat & _X, Mat & _x)
{
  int d = _X.rows - 1;

  const Mat_<T> & X_rows = _X.rowRange(0,d);
  const Mat_<T> h = _X.row(d);

  const T * h_ptr = h[0], *h_ptr_end = h_ptr + h.cols;
  const T * X_ptr = X_rows[0];
  T * x_ptr = _x.ptr<T>(0);
  for (; h_ptr != h_ptr_end; ++h_ptr, ++X_ptr, ++x_ptr)
  {
    const T * X_col_ptr = X_ptr;
    T * x_col_ptr = x_ptr, *x_col_ptr_end = x_col_ptr + d * _x.step1();
    for (; x_col_ptr != x_col_ptr_end; X_col_ptr+=X_rows.step1(), x_col_ptr+=_x.step1() )
      *x_col_ptr = (*X_col_ptr) / (*h_ptr);
  }
}

void
homogeneousToEuclidean(InputArray _X, OutputArray _x)
{
  // src
  const Mat X = _X.getMat();

  // dst
   _x.create(X.rows-1, X.cols, X.type());
  Mat x = _x.getMat();

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
euclideanToHomogeneous(InputArray _x, OutputArray _X)
{
  const Mat x = _x.getMat();
  const Mat last_row = Mat::ones(1, x.cols, x.type());
  vconcat(x, last_row, _X);
}

template<typename T>
void
projectionFromKRt(const Mat_<T> &K, const Mat_<T> &R, const Mat_<T> &t, Mat_<T> P)
{
  hconcat( K*R, K*t, P );
}

void
projectionFromKRt(InputArray _K, InputArray _R, InputArray _t, OutputArray _P)
{
  const Mat K = _K.getMat(), R = _R.getMat(), t = _t.getMat();
  const int depth = K.depth();
  CV_Assert((K.cols == 3 && K.rows == 3) && (t.cols == 1 && t.rows == 3) && (K.size() == R.size()));
  CV_Assert((depth == CV_32F || depth == CV_64F) && depth == R.depth() && depth == t.depth());

  _P.create(3, 4, depth);

  Mat P = _P.getMat();

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
KRtFromProjection( const Mat_<T> &_P, Mat_<T> _K, Mat_<T> _R, Mat_<T> _t )
{
  libmv::Mat34 P;
  libmv::Mat3 K, R;
  libmv::Vec3 t;

  cv2eigen( _P, P );

  libmv::KRt_From_P( P, &K, &R, &t );

  eigen2cv( K, _K );
  eigen2cv( R, _R );
  eigen2cv( t, _t );
}

void
KRtFromProjection( InputArray _P, OutputArray _K, OutputArray _R, OutputArray _t )
{
  const Mat P = _P.getMat();
  const int depth = P.depth();
  CV_Assert((P.cols == 4 && P.rows == 3) && (depth == CV_32F || depth == CV_64F));

  _K.create(3, 3, depth);
  _R.create(3, 3, depth);
  _t.create(3, 1, depth);

  Mat K = _K.getMat(), R = _R.getMat(), t = _t.getMat();

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
depthValue( const Mat_<T> &_R, const Mat_<T> &_t, const Mat_<T> &_X )
{
  Matx<T,3,3> R(_R);
  Vec<T,3> t(_t);

  if ( _X.rows == 3)
  {
    Vec<T,3> X(_X);
    return (R*X)(2) + t(2);
  }
  else
  {
    Vec<T,4> X(_X);
    Vec<T,3> Xe;
    homogeneousToEuclidean(X,Xe);
    return depthValue<T>( Mat(R), Mat(t), Mat(Xe) );
  }
}

double
depth( InputArray _R, InputArray _t, InputArray _X)
{
  const Mat R = _R.getMat(), t = _t.getMat(), X = _X.getMat();
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
