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

// libmv headers
#include "libmv/numeric/numeric.h"

#include <iostream>

namespace cv
{
namespace sfm
{

template<typename T>
void
meanAndVarianceAlongRows( const Mat_<T> &A,
                          Mat_<T> mean,
                          Mat_<T> variance )
{
  const int n = A.rows, m = A.cols;

  for( int i = 0; i < n; ++i )
  {
    mean(i) = 0;
    variance(i) = 0;

    for( int j = 0; j < m; ++j )
    {
      T x = A(i,j);
      mean(i) += x;
      variance(i) += x*x;
    }
  }

  mean /= m;
  for (int i = 0; i < n; ++i) {
    variance(i) = variance(i) / m - (mean(i)*mean(i));
  }
}

void
meanAndVarianceAlongRows( InputArray _A,
                          OutputArray _mean,
                          OutputArray _variance )
{
  const Mat A = _A.getMat();
  const int depth = A.depth();
  CV_Assert( depth == CV_32F || depth == CV_64F );

  _mean.create(A.rows, 1, depth);
  _variance.create(A.rows, 1, depth);

  Mat mean = _mean.getMat(), variance = _variance.getMat();

  if( depth == CV_32F )
  {
    meanAndVarianceAlongRows<float>( A, mean, variance );
  }
  else
  {
    meanAndVarianceAlongRows<double>( A, mean, variance );
  }
}


//template<typename T>
//inline Mat
//skewMatMinimal( const Mat_<T> &x )
//{
//  Mat_<T> skew(2,3);
//  skew << 0, -1,  x(1),
//          1,  0, -x(0);
//  return skew;
//}
//
//Mat
//skewMatMinimal( InputArray _x )
//{
//  Mat x = _x.getMat();
//  CV_Assert( x.rows == 3 && x.cols == 1 );
//
//  int depth = x.depth();
//  if( depth == CV_32F )
//  {
//    return skewMatMinimal<float>(x);
//  }
//  else
//  {
//    return skewMatMinimal<double>(x);
//  }
//}

template<typename T>
Mat
skewMat( const Mat_<T> &x )
{
  Mat_<T> skew(3,3);
  skew <<   0 , -x(2),  x(1),
          x(2),    0 , -x(0),
         -x(1),  x(0),    0;

  return std::move(skew);
}

Mat
skew( InputArray _x )
{
  const Mat x = _x.getMat();
  const int depth = x.depth();
  CV_Assert( x.size() == Size(3,1) || x.size() == Size(1,3) );
  CV_Assert( depth == CV_32F || depth == CV_64F );

  Mat skewMatrix;
  if( depth == CV_32F )
  {
    skewMatrix = skewMat<float>(x);
  }
  else if( depth == CV_64F )
  {
    skewMatrix = skewMat<double>(x);
  }
  else
  {
    //CV_Error(CV_StsBadArg, "The DataType must be CV_32F or CV_64F");
  }

  return skewMatrix;
}


} /* namespace sfm */
} /* namespace cv */
