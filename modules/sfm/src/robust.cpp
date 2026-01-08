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
#include <opencv2/sfm/robust.hpp>
#include <opencv2/sfm/numeric.hpp>

// libmv headers
#include "libmv/multiview/robust_fundamental.h"

using namespace std;

namespace cv
{
namespace sfm
{

// TODO: unify algorithms
template<typename T>
double
fundamentalFromCorrespondences8PointRobust( const Mat_<T> &_x1,
                                            const Mat_<T> &_x2,
                                            const double max_error,
                                            Mat_<T> _F,
                                            std::vector<int> &_inliers,
                                            const double outliers_probability )
{
  libmv::Mat x1, x2;
  libmv::Mat3 F;
  libmv::vector<int> inliers;

  cv2eigen( _x1, x1 );
  cv2eigen( _x2, x2 );

  T solution_error =
    libmv::FundamentalFromCorrespondences8PointRobust( x1, x2, max_error, &F, &inliers, outliers_probability );

  eigen2cv( F, _F );

  // transform from libmv::vector to std::vector
  int n = inliers.size();
  _inliers.resize(n);
  for( int i=0; i < n; ++i )
  {
    _inliers[i] = inliers.at(i);
  }

  return static_cast<double>(solution_error);
}


double
fundamentalFromCorrespondences8PointRobust( InputArray _x1,
                                            InputArray _x2,
                                            double max_error,
                                            OutputArray _F,
                                            OutputArray _inliers,
                                            double outliers_probability )
{
  const Mat x1 = _x1.getMat(), x2 = _x2.getMat();
  const int depth =  x1.depth();
  CV_Assert(x1.size() == x2.size() && (depth == CV_32F || depth == CV_64F));

  _F.create(3, 3, depth);

  Mat F = _F.getMat();
  std::vector<int>& inliers = *(std::vector<int>*)_inliers.getObj();

  double solution_error = 0.0;

  // type
  if( depth == CV_32F )
  {
    solution_error =
      fundamentalFromCorrespondences8PointRobust<float>(
        x1, x2, max_error, F, inliers, outliers_probability);
  }
  else
  {
    solution_error =
      fundamentalFromCorrespondences8PointRobust<double>(
        x1, x2, max_error, F, inliers, outliers_probability);
  }

  return solution_error;
}

template<typename T>
double
fundamentalFromCorrespondences7PointRobust( const Mat_<T> &_x1,
                                            const Mat_<T> &_x2,
                                            const double max_error,
                                            Mat_<T> _F,
                                            std::vector<int> &_inliers,
                                            const double outliers_probability )
{
  libmv::Mat x1, x2;
  libmv::Mat3 F;
  libmv::vector<int> inliers;

  cv2eigen( _x1, x1 );
  cv2eigen( _x2, x2 );

  T solution_error =
    libmv::FundamentalFromCorrespondences7PointRobust( x1, x2, max_error, &F, &inliers, outliers_probability );

  eigen2cv( F, _F );

  // transform from libmv::vector to std::vector
  int n = inliers.size();
  _inliers.resize(n);
  for( int i=0; i < n; ++i )
  {
    _inliers[i] = inliers.at(i);
  }

  return static_cast<double>(solution_error);
}

double
fundamentalFromCorrespondences7PointRobust( InputArray _x1,
                                            InputArray _x2,
                                            double max_error,
                                            OutputArray _F,
                                            OutputArray _inliers,
                                            double outliers_probability )
{
  const Mat x1 = _x1.getMat(), x2 = _x2.getMat();
  const int depth =  x1.depth();
  CV_Assert(x1.size() == x2.size() && (depth == CV_32F || depth == CV_64F));

  _F.create(3, 3, depth);

  Mat F = _F.getMat();
  std::vector<int>& inliers = *(std::vector<int>*)_inliers.getObj();

  double solution_error = 0.0;

  // type
  if( depth == CV_32F )
  {
    solution_error =
      fundamentalFromCorrespondences7PointRobust<float>(
        x1, x2, max_error, F, inliers, outliers_probability);
  }
  else
  {
    solution_error =
      fundamentalFromCorrespondences7PointRobust<double>(
        x1, x2, max_error, F, inliers, outliers_probability);
  }

  return solution_error;
}

} /* namespace sfm */
} /* namespace cv */