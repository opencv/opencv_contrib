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

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include "test_precomp.hpp"

static cv::Matx33d randomK(bool is_projective)
{
  static cv::RNG rng;

  cv::Matx33d K = cv::Matx33d::zeros();
  K(0, 0) = rng.uniform(100, 1000);
  K(1, 1) = rng.uniform(100, 1000);
  if (is_projective)
  {
    K(0, 2) = rng.uniform(-100, 100);
    K(1, 2) = rng.uniform(-100, 100);
  }
  K(2, 2) = 1.0;

  return K;
}

void
generateScene(size_t n_views, size_t n_points, bool is_projective, cv::Matx33d & K, std::vector<cv::Matx33d> & R,
              std::vector<cv::Vec3d> & t, std::vector<cv::Matx34d> & P, cv::Mat_<double> & points3d,
              std::vector<cv::Mat_<double> > & points2d)
{
  R.resize(n_views);
  t.resize(n_views);

  cv::RNG rng;

  // Generate a bunch of random 3d points in a 0, 1 cube
  points3d.create(3, n_points);
  rng.fill(points3d, cv::RNG::UNIFORM, 0, 1);

  // Generate random intrinsics
  K = randomK(is_projective);

  // Generate random camera poses
  // TODO deal with smooth camera poses (e.g. from a video sequence)
  for (size_t i = 0; i < n_views; ++i)
  {
    // Get a random rotation axis
    cv::Vec3d vec;
    rng.fill(vec, cv::RNG::UNIFORM, 0, 1);
    // Give a random angle to the rotation vector
    vec = vec / cv::norm(vec) * rng.uniform(0.0f, float(2 * CV_PI));
    cv::Rodrigues(vec, R[i]);
    // Create a random translation
    t[i] = cv::Vec3d(rng.uniform(-0.5f, 0.5f), rng.uniform(-0.5f, 0.5f), rng.uniform(1.0f, 2.0f));
    // Make sure the shape is in front of the camera
    cv::Mat_<double> points3d_transformed = cv::Mat(R[i]) * points3d + cv::Mat(t[i]) * cv::Mat_<double> ::ones(1, n_points);
    double min_dist, max_dist;
    cv::minMaxIdx(points3d_transformed.row(2), &min_dist, &max_dist);
    if (min_dist < 0)
      t[i][2] = t[i][2] - min_dist + 1.0;
  }

  // Compute projection matrices
  P.resize(n_views);
  for (size_t i = 0; i < n_views; ++i)
  {
    cv::Matx33d K3 = K, R3 = R[i];
    cv::Vec3d t3 = t[i];
    cv::sfm::projectionFromKRt(K3, R3, t3, P[i]);
  }

  // Compute homogeneous 3d points
  cv::Mat_<double> points3d_homogeneous(4, n_points);
  points3d.copyTo(points3d_homogeneous.rowRange(0, 3));
  points3d_homogeneous.row(3).setTo(1);
  // Project those points for every view
  points2d.resize(n_views);
  for (size_t i = 0; i < n_views; ++i)
  {
    cv::Mat_<double> points2d_tmp = cv::Mat(P[i]) * points3d_homogeneous;
    points2d[i].create(2, n_points);
    for (unsigned char j = 0; j < 2; ++j)
      cv::Mat(points2d_tmp.row(j) / points2d_tmp.row(2)).copyTo(points2d[i].row(j));
  }

// TODO: remove a certain number of points per view
// TODO: add a certain number of outliers per view

}
