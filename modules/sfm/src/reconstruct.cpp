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

// Eigen
#include <Eigen/Core>

// Open CV
#include <opencv2/sfm.hpp>
#include <opencv2/sfm/projection.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d.hpp>

// libmv headers
#include "libmv/reconstruction/reconstruction.h"
#include "libmv/reconstruction/projective_reconstruction.h"

using namespace cv;
using namespace libmv;
using namespace std;

// Temp debug macro
#define BR exit(1);

namespace cv
{

  //  Reconstruction function for API
  void
  reconstruct(const InputArrayOfArrays points2d, OutputArrayOfArrays projection_matrices, InputOutputArray K,
              OutputArray points3d, bool is_projective, bool has_outliers, bool is_sequence)
  {
    const int nviews = points2d.total();
    Matx33d F;

    // OpenCV data types
    std::vector<Mat> pts2d;
    points2d.getMatVector(pts2d);
    const int depth = pts2d[0].depth();

    // Projective reconstruction

    if (is_projective)
    {

      // Two view reconstruction

      if (nviews == 2)
      {

        // Get fundamental matrix
        if ( has_outliers )
        {
          double max_error = 0.1;
          std::vector<int> inliers;
          fundamentalFromCorrespondences8PointRobust(pts2d[0], pts2d[1], max_error, F, inliers);
        }
        else
        {
          normalizedEightPointSolver(pts2d[0], pts2d[1], F);
        }

        // Get Projection matrices
        Matx34d P, Pp;
        projectionsFromFundamental(F, P, Pp);
        projection_matrices.create(2, 1, depth);
        Mat(P).copyTo(projection_matrices.getMatRef(0));
        Mat(Pp).copyTo(projection_matrices.getMatRef(1));

        // Triangulate and find 3D points using inliers
        triangulatePoints(points2d, projection_matrices, points3d);
      }

    }

    // Affine reconstruction

    else
    {

      // Two view reconstruction

      if (nviews == 2)
      {

      }
      else
      {

      }

    }

  }

  void
  reconstruct(InputArrayOfArrays points2d, OutputArrayOfArrays Rs, OutputArrayOfArrays Ts, InputOutputArray K,
              OutputArray points3d, bool is_projective, bool has_outliers, bool is_sequence)
  {
    const int nviews = points2d.total();

    std::vector<Mat> pts2d;
    points2d.getMatVector(pts2d);

    const int depth = pts2d[0].depth();
    Rs.create(nviews, 1, depth);
    Ts.create(nviews, 1, depth);

    Matx33d Ka, R;
    Vec3d t;

    if (nviews == 2)
    {
      std::vector < Mat > Ps_estimated;
      reconstruct(points2d, Ps_estimated, K, points3d, is_projective, has_outliers, is_sequence);

      for(unsigned int i = 0; i < nviews; ++i)
      {
        KRt_From_P(Ps_estimated[i], Ka, R, t);
        Mat(R).copyTo(Rs.getMatRef(i));
        Mat(t).copyTo(Ts.getMatRef(i));
      }
      Mat(Ka).copyTo(K.getMat());
    }
    else if (nviews > 2)
    {
      Ka = K.getMat();
      CV_Assert( Ka(0,0) > 0 && Ka(1,1) > 0);

      // Get tracks from 2d points
      libmv::Tracks tracks;

      if (is_sequence)
      {
        // Pairwise tracking
        parser_2D_tracks( pts2d, tracks );
      }
      else
      {
        // TODO: random images matching
      }

      // Initial reconstruction
      const int keyframe1 = 1, keyframe2 = nviews;

      // Camera data (hacked by now)
      const double focal_length = Ka(0,0);
      const double principal_x = Ka(0,2), principal_y = Ka(1,2), k1 = 0, k2 = 0, k3 = 0;

      // Refinement parameters
      libmv_Reconstruction libmv_reconstruction;
      int refine_intrinsics = SFM_BUNDLE_FOCAL_LENGTH | SFM_BUNDLE_PRINCIPAL_POINT | SFM_BUNDLE_RADIAL_K1 | SFM_BUNDLE_RADIAL_K2; // | SFM_BUNDLE_TANGENTIAL;  /* (see libmv::EuclideanBundleCommonIntrinsics) */

      // Perform reconstruction
      libmv_solveReconstruction( tracks, keyframe1, keyframe2,
                                 focal_length, principal_x, principal_y, k1, k2, k3,
                                 libmv_reconstruction, refine_intrinsics );

      // Extract estimated camera poses
      libmv::vector<libmv::EuclideanCamera> cameras = libmv_reconstruction.reconstruction.AllCameras();
      for(unsigned int i = 0; i < nviews; ++i)
      {
        eigen2cv(cameras[i].R, R);
        eigen2cv(cameras[i].t, t);
        Mat(R).copyTo(Rs.getMatRef(i));
        Mat(t).copyTo(Ts.getMatRef(i));
      }

      // Extract reconstructed points
      libmv::vector<EuclideanPoint> points = libmv_reconstruction.reconstruction.AllPoints();
      size_t n_points = (unsigned) points.size();

      points3d.create(3, n_points, CV_64F);
      Mat points3d_ = points3d.getMat();

      for ( unsigned i = 0; i < n_points; ++i )
        for ( int j = 0; j < 3; ++j )
          points3d_.at<double>(j, i) = points[i].X[j];

      // Extract refined intrinsic parameters
      eigen2cv(libmv_reconstruction.intrinsics.K(), Ka);
      Mat(Ka).copyTo(K.getMat());
    }

  }

} // namespace cv
