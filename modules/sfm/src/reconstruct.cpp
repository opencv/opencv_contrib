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

#if CERES_FOUND

// Eigen
#include <Eigen/Core>

// OpenCV
#include <opencv2/sfm.hpp>

#include <iostream>

using namespace cv;
using namespace cv::sfm;
using namespace std;

namespace cv
{
namespace sfm
{

  template<class T>
  void
  reconstruct_(const T &input, OutputArray Rs, OutputArray Ts, InputOutputArray K, OutputArray points3d, const bool refinement=true)
  {
    // Initial reconstruction
    const int keyframe1 = 1, keyframe2 = 2;
    const int select_keyframes = 1; // enable automatic keyframes selection
    const int verbosity_level = -1; // mute libmv logs

    // Refinement parameters
    const int refine_intrinsics = ( !refinement ) ? 0 :
        SFM_REFINE_FOCAL_LENGTH | SFM_REFINE_PRINCIPAL_POINT | SFM_REFINE_RADIAL_DISTORTION_K1 | SFM_REFINE_RADIAL_DISTORTION_K2;

    // Camera data
    Matx33d Ka = K.getMat();
    const double focal_length = Ka(0,0);
    const double principal_x = Ka(0,2), principal_y = Ka(1,2), k1 = 0, k2 = 0, k3 = 0;

    // Set reconstruction options
    libmv_ReconstructionOptions reconstruction_options(keyframe1, keyframe2, refine_intrinsics, select_keyframes, verbosity_level);

    libmv_CameraIntrinsicsOptions camera_instrinsic_options =
      libmv_CameraIntrinsicsOptions(SFM_DISTORTION_MODEL_POLYNOMIAL,
                                    focal_length, principal_x, principal_y,
                                    k1, k2, k3);

    //-- Instantiate reconstruction pipeline
    Ptr<BaseSFM> reconstruction =
      SFMLibmvEuclideanReconstruction::create(camera_instrinsic_options, reconstruction_options);

    //-- Run reconstruction pipeline
    reconstruction->run(input, K, Rs, Ts, points3d);

  }


  //  Reconstruction function for API
  void
  reconstruct(InputArrayOfArrays points2d, OutputArray Ps, OutputArray points3d, InputOutputArray K,
              bool is_projective)
  {
    const int nviews = points2d.total();
    CV_Assert( nviews >= 2 );

    // OpenCV data types
    std::vector<Mat> pts2d;
    points2d.getMatVector(pts2d);
    const int depth = pts2d[0].depth();

    Matx33d Ka = K.getMat();

    // Projective reconstruction

    if (is_projective)
    {

      if ( nviews == 2 )
      {
        // Get Projection matrices
        Matx33d F;
        Matx34d P, Pp;

        normalizedEightPointSolver(pts2d[0], pts2d[1], F);
        projectionsFromFundamental(F, P, Pp);
        Ps.create(2, 1, depth);
        Mat(P).copyTo(Ps.getMatRef(0));
        Mat(Pp).copyTo(Ps.getMatRef(1));

        // Triangulate and find 3D points using inliers
        triangulatePoints(points2d, Ps, points3d);
      }
      else
      {
        std::vector<Mat> Rs, Ts;
        reconstruct(points2d, Rs, Ts, Ka, points3d, is_projective);

        // From Rs and Ts, extract Ps
        const int nviews = Rs.size();
        Ps.create(nviews, 1, depth);

        Matx34d P;
        for (size_t i = 0; i < nviews; ++i)
        {
          projectionFromKRt(Ka, Rs[i], Vec3d(Ts[i]), P);
          Mat(P).copyTo(Ps.getMatRef(i));
        }

        Mat(Ka).copyTo(K.getMat());
      }

    }


    // Affine reconstruction

    else
    {
      // TODO: implement me
      CV_Error(Error::StsNotImplemented, "Affine reconstruction not yet implemented");
    }

  }


  void
  reconstruct(InputArrayOfArrays points2d, OutputArray Rs, OutputArray Ts, InputOutputArray K,
              OutputArray points3d, bool is_projective)
  {
    const int nviews = points2d.total();
    CV_Assert( nviews >= 2 );


    // Projective reconstruction

    if (is_projective)
    {

      // calls simple pipeline
      reconstruct_(points2d, Rs, Ts, K, points3d);

    }

    // Affine reconstruction

    else
    {
      // TODO: implement me
      CV_Error(Error::StsNotImplemented, "Affine reconstruction not yet implemented");
    }

  }


  void
  reconstruct(const std::vector<cv::String> images, OutputArray Ps, OutputArray points3d,
              InputOutputArray K, bool is_projective)
  {
    const int nviews = static_cast<int>(images.size());
    CV_Assert( nviews >= 2 );

    Matx33d Ka = K.getMat();
    const int depth = Mat(Ka).depth();

    // Projective reconstruction

    if ( is_projective )
    {
      std::vector<Mat> Rs, Ts;
      reconstruct(images, Rs, Ts, Ka, points3d, is_projective);

      // From Rs and Ts, extract Ps

      const int nviews_est = Rs.size();
      Ps.create(nviews_est, 1, depth);

      Matx34d P;
      for (size_t i = 0; i < nviews_est; ++i)
      {
        projectionFromKRt(Ka, Rs[i], Vec3d(Ts[i]), P);
        Mat(P).copyTo(Ps.getMatRef(i));
      }

      Mat(Ka).copyTo(K.getMat());
      }


    // Affine reconstruction

    else
    {
      // TODO: implement me
      CV_Error(Error::StsNotImplemented, "Affine reconstruction not yet implemented");
    }

  }


  void
  reconstruct(const std::vector<cv::String> images, OutputArray Rs, OutputArray Ts,
              InputOutputArray K, OutputArray points3d, bool is_projective)
  {
    const int nviews = static_cast<int>(images.size());
    CV_Assert( nviews >= 2 );

    // Projective reconstruction

    if ( is_projective )
    {
      reconstruct_(images, Rs, Ts, K, points3d);
    }


    // Affine reconstruction

    else
    {
      // TODO: implement me
      CV_Error(Error::StsNotImplemented, "Affine reconstruction not yet implemented");
    }

  }

} // namespace sfm
} // namespace cv

#endif /* HAVE_CERES */