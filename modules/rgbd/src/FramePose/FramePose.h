/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
#ifndef _H_FRAME_POSE
#define _H_FRAME_POSE
#ifdef HAVE_EIGEN
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>

class FramePose
{

public:
  FramePose() {}

  FramePose(Eigen::Matrix3d &r_mat, Eigen::Vector3d &t_vec)
  {
    set_values(r_mat, t_vec);
  }

  void set_values(Eigen::Matrix3d &r_mat, Eigen::Vector3d &t_vec)
  {
    this->rmat = r_mat;
    this->tvec = t_vec;

    calculate_values();
  }

  cv::Mat R;
  cv::Mat t;
  cv::Mat proj_mat;
  Eigen::Matrix3d rmat;
  Eigen::Vector3d tvec;
  cv::Point3d cam_pose;
  cv::Mat cam_pose_mat;

private:
  void calculate_values()
  {
    eigen2cv(rmat, R);
    eigen2cv(tvec, t);
    R.convertTo(R, CV_32F);
    t.convertTo(t, CV_32F);
    auto pos = -(rmat.transpose()) * tvec;
    cam_pose = cv::Point3d(pos[0], pos[1], pos[2]);

    proj_mat = cv::Mat(3, 4, CV_32F);
    R.copyTo(proj_mat.colRange(0, 3));
    t.copyTo(proj_mat.col(3));

    cam_pose_mat = -R.t() * t;
  }
};

#endif
#endif
