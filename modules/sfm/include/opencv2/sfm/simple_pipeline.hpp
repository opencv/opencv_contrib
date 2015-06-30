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

#ifndef __OPENCV_SFM_SIMPLE_PIPELINE_HPP__
#define __OPENCV_SFM_SIMPLE_PIPELINE_HPP__

#ifdef __cplusplus

#include <opencv2/core.hpp>

#include "libmv/simple_pipeline/pipeline.h"
#include "libmv/simple_pipeline/camera_intrinsics.h"
#include "libmv/simple_pipeline/bundle.h"

namespace cv
{

enum { SFM_BUNDLE_FOCAL_LENGTH    = libmv::BUNDLE_FOCAL_LENGTH,
       SFM_BUNDLE_PRINCIPAL_POINT = libmv::BUNDLE_PRINCIPAL_POINT,
       SFM_BUNDLE_RADIAL_K1       = libmv::BUNDLE_RADIAL_K1,
       SFM_BUNDLE_RADIAL_K2       = libmv::BUNDLE_RADIAL_K2,
       SFM_BUNDLE_TANGENTIAL      = libmv::BUNDLE_TANGENTIAL
};

typedef struct libmv_Reconstruction
{
    libmv::EuclideanReconstruction reconstruction;

    /* used for per-track average error calculation after reconstruction */
    libmv::Tracks tracks;
    libmv::CameraIntrinsics intrinsics;

    double error;
} libmv_Reconstruction;

// Based on the 'libmv_solveReconstruction()' function from 'libmv_capi' (blender API)
CV_EXPORTS
void
libmv_solveReconstruction( const libmv::Tracks &tracks,
                           int keyframe1, int keyframe2,
                           double focal_length,
                           double principal_x, double principal_y,
                           double k1, double k2, double k3,
                           libmv_Reconstruction &libmv_reconstruction,
                           int refine_intrinsics = 0 );

void
parser_2D_tracks( const std::vector<cv::Mat> &points2d, libmv::Tracks &tracks );

} /* namespace cv */

#endif /* __cplusplus */

#endif

/* End of file. */
