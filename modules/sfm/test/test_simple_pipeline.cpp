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

#if CERES_FOUND

#include "test_precomp.hpp"

#include <opencv2/sfm/simple_pipeline.hpp>

namespace opencv_test { namespace {

const string SFM_DIR = "sfm";
const string TRACK_FILENAME = "backyard_tracks.txt";

TEST(Sfm_simple_pipeline, backyard)
{
    string trackFilename =
      string(TS::ptr()->get_data_path()) + SFM_DIR + "/" + TRACK_FILENAME;

    // Get tracks from file: check backyard.blend file
    std::vector<Mat> points2d;
    parser_2D_tracks( trackFilename, points2d );

    // Initial reconstruction
    int keyframe1 = 1, keyframe2 = 30;

    // Camera data
    double focal_length = 860.986572265625;  // f = 24mm (checked debugging blender)
    double principal_x = 400, principal_y = 225, k1 = -0.158, k2 = 0.131, k3 = 0;

    int refine_intrinsics = SFM_REFINE_FOCAL_LENGTH | SFM_REFINE_PRINCIPAL_POINT | SFM_REFINE_RADIAL_DISTORTION_K1 | SFM_REFINE_RADIAL_DISTORTION_K2;
    int select_keyframes = 0; // disable automatic keyframes selection
    int verbosity_level = -1; // mute logs

    libmv_CameraIntrinsicsOptions camera_instrinsic_options =
      libmv_CameraIntrinsicsOptions(SFM_DISTORTION_MODEL_POLYNOMIAL,
                                    focal_length, principal_x, principal_y,
                                    k1, k2, k3);
    libmv_ReconstructionOptions reconstruction_options(keyframe1, keyframe2, refine_intrinsics, select_keyframes, verbosity_level);

    Ptr<SFMLibmvEuclideanReconstruction> euclidean_reconstruction =
        SFMLibmvEuclideanReconstruction::create(camera_instrinsic_options, reconstruction_options);

    // Run reconstruction pipeline
    euclidean_reconstruction->run(points2d);

    double error = euclidean_reconstruction->getError();
    //cout << "euclidean_reconstruction error = " << error << endl;

    EXPECT_LE( error, 1.4 );  // actually 1.38671
                              // UPDATE:  1.38894
}

}} // namespace
#endif /* CERES_FOUND */
