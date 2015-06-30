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

#include "test_precomp.hpp"

#include <opencv2/sfm/simple_pipeline.hpp>
//#include "third_party/ssba/Math/v3d_optimization.h"// less logging messages
//#include <opencv2/viz.hpp>        // only for visualization
//#include <opencv2/core/eigen.hpp> // only for visualization

using namespace cv;
using namespace cvtest;
using namespace std;

TEST(Sfm_simple_pipeline, backyard)
{
    //V3D::optimizerVerbosenessLevel = 0; // less logging messages

    // Get tracks from file: check backyard.blend file
    libmv::Tracks tracks;
    parser_2D_tracks( "backyard_tracks.txt", tracks );

    // Initial reconstruction
    int keyframe1 = 1, keyframe2 = 30;

    // Camera data
    double focal_length = 860.986572265625;  // f = 24mm (checked debugging blender)
    double principal_x = 400, principal_y = 225, k1 = -0.158, k2 = 0.131, k3 = 0;


    libmv_Reconstruction libmv_reconstruction;
    int refine_intrinsics = SFM_BUNDLE_FOCAL_LENGTH | SFM_BUNDLE_PRINCIPAL_POINT | SFM_BUNDLE_RADIAL_K1 | SFM_BUNDLE_RADIAL_K2; // | SFM_BUNDLE_TANGENTIAL;  /* (see libmv::EuclideanBundleCommonIntrinsics) */

    libmv_solveReconstruction( tracks, keyframe1, keyframe2,
                               focal_length, principal_x, principal_y, k1, k2, k3,
                               libmv_reconstruction, refine_intrinsics );

    cout << "libmv_reconstruction.error = " << libmv_reconstruction.error << endl;

    EXPECT_LE( libmv_reconstruction.error, 1.4 );  // actually 1.38671

/*
    // Extract data from reconstruction
    libmv::vector<libmv::EuclideanPoint> points = libmv_reconstruction.reconstruction.AllPoints();
    libmv::vector<libmv::EuclideanCamera> cameras = libmv_reconstruction.reconstruction.AllCameras();

    size_t n_points = (unsigned) points.size();

    cv::Mat points3d = cv::Mat(3, n_points, CV_64F);

    for ( unsigned i = 0; i < n_points; ++i )
      for ( int j = 0; j < 3; ++j )
        points3d.at<double>(j, i) = points[i].X[j];

    std::vector<cv::Affine3d> path;
    for (size_t i = 0; i < cameras.size(); ++i)
    {
      cv::Mat R, t;
      eigen2cv(cameras[i].R, R);
      eigen2cv(cameras[i].t, t);

      path.push_back(cv::Affine3d(R,t));
    }

    /// Create a 3D window
    viz::Viz3d myWindow("Coordinate Frame");

    /// Add coordinate axes
    myWindow.showWidget("Coordinate Widget", viz::WCoordinateSystem());

    vector<Vec3f> point_cloud;
    for (int i = 0; i < n_points; ++i) {
        // recover ground truth points3d
        Vec3f point3d((float) points3d.at<double>(0, i),
                      (float) points3d.at<double>(1, i),
                      (float) points3d.at<double>(2, i));
        point_cloud.push_back(point3d);
    }

    /// Add the pointcloud
    viz::WCloud cloud_widget(point_cloud, viz::Color::green());
    myWindow.showWidget("point_cloud", cloud_widget);
    myWindow.showWidget("cameras_frames_and_lines", viz::WTrajectory(path, viz::WTrajectory::BOTH, 0.2, viz::Color::green()));

    myWindow.spin();
*/
}
