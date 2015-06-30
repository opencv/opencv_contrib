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

using namespace cv;
using namespace std;

/* Check projection errors */
static void
check_projection_errors(const Mat& X_estimated, const vector<Matx34d>& Ps,
                        const vector<Mat_<double> >& xs, float err_max2d)
{
    Mat X;
    euclideanToHomogeneous(X_estimated, X);   // 3D point

    for (int m = 0; m < xs.size(); ++m)
    {
        Mat x;
        homogeneousToEuclidean(cv::Mat(Ps[m]) * X, x); // 2d projection
        Mat projerr = xs[m] - x;

        for (int n = 0; n < projerr.cols; ++n)
        {
            double d = cv::norm(projerr.col(n));
            EXPECT_NEAR(0, d, err_max2d);
        }
    }
}


TEST(Sfm_reconstruct, twoViewProjectiveOutliers)
{
    float err_max2d = 1e-7;
    int nviews = 2;
    int npoints = 50;
    bool is_projective = true;
    bool has_outliers = true;

    vector<Mat_<double> > points2d;
    vector<cv::Matx33d> Rs;
    vector<cv::Vec3d> ts;
    vector<cv::Matx34d> Ps;
    Matx33d K;
    Mat_<double> points3d;
    generateScene(nviews, npoints, is_projective, K, Rs, ts, Ps, points3d, points2d);

    Mat_<double> points3d_estimated;
    vector<Mat> Ps_estimated;
    reconstruct(points2d, Ps_estimated, K, points3d_estimated, is_projective, has_outliers);

    /* Check projection errors on GT */
    check_projection_errors(points3d, Ps, points2d, err_max2d);

    /* Check projection errors on estimates */
    vector<cv::Matx34d> Ps_estimated_d;
    Ps_estimated_d.resize(Ps_estimated.size());
    for(size_t i=0; i<Ps_estimated.size(); ++i)
        Ps_estimated_d[i] = Ps_estimated[i];
    check_projection_errors(points3d_estimated, Ps_estimated_d, points2d, err_max2d);
}
