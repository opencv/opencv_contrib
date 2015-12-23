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
using namespace cv::sfm;
using namespace std;

static void
checkTriangulation(int nviews, int npoints, bool is_projective, float err_max2d, float err_max3d)
{
    std::vector<Mat_<double> > points2d;
    std::vector<cv::Matx33d> Rs;
    std::vector<cv::Vec3d> ts;
    std::vector<cv::Matx34d> Ps;
    Matx33d K;
    Mat_<double> points3d;
    generateScene(nviews, npoints, is_projective, K, Rs, ts, Ps, points3d, points2d);

    // get 3d points
    cv::Mat X, X_homogeneous;
    std::vector<Mat_<double> > Ps_d(Ps.size());
    for(size_t i=0; i<Ps.size(); ++i)
        Ps_d[i] = cv::Mat_<double>(Ps[i]);
    triangulatePoints(points2d, Ps_d, X);
    euclideanToHomogeneous(X, X_homogeneous);

    for (int i = 0; i < npoints; ++i)
    {
        for (int k = 0; k < nviews; ++k)
        {
            cv::Mat x_reprojected;
            homogeneousToEuclidean( cv::Mat(Ps[k])*X_homogeneous.col(i), x_reprojected );

            // Check reprojection error. Should be nearly zero.
            double error = norm( x_reprojected - points2d[k].col(i) );
            EXPECT_LE(error*error, err_max2d);
        }

        // Check 3d error. Should be nearly zero.
        double error = norm( X.col(i) - points3d.col(i) );
        EXPECT_LE(error*error, err_max3d);
    }
}


TEST(Sfm_triangulate, TriangulateDLT)
{
    int nviews = 2;
    int npoints = 30;
    bool is_projective = true;

    checkTriangulation(nviews, npoints, is_projective, 1e-7, 1e-9);
}

TEST(Sfm_triangulate, NViewTriangulate_FiveViews)
{
    int nviews = 5;
    int npoints = 6;
    bool is_projective = true;

    checkTriangulation(nviews, npoints, is_projective, 1e-7, 1e-9);
}
