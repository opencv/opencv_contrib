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

namespace opencv_test { namespace {

TEST(Sfm_fundamental, fundamentalFromProjections)
{
    double tolerance_prop = 1e-7;
    double tolerance_near = 1e-15;

    Matx34d P1_gt, P2_gt;
    P1_gt << 1, 0, 0, 0,
             0, 1, 0, 0,
             0, 0, 1, 0;
    P2_gt << 1, 1, 1, 3,
             0, 2, 0, 3,
             0, 1, 1, 0;

    Matx33d F_gt;
    fundamentalFromProjections(P1_gt, P2_gt, F_gt);

    Matx34d P1, P2;
    projectionsFromFundamental(F_gt, P1, P2);

    Matx33d F;
    fundamentalFromProjections(P1, P2, F);

    Matx33d F_gt_norm, F_norm;
    normalizeFundamental(F_gt, F_gt_norm);
    normalizeFundamental(F, F_norm);

    EXPECT_MATRIX_PROP(F_gt, F, tolerance_prop);
    EXPECT_MATRIX_NEAR(F_gt_norm, F_norm, tolerance_near);
}


TEST(Sfm_fundamental, normalizedEightPointSolver)
{
    double tolerance = 1e-14;

    TwoViewDataSet d;
    generateTwoViewRandomScene( d );

    Matx33d F;
    normalizedEightPointSolver( d.x1, d.x2, F );
    expectFundamentalProperties( F, d.x1, d.x2, tolerance );
}


TEST(Sfm_fundamental, motionFromEssential)
{
    double tolerance = 1e-8;

    TwoViewDataSet d;
    generateTwoViewRandomScene(d);

    Matx33d E;
    essentialFromRt(d.R1, d.t1, d.R2, d.t2, E);

    Matx33d R;
    cv::Vec3d t;
    relativeCameraMotion(d.R1, d.t1, d.R2, d.t2, R, t);
    cv::normalize(t, t);

    std::vector<Mat> Rs;
    std::vector<cv::Mat> ts;
    motionFromEssential(E, Rs, ts);
    bool one_solution_is_correct = false;
    for ( int i = 0; i < Rs.size(); ++i )
    {
        if ( (cvtest::norm(Rs[i], R, NORM_L2) < tolerance) && (cvtest::norm(ts[i], t, NORM_L2) < tolerance) )
        {
            one_solution_is_correct = true;
            break;
        }
    }
    EXPECT_TRUE(one_solution_is_correct);
}


TEST(Sfm_fundamental, fundamentalToAndFromEssential)
{
    double tolerance = 1e-15;
    TwoViewDataSet d;
    generateTwoViewRandomScene(d);

    Matx33d F, E;
    essentialFromFundamental(d.F, d.K1, d.K2, E);
    fundamentalFromEssential(E, d.K1, d.K2, F);

    Matx33d F_gt_norm, F_norm;
    normalizeFundamental(d.F, F_gt_norm);
    normalizeFundamental(F, F_norm);

    EXPECT_MATRIX_NEAR(F_gt_norm, F_norm, tolerance);
}


TEST(Sfm_fundamental, essentialFromFundamental)
{
    TwoViewDataSet d;
    generateTwoViewRandomScene(d);

    Matx33d E_from_Rt;
    essentialFromRt(d.R1, d.t1, d.R2, d.t2, E_from_Rt);

    Matx33d E_from_F;
    essentialFromFundamental(d.F, d.K1, d.K2, E_from_F);

    EXPECT_MATRIX_PROP(E_from_Rt, E_from_F, 1e-6);
}


TEST(Sfm_fundamental, motionFromEssentialChooseSolution)
{
    TwoViewDataSet d;
    generateTwoViewRandomScene(d);

    Matx33d E;
    essentialFromRt(d.R1, d.t1, d.R2, d.t2, E);

    Matx33d R;
    cv::Vec3d t;
    relativeCameraMotion(d.R1, d.t1, d.R2, d.t2, R, t);
    normalize(t, t);

    std::vector < Mat > Rs;
    std::vector < cv::Mat > ts;
    motionFromEssential(E, Rs, ts);

    cv::Vec2d x1(d.x1(0, 0), d.x1(1, 0));
    cv::Vec2d x2(d.x2(0, 0), d.x2(1, 0));
    int solution = motionFromEssentialChooseSolution(Rs, ts, d.K1, x1, d.K2, x2);

    EXPECT_LE(0, solution);
    EXPECT_LE(solution, 3);
    EXPECT_LE(cvtest::norm(Rs[solution], Mat(R), NORM_L2), 1e-8);
    EXPECT_LE(cvtest::norm(ts[solution], Mat(t), NORM_L2), 1e-8);
}

}} // namespace
