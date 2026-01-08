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
#include <opencv2/sfm/projection.hpp>

namespace opencv_test { namespace {

TEST(Sfm_projection, homogeneousToEuclidean)
{
    Matx33f X(1, 2, 3,
              4, 5, 6,
              2, 1, 0);

    Matx23f XEuclidean;
    homogeneousToEuclidean(X,XEuclidean);

    EXPECT_EQ((int) X.rows-1,(int) XEuclidean.rows );

    for(int y=0;y<X.rows-1;++y)
    {
        for(int x=0;x<X.cols;++x)
        {
            if (X(X.rows-1,x)!=0)
            {
                EXPECT_LE( std::abs(X(y,x)/X(X.rows-1, x) - XEuclidean(y,x)), 1e-4 );
            }
        }
    }
}

TEST(Sfm_projection, euclideanToHomogeneous)
{
    // Testing with floats
    Matx33f x(1, 2, 3,
              4, 5, 6,
              2, 1, 0);

    Matx43f XHomogeneous;
    euclideanToHomogeneous(x,XHomogeneous);

    EXPECT_EQ((int) x.rows+1,(int)XHomogeneous.rows );
    for(int i=0;i<x.cols;++i)
        EXPECT_EQ( 1,(int) XHomogeneous(x.rows,i) );


    // Testing with doubles
    Vec2d x2(4,3);
    Vec3d X2;

    euclideanToHomogeneous(x2,X2);

    EXPECT_EQ((int) x2.rows+1,(int)X2.rows );
    EXPECT_EQ( 4, X2(0) );
    EXPECT_EQ( 3, X2(1) );
    EXPECT_EQ( 1, X2(2) );
}

TEST(Sfm_projection, P_From_KRt)
{
  Matx33d K, Kp;
  K << 10,  1, 30,
        0, 20, 40,
        0,  0,  1;

  Matx33d R, Rp;
  R << 1, 0, 0,
       0, 1, 0,
       0, 0, 1;

  Vec3d t, tp;
  t << 1, 2, 3;

  Matx34d P(3,4);
  projectionFromKRt(K, R, t, P);
  KRtFromProjection(P, Kp, Rp, tp);

  EXPECT_MATRIX_NEAR(K, Kp, 1e-8);
  EXPECT_MATRIX_NEAR(R, Rp, 1e-8);
  EXPECT_VECTOR_NEAR(t, tp, 1e-8);

  // TODO: Change the code to ensure det(R) == 1, which is not currently
  // the case. Also add a test for that here.
}


}} // namespace
