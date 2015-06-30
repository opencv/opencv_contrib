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

#include "opencv2/sfm/projection.hpp"
#include "opencv2/sfm/triangulation.hpp"
#include "opencv2/sfm/fundamental.hpp"
#include "opencv2/sfm/numeric.hpp"

#include "libmv/multiview/robust_fundamental.h"
#include "libmv/multiview/fundamental.h"
#include <opencv2/sfm/eigen.hpp>
#include <opencv2/core/eigen.hpp>

using namespace std;

namespace cv
{

    void
    projectionsFromFundamental( const Matx33d &F, Matx34d &P1, Matx34d &P2 )
    {
        P1 << 1, 0, 0, 0,
              0, 1, 0, 0,
              0, 0, 1, 0;

        Vec3d e2;
        cv::SVD::solveZ(F.t(), e2);

        Matx33d P2cols = skewMat(e2) * F;
        for(char j=0;j<3;++j) {
          for(char i=0;i<3;++i)
            P2(j,i) = P2cols(j,i);
          P2(j,3) = e2(j);
        }
    }

    void
    fundamentalFromProjections( const Matx34d &P1, const Matx34d &P2, Matx33d &_F )
    {
        Mat X[3];
        vconcat( P1.row(1), P1.row(2), X[0] );
        vconcat( P1.row(2), P1.row(0), X[1] );
        vconcat( P1.row(0), P1.row(1), X[2] );

        Mat Y[3];
        vconcat( P2.row(1), P2.row(2), Y[0] );
        vconcat( P2.row(2), P2.row(0), Y[1] );
        vconcat( P2.row(0), P2.row(1), Y[2] );

        Mat_<double> XY;
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                vconcat(X[j], Y[i], XY);
                _F(i, j) = determinant(XY);
            }
        }
    }

    void
    normalizedEightPointSolver( const Mat_<double> &_x1, const Mat_<double> &_x2, Matx33d &_F )
    {
        libmv::Mat x1, x2;
        libmv::Mat3 F;

        cv2eigen(_x1, x1);
        cv2eigen(_x2, x2);

        libmv::NormalizedEightPointSolver(x1, x2, &F);

        eigen2cv(F, _F);
    }

    void
    relativeCameraMotion( const Matx33d &R1, const Vec3d &t1, const Matx33d &R2,
                          const Vec3d &t2, Matx33d &R, Vec3d &t )
    {
        R = R2 * R1.t();
        t = t2 - R * t1;
    }

// MotionFromEssential
    void
    motionFromEssential( const Matx33d &_E, vector<Matx33d> &_Rs,
                         vector<Vec3d> &_ts )
    {
        libmv::Mat3 E;
        vector < libmv::Mat3 > Rs;
        vector < libmv::Vec3 > ts;

        cv2eigen(_E, E);

        libmv::MotionFromEssential(E, &Rs, &ts);

        _Rs.clear();
        _ts.clear();

        int n = Rs.size();
        CV_Assert(ts.size() == n);

        for ( int i = 0; i < n; ++i )
        {
            Mat R_temp, t_temp;

            eigen2cv(Rs[i], R_temp);
            _Rs.push_back(R_temp);

            eigen2cv(ts[i], t_temp);
            _ts.push_back(t_temp);
        }
    }


    int motionFromEssentialChooseSolution( const vector<Matx33d> &Rs,
                                           const vector<Vec3d> &ts,
                                           const Matx33d &K1,
                                           const Vec2d &x1,
                                           const Matx33d &K2,
                                           const Vec2d &x2 )
    {

        CV_Assert( 4 == Rs.size());
        CV_Assert( 4 == ts.size());

        Matx34d P1, P2;
        Matx33d R1 = Matx33d::eye();
        Vec3d t1(0.0, 0.0, 0.0);

        P_From_KRt(K1, R1, t1, P1);

        for ( int i = 0; i < 4; ++i )
        {
            const Matx33d R2 = Rs[i];
            const Vec3d t2 = ts[i];
            P_From_KRt(K2, R2, t2, P2);

            Vec3d X;
            triangulateDLT(x1, x2, P1, P2, X);

            double d1 = depth(R1, t1, X);
            double d2 = depth(R2, t2, X);

            // Test if point is front to the two cameras.
            if ( d1 > 0 && d2 > 0 )
            {
                return i;
            }
        }

        return -1;
    }

// fundamentalFromEssential
    void
    fundamentalFromEssential( const Matx33d &E, const Matx33d &K1, const Matx33d &K2,
                              Matx33d &F )
    {
        F = K2.inv().t() * E * K1.inv();
    }

// essentialFromFundamental
    void
    essentialFromFundamental( const Matx33d &F, const Matx33d &K1, const Matx33d &K2,
                              Matx33d &E )
    {
        E = K2.t() * F * K1;
    }

    void
    essentialFromRt( const Matx33d &_R1,
                    const Vec3d &_t1,
                    const Matx33d &_R2,
                    const Vec3d &_t2,
                    Matx33d &_E )
    {
        libmv::Mat3 E;
        libmv::Mat3 R1, R2;
        libmv::Vec3 t1, t2;

        cv2eigen( _R1, R1 );
        cv2eigen( _t1, t1 );
        cv2eigen( _R2, R2 );
        cv2eigen( _t2, t2 );

        libmv::EssentialFromRt( R1, t1, R2, t2, &E );

        eigen2cv( E, _E );
    }

    // normalizeFundamental
    void
    normalizeFundamental( const Matx33d &F, Matx33d &F_normalized )
    {
        F_normalized = F * (1.0/norm(F,NORM_L2));  // Frobenius Norm

        if ( F_normalized(2,2) < 0 )
        {
            F_normalized *= -1;
        }
    }

} /* namespace cv */
