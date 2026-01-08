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

// Eigen
#include <Eigen/Core>

// OpenCV
#include <opencv2/core/eigen.hpp>
#include <opencv2/sfm/projection.hpp>
#include <opencv2/sfm/triangulation.hpp>
#include <opencv2/sfm/fundamental.hpp>
#include <opencv2/sfm/numeric.hpp>
#include <opencv2/sfm/conditioning.hpp>

// libmv headers
#include "libmv/multiview/fundamental.h"

#include <iostream>
using namespace std;

namespace cv
{
namespace sfm
{
  template<typename T>
  void
  projectionsFromFundamental( const Mat_<T> &F,
                              Mat_<T> P1,
                              Mat_<T> P2 )
  {
    P1 << 1, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 1, 0;

    Vec<T,3> e2;
    cv::SVD::solveZ(F.t(), e2);

    Mat_<T> P2cols = skew(e2) * F;
    for(char j=0;j<3;++j) {
      for(char i=0;i<3;++i)
        P2(j,i) = P2cols(j,i);
      P2(j,3) = e2(j);
    }

  }

  void
  projectionsFromFundamental( InputArray _F,
                              OutputArray _P1,
                              OutputArray _P2 )
  {
    const Mat F = _F.getMat();
    const int depth = F.depth();
    CV_Assert(F.cols == 3 && F.rows == 3 && (depth == CV_32F || depth == CV_64F));

    _P1.create(3, 4, depth);
    _P2.create(3, 4, depth);

    Mat P1 = _P1.getMat(),  P2 = _P2.getMat();

    // type
    if( depth == CV_32F )
    {
      projectionsFromFundamental<float>(F, P1, P2);
    }
    else
    {
      projectionsFromFundamental<double>(F, P1, P2);
    }

  }

  template<typename T>
  void
  fundamentalFromProjections( const Mat_<T> &P1,
                              const Mat_<T> &P2,
                              Mat_<T> F )
  {
    Mat_<T> X[3];
    vconcat( P1.row(1), P1.row(2), X[0] );
    vconcat( P1.row(2), P1.row(0), X[1] );
    vconcat( P1.row(0), P1.row(1), X[2] );

    Mat_<T> Y[3];
    vconcat( P2.row(1), P2.row(2), Y[0] );
    vconcat( P2.row(2), P2.row(0), Y[1] );
    vconcat( P2.row(0), P2.row(1), Y[2] );

    Mat_<T> XY;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
      {
        vconcat(X[j], Y[i], XY);
        F(i, j) = determinant(XY);
      }
  }

  void
  fundamentalFromProjections( InputArray _P1,
                              InputArray _P2,
                              OutputArray _F )
  {
    const Mat P1 = _P1.getMat(), P2 = _P2.getMat();
    const int depth = P1.depth();
    CV_Assert((P1.cols == 4 && P1.rows == 3) && P1.rows == P2.rows && P1.cols == P2.cols);
    CV_Assert((depth == CV_32F || depth == CV_64F) && depth == P2.depth());

    _F.create(3, 3, depth);

    Mat F = _F.getMat();

    // type
    if( depth == CV_32F )
    {
      fundamentalFromProjections<float>(P1, P2, F);
    }
    else
    {
      fundamentalFromProjections<double>(P1, P2, F);
    }

  }

  template<typename T>
  void
  normalizedEightPointSolver( const Mat_<T> &_x1,
                              const Mat_<T> &_x2,
                              Mat_<T> _F )
  {
    libmv::Mat x1, x2;
    libmv::Mat3 F;

    cv2eigen(_x1, x1);
    cv2eigen(_x2, x2);

    libmv::NormalizedEightPointSolver(x1, x2, &F);

    eigen2cv(F, _F);
  }

  void
  normalizedEightPointSolver( InputArray _x1, InputArray _x2, OutputArray _F )
  {
    const Mat x1 = _x1.getMat(), x2 = _x2.getMat();
    const int depth = x1.depth();
    CV_Assert(x1.dims == 2 && x1.dims == x2.dims && (depth == CV_32F || depth == CV_64F));

    _F.create(3, 3, depth);

    Mat F = _F.getMat();

    // type
    if( depth == CV_32F )
    {
      normalizedEightPointSolver<float>(x1, x2, F);
    }
    else
    {
      normalizedEightPointSolver<double>(x1, x2, F);
    }

  }

  template<typename T>
  void
  relativeCameraMotion( const Mat_<T> &R1,
                        const Mat_<T> &t1,
                        const Mat_<T> &R2,
                        const Mat_<T> &t2,
                        Mat_<T> R,
                        Mat_<T> t )
  {
    R = R2 * R1.t();
    t = t2 - R * t1;
  }

  void
  relativeCameraMotion( InputArray _R1, InputArray _t1, InputArray _R2,
                        InputArray _t2, OutputArray _R, OutputArray _t )
  {
    const Mat R1 = _R1.getMat(), t1 = _t1.getMat(), R2 = _R2.getMat(), t2 = _t2.getMat();
    const int depth = R1.depth();
    CV_Assert((R1.cols == 3 && R1.rows == 3) && (R1.size() == R2.size()));
    CV_Assert((t1.cols == 1 && t1.rows == 3) && (t1.size() == t2.size()));
    CV_Assert((depth == CV_32F || depth == CV_64F) && depth == R2.depth() && depth == t1.depth() && depth == t2.depth());

    _R.create(3, 3, depth);
    _t.create(3, 1, depth);

    Mat R = _R.getMat(), t = _t.getMat();

    // type
    if( depth == CV_32F )
    {
      relativeCameraMotion<float>(R1, t1, R2, t2, R, t);
    }
    else
    {
      relativeCameraMotion<double>(R1, t1, R2, t2, R, t);
    }

  }

  template<typename T>
  void
  motionFromEssential( const Mat_<T> &_E,
                       std::vector<Mat> &_Rs,
                       std::vector<Mat> &_ts )
  {
    libmv::Mat3 E;
    std::vector < libmv::Mat3 > Rs;
    std::vector < libmv::Vec3 > ts;

    cv2eigen(_E, E);

    libmv::MotionFromEssential(E, &Rs, &ts);

    _Rs.clear();
    _ts.clear();

    int n = Rs.size();
    CV_Assert(ts.size() == n);

    for ( int i = 0; i < n; ++i )
    {
      Mat_<T> R_temp, t_temp;

      eigen2cv(Rs[i], R_temp);
      _Rs.push_back(R_temp);

      eigen2cv(ts[i], t_temp);
      _ts.push_back(t_temp);
    }

  }

  void
  motionFromEssential( InputArray _E, OutputArrayOfArrays _Rs,
                       OutputArrayOfArrays _ts )
  {
    const Mat E = _E.getMat();
    const int depth = E.depth(), cn = 4;
    CV_Assert(E.cols == 3 && E.rows == 3 && (depth == CV_32F || depth == CV_64F));

    _Rs.create(cn, 1, depth);
    _ts.create(cn, 1, depth);
    for (int i = 0; i < cn; ++i)
    {
      _Rs.create(Size(3,3), depth, i);
      _ts.create(Size(3,1), depth, i);
    }

    std::vector<Mat> Rs, ts;
    _Rs.getMatVector(Rs);
    _ts.getMatVector(ts);

    // type
    if( depth == CV_32F )
    {
      motionFromEssential<float>(E, Rs, ts);
    }
    else
    {
      motionFromEssential<double>(E, Rs, ts);
    }

    for (int i = 0; i < cn; ++i)
    {
      Rs[i].copyTo(_Rs.getMatRef(i));
      ts[i].copyTo(_ts.getMatRef(i));
    }

  }

  template<typename T>
  int motionFromEssentialChooseSolution( const std::vector<Mat> &Rs,
                                         const std::vector<Mat> &ts,
                                         const Mat_<T> &K1,
                                         const Mat_<T> &x1,
                                         const Mat_<T> &K2,
                                         const Mat_<T> &x2 )
  {
    Mat_<T> P1, P2, R1 = Mat_<T>::eye(3,3);

    T val = static_cast<T>(0.0);
    Vec<T,3> t1(val, val, val);

    projectionFromKRt(K1, R1, t1, P1);

    std::vector<Mat_<T> > points2d;
    points2d.push_back(x1);
    points2d.push_back(x2);

    for ( int i = 0; i < 4; ++i )
    {
      const Mat_<T> R2 = Rs[i];
      const Vec<T,3> t2 = ts[i];
      projectionFromKRt(K2, R2, t2, P2);

      std::vector<Mat_<T> > Ps;
      Ps.push_back(P1);
      Ps.push_back(P2);

      Vec<T,3> X;
      triangulatePoints(points2d, Ps, X);

      T d1 = depth(R1, t1, X);
      T d2 = depth(R2, t2, X);

      // Test if point is front to the two cameras.
      if ( d1 > 0 && d2 > 0 )
      {
        return i;
      }
    }

    return -1;
  }

  int motionFromEssentialChooseSolution( InputArrayOfArrays _Rs,
                                         InputArrayOfArrays _ts,
                                         InputArray _K1,
                                         InputArray _x1,
                                         InputArray _K2,
                                         InputArray _x2 )
  {
    std::vector<Mat> Rs, ts;
    _Rs.getMatVector(Rs);
    _ts.getMatVector(ts);
    const Mat K1 = _K1.getMat(), x1 = _x1.getMat(), K2 = _K2.getMat(), x2 = _x2.getMat();
    const int depth = K1.depth();
    CV_Assert( Rs.size() == 4 && ts.size() == 4 );
    CV_Assert((K1.cols == 3 && K1.rows == 3) && (K1.size() == K2.size()));
    CV_Assert((x1.cols == 1 && x1.rows == 2) && (x1.size() == x2.size()));
    CV_Assert((depth == CV_32F || depth == CV_64F) && depth == K2.depth() && depth == x1.depth() && depth == x2.depth());

    int solution = 0;

    // type
    if( depth == CV_32F )
    {
      solution = motionFromEssentialChooseSolution<float>(Rs, ts, K1, x1, K2, x2);
    }
    else
    {
      solution = motionFromEssentialChooseSolution<double>(Rs, ts, K1, x1, K2, x2);
    }

    return solution;
  }

  template<typename T>
  void
  fundamentalFromEssential( const Mat_<T> &E,
                            const Mat_<T> &K1,
                            const Mat_<T> &K2,
                            Mat_<T> F )
  {
    F = K2.inv().t() * E * K1.inv();
  }

  void
  fundamentalFromEssential( InputArray _E,
                            InputArray _K1,
                            InputArray _K2,
                            OutputArray _F )
  {
    const Mat E = _E.getMat(), K1 = _K1.getMat(), K2 = _K2.getMat();
    const int depth =  E.depth();
    CV_Assert(E.cols == 3 && E.rows == 3 && E.size() == _K1.size() && E.size() == _K2.size() && (depth == CV_32F || depth == CV_64F));

    _F.create(3, 3, depth);

    Mat F = _F.getMat();

    // type
    if( depth == CV_32F )
    {
      fundamentalFromEssential<float>(E, K1, K2, F);
    }
    else
    {
      fundamentalFromEssential<double>(E, K1, K2, F);
    }

  }

  template<typename T>
  void
  essentialFromFundamental( const Mat_<T> &F,
                            const Mat_<T> &K1,
                            const Mat_<T> &K2,
                            Mat_<T> E )
  {
    E = K2.t() * F * K1;
  }

  void
  essentialFromFundamental( InputArray _F,
                            InputArray _K1,
                            InputArray _K2,
                            OutputArray _E )
  {
    const Mat F = _F.getMat(), K1 = _K1.getMat(), K2 = _K2.getMat();
    const int depth =  F.depth();
    CV_Assert(F.cols == 3 && F.rows == 3 && F.size() == _K1.size() && F.size() == _K2.size() && (depth == CV_32F || depth == CV_64F));

    _E.create(3, 3, depth);

    Mat E = _E.getMat();

    // type
    if( depth == CV_32F )
    {
      essentialFromFundamental<float>(F, K1, K2, E);
    }
    else
    {
      essentialFromFundamental<double>(F, K1, K2, E);
    }
  }

  template<typename T>
  void
  essentialFromRt( const Mat_<T> &_R1,
                   const Mat_<T> &_t1,
                   const Mat_<T> &_R2,
                   const Mat_<T> &_t2,
                   Mat_<T> _E )
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

  void
  essentialFromRt( InputArray _R1,
                   InputArray _t1,
                   InputArray _R2,
                   InputArray _t2,
                   OutputArray _E )
  {
    const Mat R1 = _R1.getMat(), t1 = _t1.getMat(), R2 = _R2.getMat(), t2 = _t2.getMat();
    const int depth = R1.depth();
    CV_Assert((R1.cols == 3 && R1.rows == 3) && (R1.size() == R2.size()));
    CV_Assert((t1.cols == 1 && t1.rows == 3) && (t1.size() == t2.size()));
    CV_Assert((depth == CV_32F || depth == CV_64F) && depth == R2.depth() && depth == t1.depth() && depth == t2.depth());

    _E.create(3, 3, depth);

    Mat E = _E.getMat();

    // type
    if( depth == CV_32F )
    {
      essentialFromRt<float>(R1, t1, R2, t2, E);
    }
    else
    {
      essentialFromRt<double>(R1, t1, R2, t2, E);
    }

  }

  template<typename T>
  void
  normalizeFundamental( const Mat_<T> &F, Mat_<T> F_normalized )
  {
    F_normalized = F * (1.0/norm(F,NORM_L2));  // Frobenius Norm

    if ( F_normalized(2,2) < 0 )
    {
        F_normalized *= -1;
    }
  }

  void
  normalizeFundamental( InputArray _F,
                        OutputArray _F_normalized )
  {
    const Mat F = _F.getMat();
    const int depth =  F.depth();
    CV_Assert(F.cols == 3 && F.rows == 3 && (depth == CV_32F || depth == CV_64F));

    _F_normalized.create(3, 3, depth);

    Mat F_normalized = _F_normalized.getMat();

    // type
    if( depth == CV_32F )
    {
      normalizeFundamental<float>(F, F_normalized);
    }
    else
    {
      normalizeFundamental<double>(F, F_normalized);
    }
  }

  template<typename T>
  void
  computeOrientation( const Mat_<T> &x1,
                      const Mat_<T> &x2,
                      Mat_<T> R,
                      Mat_<T> t,
                      T s )
  {
    Mat_<T> rr, rl, rt, lt;
    normalizePoints(x1, rr, rt);
    normalizePoints(x2, rl, lt);

    Mat_<T> rrBar, rlBar, rVar, lVar;
    meanAndVarianceAlongRows(rr, rrBar, rVar);
    meanAndVarianceAlongRows(rl, rlBar, lVar);

    Mat_<T> rrp, rlp;
    rrp = rr - repeat(rrBar, x1.rows, x1.cols);
    rlp = rl - repeat(rlBar, x2.rows, x2.cols);

    // TODO: finish implementation
    // https://github.com/vrabaud/sfm_toolbox/blob/master/sfm/computeOrientation.m#L44
  }

  void
  computeOrientation( InputArrayOfArrays _x1,
                      InputArrayOfArrays _x2,
                      OutputArray _R,
                      OutputArray _t,
                      double s )
  {
    const Mat x1 = _x1.getMat(), x2 = _x2.getMat();
    const int depth =  x1.depth();
    CV_Assert(x1.size() == x2.size() && (depth == CV_32F || depth == CV_64F));

    _R.create(3, 3, depth);
    _t.create(3, 1, depth);

    Mat R = _R.getMat(), t = _t.getMat();

    // type
    if( depth == CV_32F )
    {
      computeOrientation<float>(x1, x2, R, t, s);
    }
    else
    {
      computeOrientation<double>(x1, x2, R, t, s);
    }
  }

} /* namespace sfm */
} /* namespace cv */
