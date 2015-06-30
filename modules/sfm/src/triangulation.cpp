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

#include "opencv2/sfm/triangulation.hpp"
#include "opencv2/sfm/projection.hpp"

#include "libmv/multiview/twoviewtriangulation.h"
#include "libmv/multiview/fundamental.h"
#include <opencv2/core/eigen.hpp>


using namespace cv;
using namespace std;

namespace cv
{

// HZ 12.2 pag.312
void
triangulateDLT( const Vec2d &xl, const Vec2d &xr,
                const Matx34d &Pl, const Matx34d &Pr,
                Vec3d &points3d )
{
    Matx44d design;
    for (int i = 0; i < 4; ++i)
    {
        design(0,i) = xl(0) * Pl(2,i) - Pl(0,i);
        design(1,i) = xl(1) * Pl(2,i) - Pl(1,i);
        design(2,i) = xr(0) * Pr(2,i) - Pr(0,i);
        design(3,i) = xr(1) * Pr(2,i) - Pr(1,i);
    }

    Vec4d XHomogeneous;
    cv::SVD::solveZ(design, XHomogeneous);

    homogeneousToEuclidean(XHomogeneous, points3d);
}


// It is the standard DLT; for derivation see appendix of Keir's thesis.
void
nViewTriangulate(const Mat_<double> &x, const vector<Matx34d> &Ps, Vec3d &X)
{
    CV_Assert(x.rows == 2);
    unsigned nviews = x.cols;
    CV_Assert(nviews == Ps.size());

    cv::Mat_<double> design = cv::Mat_<double>::zeros(3*nviews, 4 + nviews);
    for (unsigned i=0; i < nviews; ++i) {
        for(char jj=0; jj<3; ++jj)
            for(char ii=0; ii<4; ++ii)
                design(3*i+jj, ii) = -Ps[i](jj, ii);
        design(3*i + 0, 4 + i) = x(0, i);
        design(3*i + 1, 4 + i) = x(1, i);
        design(3*i + 2, 4 + i) = 1.0;
    }

    Mat X_and_alphas;
    cv::SVD::solveZ(design, X_and_alphas);
    homogeneousToEuclidean(X_and_alphas.rowRange(0, 4), X);
}


void
triangulatePoints(InputArrayOfArrays _points2d, InputArrayOfArrays _projection_matrices,
                  OutputArray _points3d)
{
    // check
    size_t nviews = (unsigned) _points2d.total();
    CV_Assert(nviews >= 2 && nviews == _projection_matrices.total());

    // inputs
    size_t n_points;
    vector<Mat_<double> > points2d(nviews);
    vector<Matx34d> projection_matrices(nviews);
    {
        vector<Mat> points2d_tmp;
        _points2d.getMatVector(points2d_tmp);
        n_points = points2d_tmp[0].cols;

        vector<Mat> projection_matrices_tmp;
        _projection_matrices.getMatVector(projection_matrices_tmp);

        // Make sure the dimensions are right
        for(size_t i=0; i<nviews; ++i) {
            CV_Assert(points2d_tmp[i].rows == 2 && points2d_tmp[i].cols == n_points);
            if (points2d_tmp[i].type() == CV_64F)
                points2d[i] = points2d_tmp[i];
            else
                points2d_tmp[i].convertTo(points2d[i], CV_64F);

            CV_Assert(projection_matrices_tmp[i].rows == 3 && projection_matrices_tmp[i].cols == 4);
            if (projection_matrices_tmp[i].type() == CV_64F)
              projection_matrices[i] = projection_matrices_tmp[i];
            else
              projection_matrices_tmp[i].convertTo(projection_matrices[i], CV_64F);
        }
    }

    // output
    _points3d.create(3, n_points, CV_64F);
    cv::Mat points3d = _points3d.getMat();

    // Two view
    if( nviews == 2 )
    {
        const Mat_<double> &xl = points2d[0], &xr = points2d[1];

        const Matx34d & Pl = projection_matrices[0];    // left matrix projection
        const Matx34d & Pr = projection_matrices[1];    // right matrix projection

        // triangulate
        for( unsigned i = 0; i < n_points; ++i )
        {
            Vec3d point3d;
            triangulateDLT( Vec2d(xl(0,i), xl(1,i)), Vec2d(xr(0,i), xr(1,i)), Pl, Pr, point3d );
            for(char j=0; j<3; ++j)
                points3d.at<double>(j, i) = point3d[j];
        }
    }
    else if( nviews > 2 )
    {
        // triangulate
        for( unsigned i=0; i < n_points; ++i )
        {
            // build x matrix (one point per view)
            Mat_<double> x( 2, nviews );
            for( unsigned k=0; k < nviews; ++k )
            {
                points2d.at(k).col(i).copyTo( x.col(k) );
            }

            Vec3d point3d;
            nViewTriangulate( x, projection_matrices, point3d );
            for(char j=0; j<3; ++j)
                points3d.at<double>(j, i) = point3d[j];
        }
    }
}




// template<typename T>
// void
// triangulatePoints_(unsigned nviews, const vector<Mat> & points2d, const vector<Mat> & K,
//                    const vector<Mat> & R, const vector<Mat> & t,
//                    Mat & points3d, int method)
// {
//     if( method == CV_TRIANG_DLT)
//     {
//         // Compute projection matrices
//         std::vector<cv::Mat> P;
//         P.resize(nviews);
//         for (unsigned i = 0; i < nviews; ++i)
//         {
//             P_From_KRt(K[i], R[i], t[i], P[i]);
//         }
//
//         triangulatePoints_<T>(nviews, points2d, P, points3d);
//     }
//     else if( method == CV_TRIANG_BY_PLANE )
//     {
//         // Two view
//         if( nviews == 2 )
//         {
//             libmv::Vec3 xl, xr;
//             cv2eigen( points2d.at(0), xl );
//             cv2eigen( points2d.at(1), xr );
//
//             // Fundamental matrix
//             libmv::Mat3 F;
//             libmv::NormalizedEightPointSolver(xl, xr, &F);
//
//             libmv::Mat K1, K2;
//             cv2eigen( K.at(0), K1 );
//             cv2eigen( K.at(1), K2 );
//
//             // Essential matrix
//             libmv::Mat3 E;
//             libmv::EssentialFromFundamental(F, K1, K2, &E);
//
//             // Mat Pl; // [ I | 0 ]
//             Mat Pr;
//             cv::Mat(K[1] * R[1]).copyTo(Pr.colRange(0, 3));
//             cv::Mat(K[1] * t[1]).copyTo(Pr.col(3));
//             libmv::Mat34 P2;
//             cv2eigen( Pr, P2 );
//
//             // triangulation by planes
//             libmv::Vec4 XEuclidean;
//             libmv::TwoViewTriangulationByPlanes(xl, xr, P2, E, &XEuclidean);
//
//             eigen2cv( XEuclidean, points3d );
//         }
//         else if( nviews > 2 )
//         {
// //             CV_ERROR( CV_StsBadArg, "Invalid number of views" );
//         }
//     }
//     else
//     {
// //         CV_ERROR( CV_StsBadArg, "Invalid method" );
//     }
// }


// void
// triangulatePoints(InputArrayOfArrays _points2d, InputArrayOfArrays _K,
//                   InputArrayOfArrays _R, InputArrayOfArrays _t,
//                   OutputArray _points3d, int method)
// {
//     // check
//     unsigned nviews = (unsigned) _points2d.total();
//     CV_Assert(nviews >= 2 && nviews == _R.total() && nviews == _t.total());
//
//     // inputs
//     vector<Mat> points2d, K, R, t;
//     _points2d.getMatVector(points2d);
//     _K.getMatVector(K);
//     _R.getMatVector(R);
//     _t.getMatVector(t);
//
//     // output
//     cv::Mat points3d = _points3d.getMat();
//
//     // type: float or double
//     if( _points2d.getMat(0).depth() == CV_32F )
//     {
//         triangulatePoints_<float>(nviews, points2d, K, R, t, points3d, method);
//     }
//     else
//     {
//         triangulatePoints_<double>(nviews, points2d, K, R, t, points3d, method);
//     }
//
//     points3d.copyTo(_points3d);
// }


} /* namespace cv */
