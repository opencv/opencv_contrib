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
#include <opencv2/sfm/triangulation.hpp>
#include <opencv2/sfm/projection.hpp>

// libmv headers
#include "libmv/multiview/twoviewtriangulation.h"
#include "libmv/multiview/fundamental.h"

using namespace cv;
using namespace std;

namespace cv
{
namespace sfm
{

/** @brief Triangulates the a 3d position between two 2d correspondences, using the DLT.
  @param xl Input vector with first 2d point.
  @param xr Input vector with second 2d point.
  @param Pl Input 3x4 first projection matrix.
  @param Pr Input 3x4 second projection matrix.
  @param objectPoint Output vector with computed 3d point.

  Reference: @cite HartleyZ00 12.2 pag.312
 */
void
triangulateDLT( const Vec2d &xl, const Vec2d &xr,
                const Matx34d &Pl, const Matx34d &Pr,
                Vec3d &point3d )
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

    homogeneousToEuclidean(XHomogeneous, point3d);
}


/** @brief Triangulates the 3d position of 2d correspondences between n images, using the DLT
 * @param x Input vectors of 2d points (the inner vector is per image). Has to be 2xN
 * @param Ps Input vector with 3x4 projections matrices of each image.
 * @param X Output vector with computed 3d point.

 * Reference: it is the standard DLT; for derivation see appendix of Keir's thesis
 */
void
triangulateNViews(const Mat_<double> &x, const std::vector<Matx34d> &Ps, Vec3d &X)
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
    std::vector<Mat_<double> > points2d(nviews);
    std::vector<Matx34d> projection_matrices(nviews);
    {
        std::vector<Mat> points2d_tmp;
        _points2d.getMatVector(points2d_tmp);
        n_points = points2d_tmp[0].cols;

        std::vector<Mat> projection_matrices_tmp;
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
            triangulateNViews( x, projection_matrices, point3d );
            for(char j=0; j<3; ++j)
                points3d.at<double>(j, i) = point3d[j];
        }
    }
}

} /* namespace sfm */
} /* namespace cv */