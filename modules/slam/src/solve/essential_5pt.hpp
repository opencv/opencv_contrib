// Copyright (c) 2011 libmv authors.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

#ifndef SLAM_SOLVE_ESSENTIAL_5PT_H
#define SLAM_SOLVE_ESSENTIAL_5PT_H

#include <Eigen/Dense>

#include "type.hpp"

/**
 * This file contains helper functions to compute the essential matrix from 5 bearing vector correspondences.
 * The code is adapted from the libmv implementation (MIT license cited above) of Stewenius's et. al's algorithm from
 * "Recent developments on direct relative orientation".
 *
 */

namespace cv::slam {

// some special sized eigen matrices we only use here
using Mat10_t = Eigen::Matrix<double, 10, 10>;

using Vec20_t = Eigen::Matrix<double, 1, 20>;

// Polynomial coefficients
// slightly different from the paper, which seems to have a typo here
enum {
    coeff_xxx,
    coeff_xxy,
    coeff_xyy,
    coeff_yyy,
    coeff_xxz,
    coeff_xyz,
    coeff_yyz,
    coeff_xzz,
    coeff_yzz,
    coeff_zzz,
    coeff_xx,
    coeff_xy,
    coeff_yy,
    coeff_xz,
    coeff_yz,
    coeff_zz,
    coeff_x,
    coeff_y,
    coeff_z,
    coeff_1
};

MatX_t find_nullspace_of_epipolar_constraint(const eigen_alloc_vector<Vec3_t>& x1, const eigen_alloc_vector<Vec3_t>& x2, bool& success) {
    Eigen::Matrix<double, 9, 9> epipolar_constraint = Eigen::Matrix<double, 9, 9>::Constant(0.0);
    // form the epipolar constraint from the bearing vectors
    for (size_t i = 0; i < x1.size(); ++i) {
        epipolar_constraint.row(i) << x2.at(i)(0) * x1.at(i).transpose(),
            x2.at(i)(1) * x1.at(i).transpose(),
            x2.at(i)(2) * x1.at(i).transpose();
    }

    // Use LU decomposition in the minimal case and SVD in the non-minimal case
    success = false;
    MatX_t null_space;
    if (x1.size() == 5) {
        const Eigen::FullPivLU<MatX_t> lu(epipolar_constraint);
        success = (lu.dimensionOfKernel() >= 4);
        null_space = lu.kernel();
    }
    else {
        const Eigen::JacobiSVD<MatX_t> svd(
            epipolar_constraint.transpose() * epipolar_constraint,
            Eigen::ComputeFullV);
        null_space = svd.matrixV().rightCols<4>();
        success = true;
    }
    return null_space;
}

/**
 * Multiply two degree one polynomials of variables x, y, z.
 * using GrLex order
 * [... xx xy yy xz yz zz x y z 1]
 */
Vec20_t deg_one_poly_product(const Vec20_t& poly1, const Vec20_t& poly2) {
    Vec20_t product = Vec20_t::Zero();

    product(coeff_xx) = poly1(coeff_x) * poly2(coeff_x); // x*x'
    product(coeff_xy)
        = poly1(coeff_x) * poly2(coeff_y) + poly1(coeff_y) * poly2(coeff_x);               // x*y' + y*x'
    product(coeff_xz) = poly1(coeff_x) * poly2(coeff_z) + poly1(coeff_z) * poly2(coeff_x); // x*z' + z*x'
    product(coeff_yy) = poly1(coeff_y) * poly2(coeff_y);                                   // y * y'
    product(coeff_yz) = poly1(coeff_y) * poly2(coeff_z) + poly1(coeff_z) * poly2(coeff_y); // y*z' + z * y'
    product(coeff_zz) = poly1(coeff_z) * poly2(coeff_z);                                   // z * z'
    product(coeff_x) = poly1(coeff_x) * poly2(coeff_1) + poly1(coeff_1) * poly2(coeff_x);  // x * c' + c * x'
    product(coeff_y) = poly1(coeff_y) * poly2(coeff_1) + poly1(coeff_1) * poly2(coeff_y);  // y * c' + c * y'
    product(coeff_z) = poly1(coeff_z) * poly2(coeff_1) + poly1(coeff_1) * poly2(coeff_z);  // z * c' + c * z'
    product(coeff_1) = poly1(coeff_1) * poly2(coeff_1);                                    // c * c'

    return product;
}

/**
 * Multiply a 2 deg poly, poly1 and a one deg poly, poly2 (in x, y, z) using GrLex order
 * [xxx xxy xyy yyy xxz xyz yyz xzz yzz zzz xx xy yy xz yz zz x y z 1]
 */
Vec20_t deg_two_poly_product(const VecX_t& poly1, const VecX_t& poly2) {
    Vec20_t product(20);

    product(coeff_xxx) = poly1(coeff_xx) * poly2(coeff_x);
    product(coeff_xxy) = poly1(coeff_xx) * poly2(coeff_y)
                         + poly1(coeff_xy) * poly2(coeff_x);
    product(coeff_xxz) = poly1(coeff_xx) * poly2(coeff_z)
                         + poly1(coeff_xz) * poly2(coeff_x);
    product(coeff_xyy) = poly1(coeff_xy) * poly2(coeff_y)
                         + poly1(coeff_yy) * poly2(coeff_x);
    product(coeff_xyz) = poly1(coeff_xy) * poly2(coeff_z)
                         + poly1(coeff_yz) * poly2(coeff_x)
                         + poly1(coeff_xz) * poly2(coeff_y);
    product(coeff_xzz) = poly1(coeff_xz) * poly2(coeff_z)
                         + poly1(coeff_zz) * poly2(coeff_x);
    product(coeff_yyy) = poly1(coeff_yy) * poly2(coeff_y);
    product(coeff_yyz) = poly1(coeff_yy) * poly2(coeff_z)
                         + poly1(coeff_yz) * poly2(coeff_y);
    product(coeff_yzz) = poly1(coeff_yz) * poly2(coeff_z)
                         + poly1(coeff_zz) * poly2(coeff_y);
    product(coeff_zzz) = poly1(coeff_zz) * poly2(coeff_z);
    product(coeff_xx) = poly1(coeff_xx) * poly2(coeff_1)
                        + poly1(coeff_x) * poly2(coeff_x);
    product(coeff_xy) = poly1(coeff_xy) * poly2(coeff_1)
                        + poly1(coeff_x) * poly2(coeff_y)
                        + poly1(coeff_y) * poly2(coeff_x);
    product(coeff_xz) = poly1(coeff_xz) * poly2(coeff_1)
                        + poly1(coeff_x) * poly2(coeff_z)
                        + poly1(coeff_z) * poly2(coeff_x);
    product(coeff_yy) = poly1(coeff_yy) * poly2(coeff_1)
                        + poly1(coeff_y) * poly2(coeff_y);
    product(coeff_yz) = poly1(coeff_yz) * poly2(coeff_1)
                        + poly1(coeff_y) * poly2(coeff_z)
                        + poly1(coeff_z) * poly2(coeff_y);
    product(coeff_zz) = poly1(coeff_zz) * poly2(coeff_1)
                        + poly1(coeff_z) * poly2(coeff_z);
    product(coeff_x) = poly1(coeff_x) * poly2(coeff_1)
                       + poly1(coeff_1) * poly2(coeff_x);
    product(coeff_y) = poly1(coeff_y) * poly2(coeff_1)
                       + poly1(coeff_1) * poly2(coeff_y);
    product(coeff_z) = poly1(coeff_z) * poly2(coeff_1)
                       + poly1(coeff_1) * poly2(coeff_z);
    product(coeff_1) = poly1(coeff_1) * poly2(coeff_1);

    return product;
}

Eigen::Matrix<double, 10, 20> form_polynomial_constraint_matrix(const Eigen::Matrix<double, 9, 4>& E_basis) {
    // Build the polynomial form of E
    VecX_t E[3][3];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            E[i][j] = VecX_t::Zero(20);
            E[i][j](coeff_x) = E_basis(3 * i + j, 0);
            E[i][j](coeff_y) = E_basis(3 * i + j, 1);
            E[i][j](coeff_z) = E_basis(3 * i + j, 2);
            E[i][j](coeff_1) = E_basis(3 * i + j, 3);
        }
    }

    // The constraint matrix we want to construct here.
    Eigen::Matrix<double, 10, 20> M;
    int mrow = 0;

    // Theorem 1: Determinant constraint det(E) = 0 is the first part of M
    M.row(mrow++) = deg_two_poly_product(deg_one_poly_product(E[0][1], E[1][2]) - deg_one_poly_product(E[0][2], E[1][1]), E[2][0])
                    + deg_two_poly_product(deg_one_poly_product(E[0][2], E[1][0]) - deg_one_poly_product(E[0][0], E[1][2]), E[2][1])
                    + deg_two_poly_product(deg_one_poly_product(E[0][0], E[1][1]) - deg_one_poly_product(E[0][1], E[1][0]), E[2][2]);

    // Theorem 2: the trace constraint: EEtE - 1/2 trace(EEt)E = 0
    // is the rest of M

    // EEt == E * E.transpose()
    Vec20_t EET[3][3];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (i <= j) {
                EET[i][j] = deg_one_poly_product(E[i][0], E[j][0])
                            + deg_one_poly_product(E[i][1], E[j][1])
                            + deg_one_poly_product(E[i][2], E[j][2]);
            }
            else {
                // EET is symmetric
                EET[i][j] = EET[j][i];
            }
        }
    }

    // EEt - 1/2 trace(EEt)
    Vec20_t(&trace_constraint)[3][3] = EET;
    const Vec20_t trace = 0.5 * (EET[0][0] + EET[1][1] + EET[2][2]);
    for (int i = 0; i < 3; ++i) {
        trace_constraint[i][i] -= trace;
    }

    // (EEt - 1/2 trace(EEt)) * E --> EEtE - 1/2 trace(EEt)E = 0
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            M.row(mrow++) = deg_two_poly_product(trace_constraint[i][0], E[0][j])
                            + deg_two_poly_product(trace_constraint[i][1], E[1][j])
                            + deg_two_poly_product(trace_constraint[i][2], E[2][j]);
        }
    }

    return M;
}

} // namespace cv::slam

#endif // SLAM_SOLVE_ESSENTIAL_5PT_H
