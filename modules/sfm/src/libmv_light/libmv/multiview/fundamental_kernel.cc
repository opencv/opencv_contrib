// Copyright (c) 2009 libmv authors.
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

#include <cstdio>

// TODO(keir): This code is plain unfinished! Doesn't even compile!

#include "libmv/base/vector.h"
#include "libmv/multiview/fundamental_kernel.h"
#include "libmv/numeric/numeric.h"
#include "libmv/numeric/poly.h"
#include "libmv/logging/logging.h"

namespace libmv {
namespace fundamental {
namespace kernel {

void SevenPointSolver::Solve(const Mat &x1, const Mat &x2, vector<Mat3> *F) {
  assert(2 == x1.rows());
  assert(7 <= x1.cols());
  assert(x1.rows() == x2.rows());
  assert(x1.cols() == x2.cols());

  // Set up the homogeneous system Af = 0 from the equations x'T*F*x = 0.
  MatX9 A(x1.cols(), 9);
  EncodeEpipolarEquation(x1, x2, &A);

  // Find the two F matrices in the nullspace of A.
  Vec9 f1, f2;
  Nullspace2(&A, &f1, &f2);
  Mat3 F1 = Map<RMat3>(f1.data());
  Mat3 F2 = Map<RMat3>(f2.data());

  // Then, use the condition det(F) = 0 to determine F. In other words, solve
  // det(F1 + a*F2) = 0 for a.
  double a = F1(0, 0), j = F2(0, 0),
         b = F1(0, 1), k = F2(0, 1),
         c = F1(0, 2), l = F2(0, 2),
         d = F1(1, 0), m = F2(1, 0),
         e = F1(1, 1), n = F2(1, 1),
         f = F1(1, 2), o = F2(1, 2),
         g = F1(2, 0), p = F2(2, 0),
         h = F1(2, 1), q = F2(2, 1),
         i = F1(2, 2), r = F2(2, 2);

  // Run fundamental_7point_coeffs.py to get the below coefficients.
  // The coefficients are in ascending powers of alpha, i.e. P[N]*x^N.
  double P[4] = {
    a*e*i + b*f*g + c*d*h - a*f*h - b*d*i - c*e*g,
    a*e*r + a*i*n + b*f*p + b*g*o + c*d*q + c*h*m + d*h*l + e*i*j + f*g*k -
    a*f*q - a*h*o - b*d*r - b*i*m - c*e*p - c*g*n - d*i*k - e*g*l - f*h*j,
    a*n*r + b*o*p + c*m*q + d*l*q + e*j*r + f*k*p + g*k*o + h*l*m + i*j*n -
    a*o*q - b*m*r - c*n*p - d*k*r - e*l*p - f*j*q - g*l*n - h*j*o - i*k*m,
    j*n*r + k*o*p + l*m*q - j*o*q - k*m*r - l*n*p,
  };

  // Solve for the roots of P[3]*x^3 + P[2]*x^2 + P[1]*x + P[0] = 0.
  double roots[3];
  int num_roots = SolveCubicPolynomial(P, roots);

  // Build the fundamental matrix for each solution.
  for (int kk = 0; kk < num_roots; ++kk)  {
    F->push_back(F1 + roots[kk] * F2);
  }
}

//template<bool force_essential_constraint>
void EightPointSolver::Solve(const Mat &x1, const Mat &x2, vector<Mat3> *Fs) {
  assert(2 == x1.rows());
  assert(8 <= x1.cols());
  assert(x1.rows() == x2.rows());
  assert(x1.cols() == x2.cols());

  MatX9 A(x1.cols(), 9);
  EncodeEpipolarEquation(x1, x2, &A);

  Vec9 f;
  Nullspace(&A, &f);
  Mat3 F = Map<RMat3>(f.data());

  // Force the fundamental property if the A matrix has full rank.
  if (x1.cols() > 8) {
    Eigen::JacobiSVD<Mat3> USV(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Vec3 d = USV.singularValues();
    d[2] = 0.0;
    F = USV.matrixU() * d.asDiagonal() * USV.matrixV().transpose();
  }
  Fs->push_back(F);
}

}  // namespace kernel
}  // namespace fundamental
}  // namespace libmv
