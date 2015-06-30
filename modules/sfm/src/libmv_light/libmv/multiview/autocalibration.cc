// Copyright (c) 2007, 2008 libmv authors.
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

#include "libmv/multiview/autocalibration.h"

namespace libmv {

void K_From_AbsoluteConic(const Mat3 &W, Mat3 *K) {
  // To compute upper-triangular Cholesky, we flip the indices of the input
  // matrix, compute lower-triangular Cholesky, and then unflip the result.
  Mat3 dual = W.inverse();
  Mat3 flipped_dual;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      flipped_dual(i,j) = dual(2 - i, 2 - j);
    }
  }

  Eigen::LLT<Mat3> llt(flipped_dual);
  Mat3 L = llt.matrixL();

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      (*K)(i,j) = L(2 - i, 2 - j);
    }
  }

  // Resolve sign ambiguities assuming positive diagonal.
  for (int j = 0; j < 3; ++j) {
    if ((*K)(j, j) < 0) {
      for (int i = 0; i < 3; ++i) {
        (*K)(i, j) = -(*K)(i, j);
      }
    }
  }
}

int AutoCalibrationLinear::AddProjection(const Mat34 &P,
                                         double width, double height) {
  Mat34 P_normalized;
  NormalizeProjection(P, width, height, &P_normalized);

  AddProjectionConstraints(P_normalized);

  // Store input
  projections_.push_back(P_normalized);
  widths_.push_back(width);
  heights_.push_back(height);

  return projections_.size() - 1;
}

// TODO(pau): make this generic and move it to numeric.h
static void SortEigenVectors(const Vec &values,
                             const Mat &vectors,
                             Vec *sorted_values,
                             Mat *sorted_vectors) {
  // Compute eigenvalues order.
  std::pair<double, int> order[4];
  for (int i = 0; i < 4; ++i) {
    order[i].first = -values(i);
    order[i].second = i;
  }
  std::sort(order, order + 4);

  sorted_values->resize(4);
  sorted_vectors->resize(4,4);
  for (int i = 0; i < 4; ++i) {
    (*sorted_values)(i) = values[order[i].second];
    sorted_vectors->col(i) = vectors.col(order[i].second);
  }
}

Mat4 AutoCalibrationLinear::MetricTransformation() {
  // Compute the dual absolute quadric, Q.
  Mat A(constraints_.size(), 10);
  for (int i = 0; i < A.rows(); ++i) {
    A.row(i) = constraints_[i];
  }
  Vec q;
  Nullspace(&A, &q);
  Mat4 Q = AbsoluteQuadricMatFromVec(q);
  // TODO(pau) force rank 3.

  // Compute a transformation to a metric frame by decomposing Q.
  Eigen::SelfAdjointEigenSolver<Mat4> eigen_solver(Q);

  // Eigen values should be possitive,
  Vec temp_values = eigen_solver.eigenvalues();
  if (temp_values.sum() < 0) {
    temp_values = -temp_values;
  }

  // and sorted, so that last one is 0.
  Vec eigenvalues;
  Mat eigenvectors;
  SortEigenVectors(temp_values, eigen_solver.eigenvectors(),
                   &eigenvalues, &eigenvectors);

  LOG(INFO) << "Q\n" << Q << "\n";
  LOG(INFO) << "eigen values\n" << eigenvalues << "\n";
  LOG(INFO) << "eigen vectors\n" << eigenvectors << "\n";

  // Compute the transformation from the eigen descomposition.  See last
  // paragraph of page 3 in
  //   "Autocalibration and the absolute quadric" by B. Triggs.
  eigenvalues(3) = 1;
  eigenvalues = eigenvalues.array().sqrt();
  Mat H = eigenvectors * eigenvalues.asDiagonal();
  return H;
}

void AutoCalibrationLinear::AddProjectionConstraints(const Mat34 &P) {
  double nu = 1;

  // Non-extreme focal lenght.
  constraints_.push_back((wc(P, 0, 0) - wc(P, 2, 2)) / 9 / nu);
  constraints_.push_back((wc(P, 1, 1) - wc(P, 2, 2)) / 9 / nu);

  // Aspect ratio is near 1.
  constraints_.push_back((wc(P, 0, 0) - wc(P, 1, 1)) / 0.2 / nu);

  // No skew and principal point near 0,0.
  // Note that there is a typo in the Pollefeys' paper: the 0.01 is not at the
  // correct equation.
  constraints_.push_back(wc(P, 0, 1) / 0.01 / nu);
  constraints_.push_back(wc(P, 0, 2) / 0.1 / nu);
  constraints_.push_back(wc(P, 1, 2) / 0.1 / nu);
}

Vec AutoCalibrationLinear::wc(const Mat34 &P, int i, int j) {
  Vec constraint(10);
  for (int k = 0; k < 10; ++k) {
    Vec q = Vec::Zero(10);
    q(k) = 1;
    Mat4 Q = AbsoluteQuadricMatFromVec(q);

    Mat3 w = P * Q * P.transpose();

    constraint(k) = w(i, j);
  }
  return constraint;
}

Mat4 AutoCalibrationLinear::AbsoluteQuadricMatFromVec(const Vec &q) {
  Mat4 Q;
  Q << q(0), q(1), q(2), q(3),
       q(1), q(4), q(5), q(6),
       q(2), q(5), q(7), q(8),
       q(3), q(6), q(8), q(9);
  return Q;
}

void AutoCalibrationLinear::NormalizeProjection(const Mat34 &P,
                                                double width,
                                                double height,
                                                Mat34 *P_new) {
  Mat3 T;
  T << width + height,              0,  (width - 1) / 2,
                    0, width + height, (height - 1) / 2,
                    0,              0,                1;
  *P_new = T.inverse() * P;
}

void AutoCalibrationLinear::DenormalizeProjection(const Mat34 &P,
                                                  double width,
                                                  double height,
                                                  Mat34 *P_new) {
  Mat3 T;
  T << width + height,              0,  (width - 1) / 2,
                    0, width + height, (height - 1) / 2,
                    0,              0,                1;
  *P_new = T * P;
}


}  // namespace libmv
