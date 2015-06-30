// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2010, 2011, 2012 Google Inc. All rights reserved.
// http://code.google.com/p/ceres-solver/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: sameeragarwal@google.com (Sameer Agarwal)

#include "ceres/normal_prior.h"

#include <cstddef>

#include "gtest/gtest.h"
#include "ceres/internal/eigen.h"
#include "ceres/random.h"

namespace ceres {
namespace internal {

void RandomVector(Vector* v) {
  for (int r = 0; r < v->rows(); ++r)
    (*v)[r] = 2 * RandDouble() - 1;
}

void RandomMatrix(Matrix* m) {
  for (int r = 0; r < m->rows(); ++r) {
    for (int c = 0; c < m->cols(); ++c) {
      (*m)(r, c) = 2 * RandDouble() - 1;
    }
  }
}

TEST(NormalPriorTest, ResidualAtRandomPosition) {
  srand(5);

  for (int num_rows = 1; num_rows < 5; ++num_rows) {
    for (int num_cols = 1; num_cols < 5; ++num_cols) {
      Vector b(num_cols);
      RandomVector(&b);

      Matrix A(num_rows, num_cols);
      RandomMatrix(&A);

      double * x = new double[num_cols];
      for (int i = 0; i < num_cols; ++i)
        x[i] = 2 * RandDouble() - 1;

      double * jacobian = new double[num_rows * num_cols];
      Vector residuals(num_rows);

      NormalPrior prior(A, b);
      prior.Evaluate(&x, residuals.data(), &jacobian);

      // Compare the norm of the residual
      double residual_diff_norm =
          (residuals - A * (VectorRef(x, num_cols) - b)).squaredNorm();
      EXPECT_NEAR(residual_diff_norm, 0, 1e-10);

      // Compare the jacobians
      MatrixRef J(jacobian, num_rows, num_cols);
      double jacobian_diff_norm = (J - A).norm();
      EXPECT_NEAR(jacobian_diff_norm, 0.0, 1e-10);

      delete []x;
      delete []jacobian;
    }
  }
}

TEST(NormalPriorTest, ResidualAtRandomPositionNullJacobians) {
  srand(5);

  for (int num_rows = 1; num_rows < 5; ++num_rows) {
    for (int num_cols = 1; num_cols < 5; ++num_cols) {
      Vector b(num_cols);
      RandomVector(&b);

      Matrix A(num_rows, num_cols);
      RandomMatrix(&A);

      double * x = new double[num_cols];
      for (int i = 0; i < num_cols; ++i)
        x[i] = 2 * RandDouble() - 1;

      double* jacobians[1];
      jacobians[0] = NULL;

      Vector residuals(num_rows);

      NormalPrior prior(A, b);
      prior.Evaluate(&x, residuals.data(), jacobians);

      // Compare the norm of the residual
      double residual_diff_norm =
          (residuals - A * (VectorRef(x, num_cols) - b)).squaredNorm();
      EXPECT_NEAR(residual_diff_norm, 0, 1e-10);

      prior.Evaluate(&x, residuals.data(), NULL);
      // Compare the norm of the residual
      residual_diff_norm =
          (residuals - A * (VectorRef(x, num_cols) - b)).squaredNorm();
      EXPECT_NEAR(residual_diff_norm, 0, 1e-10);


      delete []x;
    }
  }
}

}  // namespace internal
}  // namespace ceres
