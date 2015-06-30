// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2013 Google Inc. All rights reserved.
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
// Author: keir@google.com (Keir Mierle)
//         sameeragarwal@google.com (Sameer Agarwal)

#include "ceres/evaluator_test_utils.h"
#include "ceres/internal/eigen.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

void CompareEvaluations(int expected_num_rows,
                        int expected_num_cols,
                        double expected_cost,
                        const double* expected_residuals,
                        const double* expected_gradient,
                        const double* expected_jacobian,
                        const double actual_cost,
                        const double* actual_residuals,
                        const double* actual_gradient,
                        const double* actual_jacobian) {
  EXPECT_EQ(expected_cost, actual_cost);

  if (expected_residuals != NULL) {
    ConstVectorRef expected_residuals_vector(expected_residuals,
                                             expected_num_rows);
    ConstVectorRef actual_residuals_vector(actual_residuals,
                                           expected_num_rows);
    EXPECT_TRUE((actual_residuals_vector.array() ==
                 expected_residuals_vector.array()).all())
        << "Actual:\n" << actual_residuals_vector
        << "\nExpected:\n" << expected_residuals_vector;
  }

  if (expected_gradient != NULL) {
    ConstVectorRef expected_gradient_vector(expected_gradient,
                                            expected_num_cols);
    ConstVectorRef actual_gradient_vector(actual_gradient,
                                            expected_num_cols);

    EXPECT_TRUE((actual_gradient_vector.array() ==
                 expected_gradient_vector.array()).all())
        << "Actual:\n" << actual_gradient_vector.transpose()
        << "\nExpected:\n" << expected_gradient_vector.transpose();
  }

  if (expected_jacobian != NULL) {
    ConstMatrixRef expected_jacobian_matrix(expected_jacobian,
                                            expected_num_rows,
                                            expected_num_cols);
    ConstMatrixRef actual_jacobian_matrix(actual_jacobian,
                                          expected_num_rows,
                                          expected_num_cols);
    EXPECT_TRUE((actual_jacobian_matrix.array() ==
                 expected_jacobian_matrix.array()).all())
        << "Actual:\n" << actual_jacobian_matrix
        << "\nExpected:\n" << expected_jacobian_matrix;
  }
}

}  // namespace internal
}  // namespace ceres
