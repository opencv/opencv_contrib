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

#include "ceres/casts.h"
#include "ceres/compressed_row_sparse_matrix.h"
#include "ceres/internal/scoped_ptr.h"
#include "ceres/linear_least_squares_problems.h"
#include "ceres/linear_solver.h"
#include "ceres/triplet_sparse_matrix.h"
#include "ceres/types.h"
#include "glog/logging.h"
#include "gtest/gtest.h"


namespace ceres {
namespace internal {

class UnsymmetricLinearSolverTest : public ::testing::Test {
 protected :
  virtual void SetUp() {
    scoped_ptr<LinearLeastSquaresProblem> problem(
        CreateLinearLeastSquaresProblemFromId(0));

    CHECK_NOTNULL(problem.get());
    A_.reset(down_cast<TripletSparseMatrix*>(problem->A.release()));
    b_.reset(problem->b.release());
    D_.reset(problem->D.release());
    sol_unregularized_.reset(problem->x.release());
    sol_regularized_.reset(problem->x_D.release());
  }

  void TestSolver(
      LinearSolverType linear_solver_type,
      SparseLinearAlgebraLibraryType sparse_linear_algebra_library) {
    LinearSolver::Options options;
    options.type = linear_solver_type;
    options.sparse_linear_algebra_library = sparse_linear_algebra_library;
    scoped_ptr<LinearSolver> solver(LinearSolver::Create(options));

    LinearSolver::PerSolveOptions per_solve_options;
    LinearSolver::Summary unregularized_solve_summary;
    LinearSolver::Summary regularized_solve_summary;
    Vector x_unregularized(A_->num_cols());
    Vector x_regularized(A_->num_cols());

    scoped_ptr<SparseMatrix> transformed_A;

    if (linear_solver_type == DENSE_QR ||
        linear_solver_type == DENSE_NORMAL_CHOLESKY) {
      transformed_A.reset(new DenseSparseMatrix(*A_));
    } else if (linear_solver_type == SPARSE_NORMAL_CHOLESKY) {
      transformed_A.reset(new   CompressedRowSparseMatrix(*A_));
    } else {
      LOG(FATAL) << "Unknown linear solver : " << linear_solver_type;
    }
    // Unregularized
    unregularized_solve_summary =
        solver->Solve(transformed_A.get(),
                      b_.get(),
                      per_solve_options,
                      x_unregularized.data());

    // Regularized solution
    per_solve_options.D = D_.get();
    regularized_solve_summary =
        solver->Solve(transformed_A.get(),
                      b_.get(),
                      per_solve_options,
                      x_regularized.data());

    EXPECT_EQ(unregularized_solve_summary.termination_type, TOLERANCE);

    for (int i = 0; i < A_->num_cols(); ++i) {
      EXPECT_NEAR(sol_unregularized_[i], x_unregularized[i], 1e-8);
    }

    EXPECT_EQ(regularized_solve_summary.termination_type, TOLERANCE);
    for (int i = 0; i < A_->num_cols(); ++i) {
      EXPECT_NEAR(sol_regularized_[i], x_regularized[i], 1e-8);
    }
  }

  scoped_ptr<TripletSparseMatrix> A_;
  scoped_array<double> b_;
  scoped_array<double> D_;
  scoped_array<double> sol_unregularized_;
  scoped_array<double> sol_regularized_;
};

TEST_F(UnsymmetricLinearSolverTest, DenseQR) {
  TestSolver(DENSE_QR, SUITE_SPARSE);
}

TEST_F(UnsymmetricLinearSolverTest, DenseNormalCholesky) {
  TestSolver(DENSE_NORMAL_CHOLESKY, SUITE_SPARSE);
}

#ifndef CERES_NO_SUITESPARSE
TEST_F(UnsymmetricLinearSolverTest, SparseNormalCholeskyUsingSuiteSparse) {
  TestSolver(SPARSE_NORMAL_CHOLESKY, SUITE_SPARSE);
}
#endif

#ifndef CERES_NO_CXSPARSE
TEST_F(UnsymmetricLinearSolverTest, SparseNormalCholeskyUsingCXSparse) {
  TestSolver(SPARSE_NORMAL_CHOLESKY, CX_SPARSE);
}
#endif

}  // namespace internal
}  // namespace ceres
