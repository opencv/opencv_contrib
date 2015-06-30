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
//
// TODO(sameeragarwal): Add support for larger, more complicated and
// poorly conditioned problems both for correctness testing as well as
// benchmarking.

#include "ceres/iterative_schur_complement_solver.h"

#include <cstddef>
#include "Eigen/Dense"
#include "ceres/block_random_access_dense_matrix.h"
#include "ceres/block_sparse_matrix.h"
#include "ceres/casts.h"
#include "ceres/internal/eigen.h"
#include "ceres/internal/scoped_ptr.h"
#include "ceres/linear_least_squares_problems.h"
#include "ceres/linear_solver.h"
#include "ceres/schur_eliminator.h"
#include "ceres/triplet_sparse_matrix.h"
#include "ceres/types.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

using testing::AssertionResult;

const double kEpsilon = 1e-14;

class IterativeSchurComplementSolverTest : public ::testing::Test {
 protected :
  virtual void SetUp() {
    scoped_ptr<LinearLeastSquaresProblem> problem(
        CreateLinearLeastSquaresProblemFromId(2));

    CHECK_NOTNULL(problem.get());
    A_.reset(down_cast<BlockSparseMatrix*>(problem->A.release()));
    b_.reset(problem->b.release());
    D_.reset(problem->D.release());

    num_cols_ = A_->num_cols();
    num_rows_ = A_->num_rows();
    num_eliminate_blocks_ = problem->num_eliminate_blocks;
  }

  AssertionResult TestSolver(double* D) {
    TripletSparseMatrix triplet_A(A_->num_rows(),
                                  A_->num_cols(),
                                  A_->num_nonzeros());
    A_->ToTripletSparseMatrix(&triplet_A);

    DenseSparseMatrix dense_A(triplet_A);

    LinearSolver::Options options;
    options.type = DENSE_QR;
    scoped_ptr<LinearSolver> qr(LinearSolver::Create(options));

    LinearSolver::PerSolveOptions per_solve_options;
    per_solve_options.D = D;
    Vector reference_solution(num_cols_);
    qr->Solve(&dense_A, b_.get(), per_solve_options, reference_solution.data());

    options.elimination_groups.push_back(num_eliminate_blocks_);
    options.max_num_iterations = num_cols_;
    IterativeSchurComplementSolver isc(options);

    Vector isc_sol(num_cols_);
    per_solve_options.r_tolerance  = 1e-12;
    isc.Solve(A_.get(), b_.get(), per_solve_options, isc_sol.data());
    double diff = (isc_sol - reference_solution).norm();
    if (diff < kEpsilon) {
      return testing::AssertionSuccess();
    } else {
      return testing::AssertionFailure()
          << "The reference solution differs from the ITERATIVE_SCHUR"
          << " solution by " << diff << " which is more than " << kEpsilon;
    }
  }

  int num_rows_;
  int num_cols_;
  int num_eliminate_blocks_;
  scoped_ptr<BlockSparseMatrix> A_;
  scoped_array<double> b_;
  scoped_array<double> D_;
};

TEST_F(IterativeSchurComplementSolverTest, SolverTest) {
  EXPECT_TRUE(TestSolver(NULL));
  EXPECT_TRUE(TestSolver(D_.get()));
}

}  // namespace internal
}  // namespace ceres
