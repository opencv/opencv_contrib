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

#include <cstddef>
#include "ceres/block_sparse_matrix.h"
#include "ceres/block_structure.h"
#include "ceres/casts.h"
#include "ceres/internal/scoped_ptr.h"
#include "ceres/linear_least_squares_problems.h"
#include "ceres/linear_solver.h"
#include "ceres/schur_complement_solver.h"
#include "ceres/triplet_sparse_matrix.h"
#include "ceres/types.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

class SchurComplementSolverTest : public ::testing::Test {
 protected:
  void SetUpFromProblemId(int problem_id) {
    scoped_ptr<LinearLeastSquaresProblem> problem(
        CreateLinearLeastSquaresProblemFromId(problem_id));

    CHECK_NOTNULL(problem.get());
    A.reset(down_cast<BlockSparseMatrix*>(problem->A.release()));
    b.reset(problem->b.release());
    D.reset(problem->D.release());

    num_cols = A->num_cols();
    num_rows = A->num_rows();
    num_eliminate_blocks = problem->num_eliminate_blocks;

    x.reset(new double[num_cols]);
    sol.reset(new double[num_cols]);
    sol_d.reset(new double[num_cols]);

    LinearSolver::Options options;
    options.type = DENSE_QR;

    scoped_ptr<LinearSolver> qr(LinearSolver::Create(options));

    TripletSparseMatrix triplet_A(A->num_rows(),
                                  A->num_cols(),
                                  A->num_nonzeros());
    A->ToTripletSparseMatrix(&triplet_A);

    // Gold standard solutions using dense QR factorization.
    DenseSparseMatrix dense_A(triplet_A);
    qr->Solve(&dense_A, b.get(), LinearSolver::PerSolveOptions(), sol.get());

    // Gold standard solution with appended diagonal.
    LinearSolver::PerSolveOptions per_solve_options;
    per_solve_options.D = D.get();
    qr->Solve(&dense_A, b.get(), per_solve_options, sol_d.get());
  }

  void ComputeAndCompareSolutions(
      int problem_id,
      bool regularization,
      ceres::LinearSolverType linear_solver_type,
      ceres::SparseLinearAlgebraLibraryType sparse_linear_algebra_library) {
    SetUpFromProblemId(problem_id);
    LinearSolver::Options options;
    options.elimination_groups.push_back(num_eliminate_blocks);
    options.elimination_groups.push_back(
        A->block_structure()->cols.size() - num_eliminate_blocks);
    options.type = linear_solver_type;
    options.sparse_linear_algebra_library = sparse_linear_algebra_library;

    scoped_ptr<LinearSolver> solver(LinearSolver::Create(options));

    LinearSolver::PerSolveOptions per_solve_options;
    LinearSolver::Summary summary;
    if (regularization) {
      per_solve_options.D = D.get();
    }

    summary = solver->Solve(A.get(), b.get(), per_solve_options, x.get());

    if (regularization) {
      for (int i = 0; i < num_cols; ++i) {
        ASSERT_NEAR(sol_d.get()[i], x[i], 1e-10);
      }
    } else {
      for (int i = 0; i < num_cols; ++i) {
        ASSERT_NEAR(sol.get()[i], x[i], 1e-10);
      }
    }
  }

  int num_rows;
  int num_cols;
  int num_eliminate_blocks;

  scoped_ptr<BlockSparseMatrix> A;
  scoped_array<double> b;
  scoped_array<double> x;
  scoped_array<double> D;
  scoped_array<double> sol;
  scoped_array<double> sol_d;
};

#ifndef CERES_NO_SUITESPARSE
TEST_F(SchurComplementSolverTest, SparseSchurWithSuiteSparse) {
  ComputeAndCompareSolutions(2, false, SPARSE_SCHUR, SUITE_SPARSE);
  ComputeAndCompareSolutions(3, false, SPARSE_SCHUR, SUITE_SPARSE);
  ComputeAndCompareSolutions(2, true, SPARSE_SCHUR, SUITE_SPARSE);
  ComputeAndCompareSolutions(3, true, SPARSE_SCHUR, SUITE_SPARSE);
}
#endif  // CERES_NO_SUITESPARSE

#ifndef CERES_NO_CXSPARSE
TEST_F(SchurComplementSolverTest, SparseSchurWithCXSparse) {
  ComputeAndCompareSolutions(2, false, SPARSE_SCHUR, CX_SPARSE);
  ComputeAndCompareSolutions(3, false, SPARSE_SCHUR, CX_SPARSE);
  ComputeAndCompareSolutions(2, true, SPARSE_SCHUR, CX_SPARSE);
  ComputeAndCompareSolutions(3, true, SPARSE_SCHUR, CX_SPARSE);
}
#endif  // CERES_NO_CXSPARSE

TEST_F(SchurComplementSolverTest, DenseSchur) {
  // The sparse linear algebra library type is ignored for
  // DENSE_SCHUR.
  ComputeAndCompareSolutions(2, false, DENSE_SCHUR, SUITE_SPARSE);
  ComputeAndCompareSolutions(3, false, DENSE_SCHUR, SUITE_SPARSE);
  ComputeAndCompareSolutions(2, true, DENSE_SCHUR, SUITE_SPARSE);
  ComputeAndCompareSolutions(3, true, DENSE_SCHUR, SUITE_SPARSE);
}

}  // namespace internal
}  // namespace ceres
