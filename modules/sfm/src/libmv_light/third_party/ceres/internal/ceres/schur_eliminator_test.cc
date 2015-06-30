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

#include "ceres/schur_eliminator.h"

#include "Eigen/Dense"
#include "ceres/block_random_access_dense_matrix.h"
#include "ceres/block_sparse_matrix.h"
#include "ceres/casts.h"
#include "ceres/detect_structure.h"
#include "ceres/internal/eigen.h"
#include "ceres/internal/scoped_ptr.h"
#include "ceres/linear_least_squares_problems.h"
#include "ceres/test_util.h"
#include "ceres/triplet_sparse_matrix.h"
#include "ceres/types.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

// TODO(sameeragarwal): Reduce the size of these tests and redo the
// parameterization to be more efficient.

namespace ceres {
namespace internal {

class SchurEliminatorTest : public ::testing::Test {
 protected:
  void SetUpFromId(int id) {
    scoped_ptr<LinearLeastSquaresProblem>
        problem(CreateLinearLeastSquaresProblemFromId(id));
    CHECK_NOTNULL(problem.get());
    SetupHelper(problem.get());
  }

  void SetUpFromFilename(const string& filename) {
    scoped_ptr<LinearLeastSquaresProblem>
        problem(CreateLinearLeastSquaresProblemFromFile(filename));
    CHECK_NOTNULL(problem.get());
    SetupHelper(problem.get());
  }

  void SetupHelper(LinearLeastSquaresProblem* problem) {
    A.reset(down_cast<BlockSparseMatrix*>(problem->A.release()));
    b.reset(problem->b.release());
    D.reset(problem->D.release());

    num_eliminate_blocks = problem->num_eliminate_blocks;
    num_eliminate_cols = 0;
    const CompressedRowBlockStructure* bs = A->block_structure();

    for (int i = 0; i < num_eliminate_blocks; ++i) {
      num_eliminate_cols += bs->cols[i].size;
    }
  }

  // Compute the golden values for the reduced linear system and the
  // solution to the linear least squares problem using dense linear
  // algebra.
  void ComputeReferenceSolution(const Vector& D) {
    Matrix J;
    A->ToDenseMatrix(&J);
    VectorRef f(b.get(), J.rows());

    Matrix H  =  (D.cwiseProduct(D)).asDiagonal();
    H.noalias() += J.transpose() * J;

    const Vector g = J.transpose() * f;
    const int schur_size = J.cols() - num_eliminate_cols;

    lhs_expected.resize(schur_size, schur_size);
    lhs_expected.setZero();

    rhs_expected.resize(schur_size);
    rhs_expected.setZero();

    sol_expected.resize(J.cols());
    sol_expected.setZero();

    Matrix P = H.block(0, 0, num_eliminate_cols, num_eliminate_cols);
    Matrix Q = H.block(0,
                       num_eliminate_cols,
                       num_eliminate_cols,
                       schur_size);
    Matrix R = H.block(num_eliminate_cols,
                       num_eliminate_cols,
                       schur_size,
                       schur_size);
    int row = 0;
    const CompressedRowBlockStructure* bs = A->block_structure();
    for (int i = 0; i < num_eliminate_blocks; ++i) {
      const int block_size =  bs->cols[i].size;
      P.block(row, row,  block_size, block_size) =
          P
          .block(row, row,  block_size, block_size)
          .ldlt()
          .solve(Matrix::Identity(block_size, block_size));
      row += block_size;
    }

    lhs_expected
        .triangularView<Eigen::Upper>() = R - Q.transpose() * P * Q;
    rhs_expected =
        g.tail(schur_size) - Q.transpose() * P * g.head(num_eliminate_cols);
    sol_expected = H.ldlt().solve(g);
  }

  void EliminateSolveAndCompare(const VectorRef& diagonal,
                                bool use_static_structure,
                                const double relative_tolerance) {
    const CompressedRowBlockStructure* bs = A->block_structure();
    const int num_col_blocks = bs->cols.size();
    vector<int> blocks(num_col_blocks - num_eliminate_blocks, 0);
    for (int i = num_eliminate_blocks; i < num_col_blocks; ++i) {
      blocks[i - num_eliminate_blocks] = bs->cols[i].size;
    }

    BlockRandomAccessDenseMatrix lhs(blocks);

    const int num_cols = A->num_cols();
    const int schur_size = lhs.num_rows();

    Vector rhs(schur_size);

    LinearSolver::Options options;
    options.elimination_groups.push_back(num_eliminate_blocks);
    if (use_static_structure) {
      DetectStructure(*bs,
                      num_eliminate_blocks,
                      &options.row_block_size,
                      &options.e_block_size,
                      &options.f_block_size);
    }

    scoped_ptr<SchurEliminatorBase> eliminator;
    eliminator.reset(SchurEliminatorBase::Create(options));
    eliminator->Init(num_eliminate_blocks, A->block_structure());
    eliminator->Eliminate(A.get(), b.get(), diagonal.data(), &lhs, rhs.data());

    MatrixRef lhs_ref(lhs.mutable_values(), lhs.num_rows(), lhs.num_cols());
    Vector reduced_sol  =
        lhs_ref
        .selfadjointView<Eigen::Upper>()
        .ldlt()
        .solve(rhs);

    // Solution to the linear least squares problem.
    Vector sol(num_cols);
    sol.setZero();
    sol.tail(schur_size) = reduced_sol;
    eliminator->BackSubstitute(A.get(),
                               b.get(),
                               diagonal.data(),
                               reduced_sol.data(),
                               sol.data());

    Matrix delta = (lhs_ref - lhs_expected).selfadjointView<Eigen::Upper>();
    double diff = delta.norm();
    EXPECT_NEAR(diff / lhs_expected.norm(), 0.0, relative_tolerance);
    EXPECT_NEAR((rhs - rhs_expected).norm() / rhs_expected.norm(), 0.0,
                relative_tolerance);
    EXPECT_NEAR((sol - sol_expected).norm() / sol_expected.norm(), 0.0,
                relative_tolerance);
  }

  scoped_ptr<BlockSparseMatrix> A;
  scoped_array<double> b;
  scoped_array<double> D;
  int num_eliminate_blocks;
  int num_eliminate_cols;

  Matrix lhs_expected;
  Vector rhs_expected;
  Vector sol_expected;
};

TEST_F(SchurEliminatorTest, ScalarProblem) {
  SetUpFromId(2);
  Vector zero(A->num_cols());
  zero.setZero();

  ComputeReferenceSolution(VectorRef(zero.data(), A->num_cols()));
  EliminateSolveAndCompare(VectorRef(zero.data(), A->num_cols()), true, 1e-14);
  EliminateSolveAndCompare(VectorRef(zero.data(), A->num_cols()), false, 1e-14);

  ComputeReferenceSolution(VectorRef(D.get(), A->num_cols()));
  EliminateSolveAndCompare(VectorRef(D.get(), A->num_cols()), true, 1e-14);
  EliminateSolveAndCompare(VectorRef(D.get(), A->num_cols()), false, 1e-14);
}

#ifndef CERES_NO_PROTOCOL_BUFFERS
TEST_F(SchurEliminatorTest, BlockProblem) {
  const string input_file = TestFileAbsolutePath("problem-6-1384-000.lsqp");

  SetUpFromFilename(input_file);
  ComputeReferenceSolution(VectorRef(D.get(), A->num_cols()));
  EliminateSolveAndCompare(VectorRef(D.get(), A->num_cols()), true, 1e-10);
  EliminateSolveAndCompare(VectorRef(D.get(), A->num_cols()), false, 1e-10);
}
#endif  // CERES_NO_PROTOCOL_BUFFERS

}  // namespace internal
}  // namespace ceres
