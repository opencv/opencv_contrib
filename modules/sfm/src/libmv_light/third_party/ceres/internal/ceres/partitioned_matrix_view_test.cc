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

#include "ceres/partitioned_matrix_view.h"

#include <vector>
#include "ceres/block_structure.h"
#include "ceres/casts.h"
#include "ceres/internal/eigen.h"
#include "ceres/internal/scoped_ptr.h"
#include "ceres/linear_least_squares_problems.h"
#include "ceres/random.h"
#include "ceres/sparse_matrix.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

const double kEpsilon = 1e-14;

class PartitionedMatrixViewTest : public ::testing::Test {
 protected :
  virtual void SetUp() {
    scoped_ptr<LinearLeastSquaresProblem> problem(
        CreateLinearLeastSquaresProblemFromId(2));
    CHECK_NOTNULL(problem.get());
    A_.reset(problem->A.release());

    num_cols_ = A_->num_cols();
    num_rows_ = A_->num_rows();
    num_eliminate_blocks_ = problem->num_eliminate_blocks;
  }

  int num_rows_;
  int num_cols_;
  int num_eliminate_blocks_;

  scoped_ptr<SparseMatrix> A_;
};

TEST_F(PartitionedMatrixViewTest, DimensionsTest) {
  PartitionedMatrixView m(*down_cast<BlockSparseMatrix*>(A_.get()),
                          num_eliminate_blocks_);
  EXPECT_EQ(m.num_col_blocks_e(), num_eliminate_blocks_);
  EXPECT_EQ(m.num_col_blocks_f(), num_cols_ - num_eliminate_blocks_);
  EXPECT_EQ(m.num_cols_e(), num_eliminate_blocks_);
  EXPECT_EQ(m.num_cols_f(), num_cols_ - num_eliminate_blocks_);
  EXPECT_EQ(m.num_cols(), A_->num_cols());
  EXPECT_EQ(m.num_rows(), A_->num_rows());
}

TEST_F(PartitionedMatrixViewTest, RightMultiplyE) {
  PartitionedMatrixView m(*down_cast<BlockSparseMatrix*>(A_.get()),
                          num_eliminate_blocks_);

  srand(5);

  Vector x1(m.num_cols_e());
  Vector x2(m.num_cols());
  x2.setZero();

  for (int i = 0; i < m.num_cols_e(); ++i) {
    x1(i) = x2(i) = RandDouble();
  }

  Vector y1 = Vector::Zero(m.num_rows());
  m.RightMultiplyE(x1.data(), y1.data());

  Vector y2 = Vector::Zero(m.num_rows());
  A_->RightMultiply(x2.data(), y2.data());

  for (int i = 0; i < m.num_rows(); ++i) {
    EXPECT_NEAR(y1(i), y2(i), kEpsilon);
  }
}

TEST_F(PartitionedMatrixViewTest, RightMultiplyF) {
  PartitionedMatrixView m(*down_cast<BlockSparseMatrix*>(A_.get()),
                          num_eliminate_blocks_);

  srand(5);

  Vector x1(m.num_cols_f());
  Vector x2 = Vector::Zero(m.num_cols());

  for (int i = 0; i < m.num_cols_f(); ++i) {
    x1(i) = RandDouble();
    x2(i + m.num_cols_e()) = x1(i);
  }

  Vector y1 = Vector::Zero(m.num_rows());
  m.RightMultiplyF(x1.data(), y1.data());

  Vector y2 = Vector::Zero(m.num_rows());
  A_->RightMultiply(x2.data(), y2.data());

  for (int i = 0; i < m.num_rows(); ++i) {
    EXPECT_NEAR(y1(i), y2(i), kEpsilon);
  }
}

TEST_F(PartitionedMatrixViewTest, LeftMultiply) {
  PartitionedMatrixView m(*down_cast<BlockSparseMatrix*>(A_.get()),
                          num_eliminate_blocks_);

  srand(5);

  Vector x = Vector::Zero(m.num_rows());
  for (int i = 0; i < m.num_rows(); ++i) {
    x(i) = RandDouble();
  }

  Vector y = Vector::Zero(m.num_cols());
  Vector y1 = Vector::Zero(m.num_cols_e());
  Vector y2 = Vector::Zero(m.num_cols_f());

  A_->LeftMultiply(x.data(), y.data());
  m.LeftMultiplyE(x.data(), y1.data());
  m.LeftMultiplyF(x.data(), y2.data());

  for (int i = 0; i < m.num_cols(); ++i) {
    EXPECT_NEAR(y(i),
                (i < m.num_cols_e()) ? y1(i) : y2(i - m.num_cols_e()),
                kEpsilon);
  }
}

TEST_F(PartitionedMatrixViewTest, BlockDiagonalEtE) {
  PartitionedMatrixView m(*down_cast<BlockSparseMatrix*>(A_.get()),
                          num_eliminate_blocks_);

  scoped_ptr<BlockSparseMatrix>
      block_diagonal_ee(m.CreateBlockDiagonalEtE());
  const CompressedRowBlockStructure* bs  = block_diagonal_ee->block_structure();

  EXPECT_EQ(block_diagonal_ee->num_rows(), 2);
  EXPECT_EQ(block_diagonal_ee->num_cols(), 2);
  EXPECT_EQ(bs->cols.size(), 2);
  EXPECT_EQ(bs->rows.size(), 2);

  EXPECT_NEAR(block_diagonal_ee->values()[0], 10.0, kEpsilon);
  EXPECT_NEAR(block_diagonal_ee->values()[1], 155.0, kEpsilon);
}

TEST_F(PartitionedMatrixViewTest, BlockDiagonalFtF) {
  PartitionedMatrixView m(*down_cast<BlockSparseMatrix*>(A_.get()),
                          num_eliminate_blocks_);

  scoped_ptr<BlockSparseMatrix>
      block_diagonal_ff(m.CreateBlockDiagonalFtF());
  const CompressedRowBlockStructure* bs  = block_diagonal_ff->block_structure();

  EXPECT_EQ(block_diagonal_ff->num_rows(), 3);
  EXPECT_EQ(block_diagonal_ff->num_cols(), 3);
  EXPECT_EQ(bs->cols.size(), 3);
  EXPECT_EQ(bs->rows.size(), 3);
  EXPECT_NEAR(block_diagonal_ff->values()[0], 70.0, kEpsilon);
  EXPECT_NEAR(block_diagonal_ff->values()[1], 17.0, kEpsilon);
  EXPECT_NEAR(block_diagonal_ff->values()[2], 37.0, kEpsilon);
}

}  // namespace internal
}  // namespace ceres
