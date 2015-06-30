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

#include "ceres/block_sparse_matrix.h"

#include <string>
#include "ceres/casts.h"
#include "ceres/internal/eigen.h"
#include "ceres/internal/scoped_ptr.h"
#include "ceres/linear_least_squares_problems.h"
#include "ceres/matrix_proto.h"
#include "ceres/triplet_sparse_matrix.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

class BlockSparseMatrixTest : public ::testing::Test {
 protected :
  virtual void SetUp() {
    scoped_ptr<LinearLeastSquaresProblem> problem(
        CreateLinearLeastSquaresProblemFromId(2));
    CHECK_NOTNULL(problem.get());
    A_.reset(down_cast<BlockSparseMatrix*>(problem->A.release()));

    problem.reset(CreateLinearLeastSquaresProblemFromId(1));
    CHECK_NOTNULL(problem.get());
    B_.reset(down_cast<TripletSparseMatrix*>(problem->A.release()));

    CHECK_EQ(A_->num_rows(), B_->num_rows());
    CHECK_EQ(A_->num_cols(), B_->num_cols());
    CHECK_EQ(A_->num_nonzeros(), B_->num_nonzeros());
  }

  scoped_ptr<BlockSparseMatrix> A_;
  scoped_ptr<TripletSparseMatrix> B_;
};

TEST_F(BlockSparseMatrixTest, SetZeroTest) {
  A_->SetZero();
  EXPECT_EQ(13, A_->num_nonzeros());
}

TEST_F(BlockSparseMatrixTest, RightMultiplyTest) {
  Vector y_a = Vector::Zero(A_->num_rows());
  Vector y_b = Vector::Zero(A_->num_rows());
  for (int i = 0; i < A_->num_cols(); ++i) {
    Vector x = Vector::Zero(A_->num_cols());
    x[i] = 1.0;
    A_->RightMultiply(x.data(), y_a.data());
    B_->RightMultiply(x.data(), y_b.data());
    EXPECT_LT((y_a - y_b).norm(), 1e-12);
  }
}

TEST_F(BlockSparseMatrixTest, LeftMultiplyTest) {
  Vector y_a = Vector::Zero(A_->num_cols());
  Vector y_b = Vector::Zero(A_->num_cols());
  for (int i = 0; i < A_->num_rows(); ++i) {
    Vector x = Vector::Zero(A_->num_rows());
    x[i] = 1.0;
    A_->LeftMultiply(x.data(), y_a.data());
    B_->LeftMultiply(x.data(), y_b.data());
    EXPECT_LT((y_a - y_b).norm(), 1e-12);
  }
}

TEST_F(BlockSparseMatrixTest, SquaredColumnNormTest) {
  Vector y_a = Vector::Zero(A_->num_cols());
  Vector y_b = Vector::Zero(A_->num_cols());
  A_->SquaredColumnNorm(y_a.data());
  B_->SquaredColumnNorm(y_b.data());
  EXPECT_LT((y_a - y_b).norm(), 1e-12);
}

TEST_F(BlockSparseMatrixTest, ToDenseMatrixTest) {
  Matrix m_a;
  Matrix m_b;
  A_->ToDenseMatrix(&m_a);
  B_->ToDenseMatrix(&m_b);
  EXPECT_LT((m_a - m_b).norm(), 1e-12);
}

#ifndef CERES_NO_PROTOCOL_BUFFERS
TEST_F(BlockSparseMatrixTest, Serialization) {
  // Roundtrip through serialization and check for equality.
  SparseMatrixProto proto;
  A_->ToProto(&proto);

  LOG(INFO) << proto.DebugString();

  BlockSparseMatrix A2(proto);

  Matrix m_a;
  Matrix m_b;
  A_->ToDenseMatrix(&m_a);
  A2.ToDenseMatrix(&m_b);

  LOG(INFO) << "\n" << m_a;
  LOG(INFO) << "\n" << m_b;

  EXPECT_LT((m_a - m_b).norm(), 1e-12);
}
#endif

}  // namespace internal
}  // namespace ceres
