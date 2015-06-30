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

#include "ceres/compressed_row_sparse_matrix.h"

#include "ceres/casts.h"
#include "ceres/crs_matrix.h"
#include "ceres/internal/eigen.h"
#include "ceres/internal/scoped_ptr.h"
#include "ceres/linear_least_squares_problems.h"
#include "ceres/matrix_proto.h"
#include "ceres/triplet_sparse_matrix.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

void CompareMatrices(const SparseMatrix* a, const SparseMatrix* b) {
  EXPECT_EQ(a->num_rows(), b->num_rows());
  EXPECT_EQ(a->num_cols(), b->num_cols());

  int num_rows = a->num_rows();
  int num_cols = a->num_cols();

  for (int i = 0; i < num_cols; ++i) {
    Vector x = Vector::Zero(num_cols);
    x(i) = 1.0;

    Vector y_a = Vector::Zero(num_rows);
    Vector y_b = Vector::Zero(num_rows);

    a->RightMultiply(x.data(), y_a.data());
    b->RightMultiply(x.data(), y_b.data());

    EXPECT_EQ((y_a - y_b).norm(), 0);
  }
}

class CompressedRowSparseMatrixTest : public ::testing::Test {
 protected :
  virtual void SetUp() {
    scoped_ptr<LinearLeastSquaresProblem> problem(
        CreateLinearLeastSquaresProblemFromId(1));

    CHECK_NOTNULL(problem.get());

    tsm.reset(down_cast<TripletSparseMatrix*>(problem->A.release()));
    crsm.reset(new CompressedRowSparseMatrix(*tsm));

    num_rows = tsm->num_rows();
    num_cols = tsm->num_cols();
  }

  int num_rows;
  int num_cols;

  scoped_ptr<TripletSparseMatrix> tsm;
  scoped_ptr<CompressedRowSparseMatrix> crsm;
};

TEST_F(CompressedRowSparseMatrixTest, RightMultiply) {
  CompareMatrices(tsm.get(), crsm.get());
}

TEST_F(CompressedRowSparseMatrixTest, LeftMultiply) {
  for (int i = 0; i < num_rows; ++i) {
    Vector a = Vector::Zero(num_rows);
    a(i) = 1.0;

    Vector b1 = Vector::Zero(num_cols);
    Vector b2 = Vector::Zero(num_cols);

    tsm->LeftMultiply(a.data(), b1.data());
    crsm->LeftMultiply(a.data(), b2.data());

    EXPECT_EQ((b1 - b2).norm(), 0);
  }
}

TEST_F(CompressedRowSparseMatrixTest, ColumnNorm) {
  Vector b1 = Vector::Zero(num_cols);
  Vector b2 = Vector::Zero(num_cols);

  tsm->SquaredColumnNorm(b1.data());
  crsm->SquaredColumnNorm(b2.data());

  EXPECT_EQ((b1 - b2).norm(), 0);
}

TEST_F(CompressedRowSparseMatrixTest, Scale) {
  Vector scale(num_cols);
  for (int i = 0; i < num_cols; ++i) {
    scale(i) = i + 1;
  }

  tsm->ScaleColumns(scale.data());
  crsm->ScaleColumns(scale.data());
  CompareMatrices(tsm.get(), crsm.get());
}

TEST_F(CompressedRowSparseMatrixTest, DeleteRows) {
  for (int i = 0; i < num_rows; ++i) {
    tsm->Resize(num_rows - i, num_cols);
    crsm->DeleteRows(crsm->num_rows() - tsm->num_rows());
    CompareMatrices(tsm.get(), crsm.get());
  }
}

TEST_F(CompressedRowSparseMatrixTest, AppendRows) {
  for (int i = 0; i < num_rows; ++i) {
    TripletSparseMatrix tsm_appendage(*tsm);
    tsm_appendage.Resize(i, num_cols);

    tsm->AppendRows(tsm_appendage);
    CompressedRowSparseMatrix crsm_appendage(tsm_appendage);
    crsm->AppendRows(crsm_appendage);

    CompareMatrices(tsm.get(), crsm.get());
  }
}

#ifndef CERES_NO_PROTOCOL_BUFFERS
TEST_F(CompressedRowSparseMatrixTest, Serialization) {
  SparseMatrixProto proto;
  crsm->ToProto(&proto);

  CompressedRowSparseMatrix n(proto);
  ASSERT_EQ(n.num_rows(), crsm->num_rows());
  ASSERT_EQ(n.num_cols(), crsm->num_cols());
  ASSERT_EQ(n.num_nonzeros(), crsm->num_nonzeros());

  for (int i = 0; i < n.num_rows() + 1; ++i) {
    ASSERT_EQ(crsm->rows()[i], proto.compressed_row_matrix().rows(i));
    ASSERT_EQ(crsm->rows()[i], n.rows()[i]);
  }

  for (int i = 0; i < crsm->num_nonzeros(); ++i) {
    ASSERT_EQ(crsm->cols()[i], proto.compressed_row_matrix().cols(i));
    ASSERT_EQ(crsm->cols()[i], n.cols()[i]);
    ASSERT_EQ(crsm->values()[i], proto.compressed_row_matrix().values(i));
    ASSERT_EQ(crsm->values()[i], n.values()[i]);
  }
}
#endif

TEST_F(CompressedRowSparseMatrixTest, ToDenseMatrix) {
  Matrix tsm_dense;
  Matrix crsm_dense;

  tsm->ToDenseMatrix(&tsm_dense);
  crsm->ToDenseMatrix(&crsm_dense);

  EXPECT_EQ((tsm_dense - crsm_dense).norm(), 0.0);
}

TEST_F(CompressedRowSparseMatrixTest, ToCRSMatrix) {
  CRSMatrix crs_matrix;
  crsm->ToCRSMatrix(&crs_matrix);
  EXPECT_EQ(crsm->num_rows(), crs_matrix.num_rows);
  EXPECT_EQ(crsm->num_cols(), crs_matrix.num_cols);
  EXPECT_EQ(crsm->num_rows() + 1, crs_matrix.rows.size());
  EXPECT_EQ(crsm->num_nonzeros(), crs_matrix.cols.size());
  EXPECT_EQ(crsm->num_nonzeros(), crs_matrix.values.size());

  for (int i = 0; i < crsm->num_rows() + 1; ++i) {
    EXPECT_EQ(crsm->rows()[i], crs_matrix.rows[i]);
  }

  for (int i = 0; i < crsm->num_nonzeros(); ++i) {
    EXPECT_EQ(crsm->cols()[i], crs_matrix.cols[i]);
    EXPECT_EQ(crsm->values()[i], crs_matrix.values[i]);
  }
}

}  // namespace internal
}  // namespace ceres
