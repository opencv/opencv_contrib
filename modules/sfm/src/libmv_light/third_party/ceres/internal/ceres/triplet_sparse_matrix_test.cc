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

#include "ceres/triplet_sparse_matrix.h"

#include "gtest/gtest.h"
#include "ceres/matrix_proto.h"
#include "ceres/internal/scoped_ptr.h"

namespace ceres {
namespace internal {

TEST(TripletSparseMatrix, DefaultConstructorReturnsEmptyObject) {
  TripletSparseMatrix m;
  EXPECT_EQ(m.num_rows(), 0);
  EXPECT_EQ(m.num_cols(), 0);
  EXPECT_EQ(m.num_nonzeros(), 0);
  EXPECT_EQ(m.max_num_nonzeros(), 0);
}

TEST(TripletSparseMatrix, SimpleConstructorAndBasicOperations) {
  // Build a matrix
  TripletSparseMatrix m(2, 5, 4);
  EXPECT_EQ(m.num_rows(), 2);
  EXPECT_EQ(m.num_cols(), 5);
  EXPECT_EQ(m.num_nonzeros(), 0);
  EXPECT_EQ(m.max_num_nonzeros(), 4);

  m.mutable_rows()[0] = 0;
  m.mutable_cols()[0] = 1;
  m.mutable_values()[0] = 2.5;

  m.mutable_rows()[1] = 1;
  m.mutable_cols()[1] = 4;
  m.mutable_values()[1] = 5.2;
  m.set_num_nonzeros(2);

  EXPECT_EQ(m.num_nonzeros(), 2);

  ASSERT_TRUE(m.AllTripletsWithinBounds());

  // We should never be able resize and lose data
  EXPECT_DEATH_IF_SUPPORTED(m.Reserve(1), "Reallocation will cause data loss");

  // We should be able to resize while preserving data
  m.Reserve(50);
  EXPECT_EQ(m.max_num_nonzeros(), 50);

  m.Reserve(3);
  EXPECT_EQ(m.max_num_nonzeros(), 50);  // The space is already reserved.

  EXPECT_EQ(m.rows()[0], 0);
  EXPECT_EQ(m.rows()[1], 1);

  EXPECT_EQ(m.cols()[0], 1);
  EXPECT_EQ(m.cols()[1], 4);

  EXPECT_DOUBLE_EQ(m.values()[0], 2.5);
  EXPECT_DOUBLE_EQ(m.values()[1], 5.2);

  // Bounds check should fail
  m.mutable_rows()[0] = 10;
  EXPECT_FALSE(m.AllTripletsWithinBounds());

  m.mutable_rows()[0] = 1;
  m.mutable_cols()[0] = 100;
  EXPECT_FALSE(m.AllTripletsWithinBounds());

  // Remove all data and then resize the data store
  m.SetZero();
  EXPECT_EQ(m.num_nonzeros(), 0);
  m.Reserve(1);
}

TEST(TripletSparseMatrix, CopyConstructor) {
  TripletSparseMatrix orig(2, 5, 4);
  orig.mutable_rows()[0] = 0;
  orig.mutable_cols()[0] = 1;
  orig.mutable_values()[0] = 2.5;

  orig.mutable_rows()[1] = 1;
  orig.mutable_cols()[1] = 4;
  orig.mutable_values()[1] = 5.2;
  orig.set_num_nonzeros(2);

  TripletSparseMatrix cpy(orig);

  EXPECT_EQ(cpy.num_rows(), 2);
  EXPECT_EQ(cpy.num_cols(), 5);
  ASSERT_EQ(cpy.num_nonzeros(), 2);
  EXPECT_EQ(cpy.max_num_nonzeros(), 4);

  EXPECT_EQ(cpy.rows()[0], 0);
  EXPECT_EQ(cpy.rows()[1], 1);

  EXPECT_EQ(cpy.cols()[0], 1);
  EXPECT_EQ(cpy.cols()[1], 4);

  EXPECT_DOUBLE_EQ(cpy.values()[0], 2.5);
  EXPECT_DOUBLE_EQ(cpy.values()[1], 5.2);
}

TEST(TripletSparseMatrix, AssignmentOperator) {
  TripletSparseMatrix orig(2, 5, 4);
  orig.mutable_rows()[0] = 0;
  orig.mutable_cols()[0] = 1;
  orig.mutable_values()[0] = 2.5;

  orig.mutable_rows()[1] = 1;
  orig.mutable_cols()[1] = 4;
  orig.mutable_values()[1] = 5.2;
  orig.set_num_nonzeros(2);

  TripletSparseMatrix cpy(3, 50, 40);
  cpy.mutable_rows()[0] = 0;
  cpy.mutable_cols()[0] = 10;
  cpy.mutable_values()[0] = 10.22;

  cpy.mutable_rows()[1] = 2;
  cpy.mutable_cols()[1] = 23;
  cpy.mutable_values()[1] = 34.45;

  cpy.mutable_rows()[0] = 0;
  cpy.mutable_cols()[0] = 10;
  cpy.mutable_values()[0] = 10.22;

  cpy.mutable_rows()[1] = 0;
  cpy.mutable_cols()[1] = 3;
  cpy.mutable_values()[1] = 4.4;
  cpy.set_num_nonzeros(3);

  cpy = orig;

  EXPECT_EQ(cpy.num_rows(), 2);
  EXPECT_EQ(cpy.num_cols(), 5);
  ASSERT_EQ(cpy.num_nonzeros(), 2);
  EXPECT_EQ(cpy.max_num_nonzeros(), 4);

  EXPECT_EQ(cpy.rows()[0], 0);
  EXPECT_EQ(cpy.rows()[1], 1);

  EXPECT_EQ(cpy.cols()[0], 1);
  EXPECT_EQ(cpy.cols()[1], 4);

  EXPECT_DOUBLE_EQ(cpy.values()[0], 2.5);
  EXPECT_DOUBLE_EQ(cpy.values()[1], 5.2);
}

TEST(TripletSparseMatrix, AppendRows) {
  // Build one matrix.
  TripletSparseMatrix m(2, 5, 4);
  m.mutable_rows()[0] = 0;
  m.mutable_cols()[0] = 1;
  m.mutable_values()[0] = 2.5;

  m.mutable_rows()[1] = 1;
  m.mutable_cols()[1] = 4;
  m.mutable_values()[1] = 5.2;
  m.set_num_nonzeros(2);

  // Build another matrix.
  TripletSparseMatrix a(10, 5, 4);
  a.mutable_rows()[0] = 0;
  a.mutable_cols()[0] = 1;
  a.mutable_values()[0] = 3.5;

  a.mutable_rows()[1] = 1;
  a.mutable_cols()[1] = 4;
  a.mutable_values()[1] = 6.2;

  a.mutable_rows()[2] = 9;
  a.mutable_cols()[2] = 5;
  a.mutable_values()[2] = 1;
  a.set_num_nonzeros(3);

  // Glue the second matrix to the bottom of the first.
  m.AppendRows(a);

  EXPECT_EQ(m.num_rows(), 12);
  EXPECT_EQ(m.num_cols(), 5);
  ASSERT_EQ(m.num_nonzeros(), 5);

  EXPECT_EQ(m.values()[0], 2.5);
  EXPECT_EQ(m.values()[1], 5.2);
  EXPECT_EQ(m.values()[2], 3.5);
  EXPECT_EQ(m.values()[3], 6.2);
  EXPECT_EQ(m.values()[4], 1);

  EXPECT_EQ(m.rows()[0], 0);
  EXPECT_EQ(m.rows()[1], 1);
  EXPECT_EQ(m.rows()[2], 2);
  EXPECT_EQ(m.rows()[3], 3);
  EXPECT_EQ(m.rows()[4], 11);

  EXPECT_EQ(m.cols()[0], 1);
  EXPECT_EQ(m.cols()[1], 4);
  EXPECT_EQ(m.cols()[2], 1);
  EXPECT_EQ(m.cols()[3], 4);
  EXPECT_EQ(m.cols()[4], 5);
}

TEST(TripletSparseMatrix, AppendCols) {
  // Build one matrix.
  TripletSparseMatrix m(2, 5, 4);
  m.mutable_rows()[0] = 0;
  m.mutable_cols()[0] = 1;
  m.mutable_values()[0] = 2.5;

  m.mutable_rows()[1] = 1;
  m.mutable_cols()[1] = 4;
  m.mutable_values()[1] = 5.2;
  m.set_num_nonzeros(2);

  // Build another matrix.
  TripletSparseMatrix a(2, 15, 4);
  a.mutable_rows()[0] = 0;
  a.mutable_cols()[0] = 1;
  a.mutable_values()[0] = 3.5;

  a.mutable_rows()[1] = 1;
  a.mutable_cols()[1] = 4;
  a.mutable_values()[1] = 6.2;

  a.mutable_rows()[2] = 0;
  a.mutable_cols()[2] = 10;
  a.mutable_values()[2] = 1;
  a.set_num_nonzeros(3);

  // Glue the second matrix to the left of the first.
  m.AppendCols(a);

  EXPECT_EQ(m.num_rows(), 2);
  EXPECT_EQ(m.num_cols(), 20);
  ASSERT_EQ(m.num_nonzeros(), 5);

  EXPECT_EQ(m.values()[0], 2.5);
  EXPECT_EQ(m.values()[1], 5.2);
  EXPECT_EQ(m.values()[2], 3.5);
  EXPECT_EQ(m.values()[3], 6.2);
  EXPECT_EQ(m.values()[4], 1);

  EXPECT_EQ(m.rows()[0], 0);
  EXPECT_EQ(m.rows()[1], 1);
  EXPECT_EQ(m.rows()[2], 0);
  EXPECT_EQ(m.rows()[3], 1);
  EXPECT_EQ(m.rows()[4], 0);

  EXPECT_EQ(m.cols()[0], 1);
  EXPECT_EQ(m.cols()[1], 4);
  EXPECT_EQ(m.cols()[2], 6);
  EXPECT_EQ(m.cols()[3], 9);
  EXPECT_EQ(m.cols()[4], 15);
}

TEST(TripletSparseMatrix, CreateDiagonalMatrix) {
  scoped_array<double> values(new double[10]);
  for (int i = 0; i < 10; ++i)
    values[i] = i;

  scoped_ptr<TripletSparseMatrix> m(
      TripletSparseMatrix::CreateSparseDiagonalMatrix(values.get(), 10));
  EXPECT_EQ(m->num_rows(), 10);
  EXPECT_EQ(m->num_cols(), 10);
  ASSERT_EQ(m->num_nonzeros(), 10);
  for (int i = 0; i < 10 ; ++i) {
    EXPECT_EQ(m->rows()[i], i);
    EXPECT_EQ(m->cols()[i], i);
    EXPECT_EQ(m->values()[i], i);
  }
}

TEST(TripletSparseMatrix, Resize) {
  TripletSparseMatrix m(10, 20, 200);
  int nnz = 0;
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 20; ++j) {
      m.mutable_rows()[nnz] = i;
      m.mutable_cols()[nnz] = j;
      m.mutable_values()[nnz++] = i+j;
    }
  }
  m.set_num_nonzeros(nnz);
  m.Resize(5, 6);
  EXPECT_EQ(m.num_rows(), 5);
  EXPECT_EQ(m.num_cols(), 6);
  ASSERT_EQ(m.num_nonzeros(), 30);
  for (int i = 0; i < 30; ++i) {
    EXPECT_EQ(m.values()[i], m.rows()[i] + m.cols()[i]);
  }
}

#ifndef CERES_NO_PROTOCOL_BUFFERS
TEST(TripletSparseMatrix, Serialization) {
  TripletSparseMatrix m(2, 5, 4);

  m.mutable_rows()[0] = 0;
  m.mutable_cols()[0] = 1;
  m.mutable_values()[0] = 2.5;

  m.mutable_rows()[1] = 1;
  m.mutable_cols()[1] = 4;
  m.mutable_values()[1] = 5.2;
  m.set_num_nonzeros(2);

  // Roundtrip through serialization and check for equality.
  SparseMatrixProto proto;
  m.ToProto(&proto);

  TripletSparseMatrix n(proto);

  ASSERT_EQ(n.num_rows(), 2);
  ASSERT_EQ(n.num_cols(), 5);

  // Note that max_num_nonzeros gets truncated; the serialization
  ASSERT_EQ(n.num_nonzeros(), 2);
  ASSERT_EQ(n.max_num_nonzeros(), 2);

  for (int i = 0; i < m.num_nonzeros(); ++i) {
    EXPECT_EQ(m.rows()[i],   n.rows()[i]);
    EXPECT_EQ(m.cols()[i],   n.cols()[i]);
    EXPECT_EQ(m.values()[i], n.values()[i]);
  }
}
#endif

}  // namespace internal
}  // namespace ceres
