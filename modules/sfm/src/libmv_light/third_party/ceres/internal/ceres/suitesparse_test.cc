// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2012 Google Inc. All rights reserved.
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

#include <algorithm>
#include "ceres/internal/port.h"
#include "ceres/suitesparse.h"
#include "ceres/triplet_sparse_matrix.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

TEST(SuiteSparse, BlockPermutationToScalarPermutation) {
  vector<int> blocks;
  //  Block structure
  //  0  --1-  ---2---  ---3---  4
  // [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  blocks.push_back(1);
  blocks.push_back(2);
  blocks.push_back(3);
  blocks.push_back(3);
  blocks.push_back(1);

  // Block ordering
  // [1, 0, 2, 4, 5]
  vector<int> block_ordering;
  block_ordering.push_back(1);
  block_ordering.push_back(0);
  block_ordering.push_back(2);
  block_ordering.push_back(4);
  block_ordering.push_back(3);

  // Expected ordering
  // [1, 2, 0, 3, 4, 5, 9, 6, 7, 8]
  vector<int> expected_scalar_ordering;
  expected_scalar_ordering.push_back(1);
  expected_scalar_ordering.push_back(2);
  expected_scalar_ordering.push_back(0);
  expected_scalar_ordering.push_back(3);
  expected_scalar_ordering.push_back(4);
  expected_scalar_ordering.push_back(5);
  expected_scalar_ordering.push_back(9);
  expected_scalar_ordering.push_back(6);
  expected_scalar_ordering.push_back(7);
  expected_scalar_ordering.push_back(8);

  vector<int> scalar_ordering;
  SuiteSparse::BlockOrderingToScalarOrdering(blocks,
                                             block_ordering,
                                             &scalar_ordering);
  EXPECT_EQ(scalar_ordering.size(), expected_scalar_ordering.size());
  for (int i = 0; i < expected_scalar_ordering.size(); ++i) {
    EXPECT_EQ(scalar_ordering[i], expected_scalar_ordering[i]);
  }
}

// Helper function to fill the sparsity pattern of a TripletSparseMatrix.
int FillBlock(const vector<int>& row_blocks,
              const vector<int>& col_blocks,
              const int row_block_id,
              const int col_block_id,
              int* rows,
              int* cols) {
  int row_pos = 0;
  for (int i = 0; i < row_block_id; ++i) {
    row_pos += row_blocks[i];
  }

  int col_pos = 0;
  for (int i = 0; i < col_block_id; ++i) {
    col_pos += col_blocks[i];
  }

  int offset = 0;
  for (int r = 0; r < row_blocks[row_block_id]; ++r) {
    for (int c = 0; c < col_blocks[col_block_id]; ++c, ++offset) {
      rows[offset] = row_pos + r;
      cols[offset] = col_pos + c;
    }
  }
  return offset;
}

TEST(SuiteSparse, ScalarMatrixToBlockMatrix) {
  // Block sparsity.
  //
  //     [1 2 3 2]
  // [1]  x   x
  // [2]    x   x
  // [2]  x x
  // num_nonzeros = 1 + 3 + 4 + 4 + 1 + 2 = 15

  vector<int> col_blocks;
  col_blocks.push_back(1);
  col_blocks.push_back(2);
  col_blocks.push_back(3);
  col_blocks.push_back(2);

  vector<int> row_blocks;
  row_blocks.push_back(1);
  row_blocks.push_back(2);
  row_blocks.push_back(2);

  TripletSparseMatrix tsm(5, 8, 18);
  int* rows = tsm.mutable_rows();
  int* cols = tsm.mutable_cols();
  fill(tsm.mutable_values(), tsm.mutable_values() + 18, 1.0);
  int offset = 0;

#define CERES_TEST_FILL_BLOCK(row_block_id, col_block_id) \
  offset += FillBlock(row_blocks, col_blocks, \
                      row_block_id, col_block_id, \
                      rows + offset, cols + offset);

  CERES_TEST_FILL_BLOCK(0, 0);
  CERES_TEST_FILL_BLOCK(2, 0);
  CERES_TEST_FILL_BLOCK(1, 1);
  CERES_TEST_FILL_BLOCK(2, 1);
  CERES_TEST_FILL_BLOCK(0, 2);
  CERES_TEST_FILL_BLOCK(1, 3);
#undef CERES_TEST_FILL_BLOCK

  tsm.set_num_nonzeros(offset);

  SuiteSparse ss;
  scoped_ptr<cholmod_sparse> ccsm(ss.CreateSparseMatrix(&tsm));

  vector<int> expected_block_rows;
  expected_block_rows.push_back(0);
  expected_block_rows.push_back(2);
  expected_block_rows.push_back(1);
  expected_block_rows.push_back(2);
  expected_block_rows.push_back(0);
  expected_block_rows.push_back(1);

  vector<int> expected_block_cols;
  expected_block_cols.push_back(0);
  expected_block_cols.push_back(2);
  expected_block_cols.push_back(4);
  expected_block_cols.push_back(5);
  expected_block_cols.push_back(6);

  vector<int> block_rows;
  vector<int> block_cols;
  SuiteSparse::ScalarMatrixToBlockMatrix(ccsm.get(),
                                         row_blocks,
                                         col_blocks,
                                         &block_rows,
                                         &block_cols);

  EXPECT_EQ(block_cols.size(), expected_block_cols.size());
  EXPECT_EQ(block_rows.size(), expected_block_rows.size());

  for (int i = 0; i < expected_block_cols.size(); ++i) {
    EXPECT_EQ(block_cols[i], expected_block_cols[i]);
  }

  for (int i = 0; i < expected_block_rows.size(); ++i) {
    EXPECT_EQ(block_rows[i], expected_block_rows[i]);
  }

  ss.Free(ccsm.release());
}

}  // namespace internal
}  // namespace ceres
