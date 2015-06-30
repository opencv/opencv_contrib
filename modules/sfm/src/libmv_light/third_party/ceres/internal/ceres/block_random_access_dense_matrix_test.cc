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

#include <vector>
#include "gtest/gtest.h"
#include "ceres/block_random_access_dense_matrix.h"
#include "ceres/internal/eigen.h"

namespace ceres {
namespace internal {

TEST(BlockRandomAccessDenseMatrix, GetCell) {
  vector<int> blocks;
  blocks.push_back(3);
  blocks.push_back(4);
  blocks.push_back(5);
  const int num_rows = 3 + 4 + 5;
  BlockRandomAccessDenseMatrix m(blocks);
  EXPECT_EQ(m.num_rows(), num_rows);
  EXPECT_EQ(m.num_cols(), num_rows);

  int row_idx = 0;
  for (int i = 0; i < blocks.size(); ++i) {
    int col_idx = 0;
    for (int j = 0; j < blocks.size(); ++j) {
      int row;
      int col;
      int row_stride;
      int col_stride;
      CellInfo* cell =
          m.GetCell(i, j, &row, &col, &row_stride, &col_stride);

      EXPECT_TRUE(cell != NULL);
      EXPECT_EQ(row, row_idx);
      EXPECT_EQ(col, col_idx);
      EXPECT_EQ(row_stride, 3 + 4 + 5);
      EXPECT_EQ(col_stride, 3 + 4 + 5);
      col_idx += blocks[j];
    }
    row_idx += blocks[i];
  }
}

TEST(BlockRandomAccessDenseMatrix, WriteCell) {
  vector<int> blocks;
  blocks.push_back(3);
  blocks.push_back(4);
  blocks.push_back(5);
  const int num_rows = 3 + 4 + 5;

  BlockRandomAccessDenseMatrix m(blocks);

  // Fill the cell (i,j) with (i + 1) * (j + 1)
  for (int i = 0; i < blocks.size(); ++i) {
    for (int j = 0; j < blocks.size(); ++j) {
      int row;
      int col;
      int row_stride;
      int col_stride;
      CellInfo* cell = m.GetCell(
          i, j, &row, &col, &row_stride, &col_stride);
      MatrixRef(cell->values, row_stride, col_stride).block(
          row, col, blocks[i], blocks[j]) =
          (i+1) * (j+1) * Matrix::Ones(blocks[i], blocks[j]);
    }
  }

  // Check the values in the array are correct by going over the
  // entries of each block manually.
  int row_idx = 0;
  for (int i = 0; i < blocks.size(); ++i) {
    int col_idx = 0;
    for (int j = 0; j < blocks.size(); ++j) {
      // Check the values of this block.
      for (int r = 0; r < blocks[i]; ++r) {
        for (int c = 0; c < blocks[j]; ++c) {
          int pos = row_idx * num_rows + col_idx;
          EXPECT_EQ(m.values()[pos], (i + 1) * (j + 1));
        }
      }
      col_idx += blocks[j];
    }
    row_idx += blocks[i];
  }
}

}  // namespace internal
}  // namespace ceres
