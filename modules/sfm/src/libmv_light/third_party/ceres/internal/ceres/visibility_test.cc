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
// Author: kushalav@google.com (Avanish Kushal)
//         sameeragarwal@google.com (Sameer Agarwal)

#ifndef CERES_NO_SUITESPARSE

#include "ceres/visibility.h"

#include <set>
#include <vector>
#include "ceres/block_structure.h"
#include "ceres/graph.h"
#include "ceres/internal/scoped_ptr.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

class VisibilityTest : public ::testing::Test {
};

TEST(VisibilityTest, SimpleMatrix) {
  //   A = [1 0 0 0 0 1
  //        1 0 0 1 0 0
  //        0 1 1 0 0 0
  //        0 1 0 0 1 0]

  int num_cols = 6;
  int num_eliminate_blocks = 2;
  CompressedRowBlockStructure bs;

  // Row 1
  {
    bs.rows.push_back(CompressedRow());
    CompressedRow& row = bs.rows.back();
    row.block.size = 2;
    row.block.position = 0;
    row.cells.push_back(Cell(0, 0));
    row.cells.push_back(Cell(5, 0));
  }

  // Row 2
  {
    bs.rows.push_back(CompressedRow());
    CompressedRow& row = bs.rows.back();
    row.block.size = 2;
    row.block.position = 2;
    row.cells.push_back(Cell(0, 1));
    row.cells.push_back(Cell(3, 1));
  }

  // Row 3
  {
    bs.rows.push_back(CompressedRow());
    CompressedRow& row = bs.rows.back();
    row.block.size = 2;
    row.block.position = 4;
    row.cells.push_back(Cell(1, 2));
    row.cells.push_back(Cell(2, 2));
  }

  // Row 4
  {
    bs.rows.push_back(CompressedRow());
    CompressedRow& row = bs.rows.back();
    row.block.size = 2;
    row.block.position = 6;
    row.cells.push_back(Cell(1, 3));
    row.cells.push_back(Cell(4, 3));
  }
  bs.cols.resize(num_cols);

  vector< set<int> > visibility;
  ComputeVisibility(bs, num_eliminate_blocks, &visibility);
  ASSERT_EQ(visibility.size(), num_cols - num_eliminate_blocks);
  for (int i = 0; i < visibility.size(); ++i) {
    ASSERT_EQ(visibility[i].size(), 1);
  }

  scoped_ptr<Graph<int> > graph(CreateSchurComplementGraph(visibility));
  EXPECT_EQ(graph->vertices().size(), visibility.size());
  for (int i = 0; i < visibility.size(); ++i) {
    EXPECT_EQ(graph->VertexWeight(i), 1.0);
  }

  for (int i = 0; i < visibility.size(); ++i) {
    for (int j = i; j < visibility.size(); ++j) {
      double edge_weight = 0.0;
      if ((i == 1 && j == 3) || (i == 0 && j == 2) || (i == j)) {
        edge_weight = 1.0;
      }

      EXPECT_EQ(graph->EdgeWeight(i, j), edge_weight)
          << "Edge: " << i << " " << j
          << " weight: " << graph->EdgeWeight(i, j)
          << " expected weight: " << edge_weight;
    }
  }
}


TEST(VisibilityTest, NoEBlocks) {
  //   A = [1 0 0 0 0 0
  //        1 0 0 0 0 0
  //        0 1 0 0 0 0
  //        0 1 0 0 0 0]

  int num_cols = 6;
  int num_eliminate_blocks = 2;
  CompressedRowBlockStructure bs;

  // Row 1
  {
    bs.rows.push_back(CompressedRow());
    CompressedRow& row = bs.rows.back();
    row.block.size = 2;
    row.block.position = 0;
    row.cells.push_back(Cell(0, 0));
  }

  // Row 2
  {
    bs.rows.push_back(CompressedRow());
    CompressedRow& row = bs.rows.back();
    row.block.size = 2;
    row.block.position = 2;
    row.cells.push_back(Cell(0, 1));
  }

  // Row 3
  {
    bs.rows.push_back(CompressedRow());
    CompressedRow& row = bs.rows.back();
    row.block.size = 2;
    row.block.position = 4;
    row.cells.push_back(Cell(1, 2));
  }

  // Row 4
  {
    bs.rows.push_back(CompressedRow());
    CompressedRow& row = bs.rows.back();
    row.block.size = 2;
    row.block.position = 6;
    row.cells.push_back(Cell(1, 3));
  }
  bs.cols.resize(num_cols);

  vector<set<int> > visibility;
  ComputeVisibility(bs, num_eliminate_blocks, &visibility);
  ASSERT_EQ(visibility.size(), num_cols - num_eliminate_blocks);
  for (int i = 0; i < visibility.size(); ++i) {
    ASSERT_EQ(visibility[i].size(), 0);
  }

  scoped_ptr<Graph<int> > graph(CreateSchurComplementGraph(visibility));
  EXPECT_EQ(graph->vertices().size(), visibility.size());
  for (int i = 0; i < visibility.size(); ++i) {
    EXPECT_EQ(graph->VertexWeight(i), 1.0);
  }

  for (int i = 0; i < visibility.size(); ++i) {
    for (int j = i; j < visibility.size(); ++j) {
      double edge_weight = 0.0;
      if (i == j) {
        edge_weight = 1.0;
      }
      EXPECT_EQ(graph->EdgeWeight(i, j), edge_weight)
          << "Edge: " << i << " " << j
          << " weight: " << graph->EdgeWeight(i, j)
          << " expected weight: " << edge_weight;
    }
  }
}

}  // namespace internal
}  // namespace ceres

#endif  // CERES_NO_SUITESPARSE
