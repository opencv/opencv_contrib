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

#include "ceres/graph_algorithms.h"

#include <algorithm>
#include "gtest/gtest.h"
#include "ceres/collections_port.h"
#include "ceres/graph.h"
#include "ceres/internal/port.h"
#include "ceres/internal/scoped_ptr.h"

namespace ceres {
namespace internal {

TEST(IndependentSetOrdering, Chain) {
  Graph<int> graph;
  graph.AddVertex(0);
  graph.AddVertex(1);
  graph.AddVertex(2);
  graph.AddVertex(3);
  graph.AddVertex(4);

  graph.AddEdge(0, 1);
  graph.AddEdge(1, 2);
  graph.AddEdge(2, 3);
  graph.AddEdge(3, 4);

  // 0-1-2-3-4
  // 0, 2, 4 should be in the independent set.
  vector<int> ordering;
  int independent_set_size = IndependentSetOrdering(graph, &ordering);

  sort(ordering.begin(), ordering.begin() + 3);
  sort(ordering.begin() + 3, ordering.end());

  EXPECT_EQ(independent_set_size, 3);
  EXPECT_EQ(ordering.size(), 5);
  EXPECT_EQ(ordering[0], 0);
  EXPECT_EQ(ordering[1], 2);
  EXPECT_EQ(ordering[2], 4);
  EXPECT_EQ(ordering[3], 1);
  EXPECT_EQ(ordering[4], 3);
}

TEST(IndependentSetOrdering, Star) {
  Graph<int> graph;
  graph.AddVertex(0);
  graph.AddVertex(1);
  graph.AddVertex(2);
  graph.AddVertex(3);
  graph.AddVertex(4);

  graph.AddEdge(0, 1);
  graph.AddEdge(0, 2);
  graph.AddEdge(0, 3);
  graph.AddEdge(0, 4);

  //      1
  //      |
  //    4-0-2
  //      |
  //      3
  // 1, 2, 3, 4 should be in the indepdendent set.
  vector<int> ordering;
  int independent_set_size = IndependentSetOrdering(graph, &ordering);
  EXPECT_EQ(independent_set_size, 4);
  EXPECT_EQ(ordering.size(), 5);
  EXPECT_EQ(ordering[4], 0);
  sort(ordering.begin(), ordering.begin() + 4);
  EXPECT_EQ(ordering[0], 1);
  EXPECT_EQ(ordering[1], 2);
  EXPECT_EQ(ordering[2], 3);
  EXPECT_EQ(ordering[3], 4);
}

TEST(Degree2MaximumSpanningForest, PreserveWeights) {
  Graph<int> graph;
  graph.AddVertex(0, 1.0);
  graph.AddVertex(1, 2.0);
  graph.AddEdge(0, 1, 0.5);
  graph.AddEdge(1, 0, 0.5);

  scoped_ptr<Graph<int> > forest(Degree2MaximumSpanningForest(graph));

  const HashSet<int>& vertices = forest->vertices();
  EXPECT_EQ(vertices.size(), 2);
  EXPECT_EQ(forest->VertexWeight(0), 1.0);
  EXPECT_EQ(forest->VertexWeight(1), 2.0);
  EXPECT_EQ(forest->Neighbors(0).size(), 1.0);
  EXPECT_EQ(forest->EdgeWeight(0, 1), 0.5);
}

TEST(Degree2MaximumSpanningForest, StarGraph) {
  Graph<int> graph;
  graph.AddVertex(0);
  graph.AddVertex(1);
  graph.AddVertex(2);
  graph.AddVertex(3);
  graph.AddVertex(4);

  graph.AddEdge(0, 1, 1.0);
  graph.AddEdge(0, 2, 2.0);
  graph.AddEdge(0, 3, 3.0);
  graph.AddEdge(0, 4, 4.0);

  scoped_ptr<Graph<int> > forest(Degree2MaximumSpanningForest(graph));
  const HashSet<int>& vertices = forest->vertices();
  EXPECT_EQ(vertices.size(), 5);

  {
    const HashSet<int>& neighbors = forest->Neighbors(0);
    EXPECT_EQ(neighbors.size(), 2);
    EXPECT_TRUE(neighbors.find(4) != neighbors.end());
    EXPECT_TRUE(neighbors.find(3) != neighbors.end());
  }

  {
    const HashSet<int>& neighbors = forest->Neighbors(3);
    EXPECT_EQ(neighbors.size(), 1);
    EXPECT_TRUE(neighbors.find(0) != neighbors.end());
  }

  {
    const HashSet<int>& neighbors = forest->Neighbors(4);
    EXPECT_EQ(neighbors.size(), 1);
    EXPECT_TRUE(neighbors.find(0) != neighbors.end());
  }

  {
    const HashSet<int>& neighbors = forest->Neighbors(1);
    EXPECT_EQ(neighbors.size(), 0);
  }

  {
    const HashSet<int>& neighbors = forest->Neighbors(2);
    EXPECT_EQ(neighbors.size(), 0);
  }
}

TEST(VertexDegreeLessThan, TotalOrdering) {
  Graph<int> graph;
  graph.AddVertex(0);
  graph.AddVertex(1);
  graph.AddVertex(2);
  graph.AddVertex(3);

  // 0-1
  //   |
  // 2-3
  // 0,1 and 2 have degree 1 and 3 has degree 2.
  graph.AddEdge(0, 1, 1.0);
  graph.AddEdge(2, 3, 1.0);
  VertexDegreeLessThan<int> less_than(graph);

  for (int i = 0; i < 4; ++i) {
    EXPECT_FALSE(less_than(i, i)) << "Failing vertex: " << i;
    for (int j = 0; j < 4; ++j) {
      if (i != j) {
        EXPECT_TRUE(less_than(i, j) ^ less_than(j, i))
            << "Failing vertex pair: " << i << " " << j;
      }
    }
  }

  for (int i = 0; i < 3; ++i) {
    EXPECT_TRUE(less_than(i, 3));
    EXPECT_FALSE(less_than(3, i));
  }
}

}  // namespace internal
}  // namespace ceres
