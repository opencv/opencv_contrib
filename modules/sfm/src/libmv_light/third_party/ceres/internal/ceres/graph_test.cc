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

#include "ceres/graph.h"

#include "gtest/gtest.h"
#include "ceres/collections_port.h"
#include "ceres/internal/scoped_ptr.h"

namespace ceres {
namespace internal {

TEST(Graph, EmptyGraph) {
  Graph<int> graph;
  EXPECT_EQ(graph.vertices().size(), 0);
}

TEST(Graph, AddVertexAndEdge) {
  Graph<int> graph;
  graph.AddVertex(0, 1.0);
  graph.AddVertex(1, 2.0);
  graph.AddEdge(0, 1, 0.5);

  const HashSet<int>& vertices = graph.vertices();
  EXPECT_EQ(vertices.size(), 2);
  EXPECT_EQ(graph.VertexWeight(0), 1.0);
  EXPECT_EQ(graph.VertexWeight(1), 2.0);
  EXPECT_EQ(graph.Neighbors(0).size(), 1);
  EXPECT_EQ(graph.Neighbors(1).size(), 1);
  EXPECT_EQ(graph.EdgeWeight(0, 1), 0.5);
  EXPECT_EQ(graph.EdgeWeight(1, 0), 0.5);
}

TEST(Graph, AddVertexIdempotence) {
  Graph<int> graph;
  graph.AddVertex(0, 1.0);
  graph.AddVertex(1, 2.0);
  graph.AddEdge(0, 1, 0.5);

  const HashSet<int>& vertices = graph.vertices();

  EXPECT_EQ(vertices.size(), 2);

  // Try adding the vertex again with a new weight.
  graph.AddVertex(0, 3.0);
  EXPECT_EQ(vertices.size(), 2);

  // The vertex weight is reset.
  EXPECT_EQ(graph.VertexWeight(0), 3.0);

  // Rest of the graph remains the same.
  EXPECT_EQ(graph.VertexWeight(1), 2.0);
  EXPECT_EQ(graph.Neighbors(0).size(), 1);
  EXPECT_EQ(graph.Neighbors(1).size(), 1);
  EXPECT_EQ(graph.EdgeWeight(0, 1), 0.5);
  EXPECT_EQ(graph.EdgeWeight(1, 0), 0.5);
}

TEST(Graph, DieOnNonExistentVertex) {
  Graph<int> graph;
  graph.AddVertex(0, 1.0);
  graph.AddVertex(1, 2.0);
  graph.AddEdge(0, 1, 0.5);

  EXPECT_DEATH_IF_SUPPORTED(graph.VertexWeight(2), "key not found");
  EXPECT_DEATH_IF_SUPPORTED(graph.Neighbors(2), "key not found");
}

TEST(Graph, NonExistentEdge) {
  Graph<int> graph;
  graph.AddVertex(0, 1.0);
  graph.AddVertex(1, 2.0);
  graph.AddEdge(0, 1, 0.5);

  // Default value for non-existent edges is 0.
  EXPECT_EQ(graph.EdgeWeight(2, 3), 0);
}

}  // namespace internal
}  // namespace ceres
