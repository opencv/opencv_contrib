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
// Author: Sameer Agarwal (sameeragarwal@google.com)
//         David Gallup (dgallup@google.com)

#ifndef CERES_NO_SUITESPARSE

#include "ceres/canonical_views_clustering.h"

#include "ceres/collections_port.h"
#include "ceres/graph.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

const int kVertexIds[] = {0, 1, 2, 3};
class CanonicalViewsTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    // The graph structure is as follows.
    //
    // Vertex weights:   0      2      2      0
    //                   V0-----V1-----V2-----V3
    // Edge weights:        0.8    0.9    0.3
    const double kVertexWeights[] = {0.0, 2.0, 2.0, -1.0};
    for (int i = 0; i < 4; ++i) {
      graph_.AddVertex(i, kVertexWeights[i]);
    }
    // Create self edges.
    // CanonicalViews requires that every view "sees" itself.
    for (int i = 0; i < 4; ++i) {
      graph_.AddEdge(i, i, 1.0);
    }

    // Create three edges.
    const double kEdgeWeights[] = {0.8, 0.9, 0.3};
    for (int i = 0; i < 3; ++i) {
      // The graph interface is directed, so remember to create both
      // edges.
      graph_.AddEdge(kVertexIds[i], kVertexIds[i + 1], kEdgeWeights[i]);
    }
  }

  void ComputeClustering() {
    ComputeCanonicalViewsClustering(graph_, options_, &centers_, &membership_);
  }

  Graph<int> graph_;

  CanonicalViewsClusteringOptions options_;
  vector<int> centers_;
  HashMap<int, int> membership_;
};

TEST_F(CanonicalViewsTest, ComputeCanonicalViewsTest) {
  options_.min_views = 0;
  options_.size_penalty_weight = 0.5;
  options_.similarity_penalty_weight = 0.0;
  options_.view_score_weight = 0.0;
  ComputeClustering();

  // 2 canonical views.
  EXPECT_EQ(centers_.size(), 2);
  EXPECT_EQ(centers_[0], kVertexIds[1]);
  EXPECT_EQ(centers_[1], kVertexIds[3]);

  // Check cluster membership.
  EXPECT_EQ(FindOrDie(membership_, kVertexIds[0]), 0);
  EXPECT_EQ(FindOrDie(membership_, kVertexIds[1]), 0);
  EXPECT_EQ(FindOrDie(membership_, kVertexIds[2]), 0);
  EXPECT_EQ(FindOrDie(membership_, kVertexIds[3]), 1);
}

// Increases size penalty so the second canonical view won't be
// chosen.
TEST_F(CanonicalViewsTest, SizePenaltyTest) {
  options_.min_views = 0;
  options_.size_penalty_weight = 2.0;
  options_.similarity_penalty_weight = 0.0;
  options_.view_score_weight = 0.0;
  ComputeClustering();

  // 1 canonical view.
  EXPECT_EQ(centers_.size(), 1);
  EXPECT_EQ(centers_[0], kVertexIds[1]);
}


// Increases view score weight so vertex 2 will be chosen.
TEST_F(CanonicalViewsTest, ViewScoreTest) {
  options_.min_views = 0;
  options_.size_penalty_weight = 0.5;
  options_.similarity_penalty_weight = 0.0;
  options_.view_score_weight = 1.0;
  ComputeClustering();

  // 2 canonical views.
  EXPECT_EQ(centers_.size(), 2);
  EXPECT_EQ(centers_[0], kVertexIds[1]);
  EXPECT_EQ(centers_[1], kVertexIds[2]);
}

// Increases similarity penalty so vertex 2 won't be chosen despite
// it's view score.
TEST_F(CanonicalViewsTest, SimilarityPenaltyTest) {
  options_.min_views = 0;
  options_.size_penalty_weight = 0.5;
  options_.similarity_penalty_weight = 3.0;
  options_.view_score_weight = 1.0;
  ComputeClustering();

  // 2 canonical views.
  EXPECT_EQ(centers_.size(), 1);
  EXPECT_EQ(centers_[0], kVertexIds[1]);
}

}  // namespace internal
}  // namespace ceres

#endif  // CERES_NO_SUITESPARSE
