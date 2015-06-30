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

#ifndef CERES_NO_SUITESPARSE

#include "ceres/visibility_based_preconditioner.h"

#include "Eigen/Dense"
#include "ceres/block_random_access_dense_matrix.h"
#include "ceres/block_random_access_sparse_matrix.h"
#include "ceres/block_sparse_matrix.h"
#include "ceres/casts.h"
#include "ceres/collections_port.h"
#include "ceres/file.h"
#include "ceres/internal/eigen.h"
#include "ceres/internal/scoped_ptr.h"
#include "ceres/linear_least_squares_problems.h"
#include "ceres/schur_eliminator.h"
#include "ceres/stringprintf.h"
#include "ceres/types.h"
#include "ceres/test_util.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

using testing::AssertionResult;
using testing::AssertionSuccess;
using testing::AssertionFailure;

static const double kTolerance = 1e-12;

class VisibilityBasedPreconditionerTest : public ::testing::Test {
 public:
  static const int kCameraSize = 9;

 protected:
  void SetUp() {
    string input_file = TestFileAbsolutePath("problem-6-1384-000.lsqp");

    scoped_ptr<LinearLeastSquaresProblem> problem(
        CHECK_NOTNULL(CreateLinearLeastSquaresProblemFromFile(input_file)));
    A_.reset(down_cast<BlockSparseMatrix*>(problem->A.release()));
    b_.reset(problem->b.release());
    D_.reset(problem->D.release());

    const CompressedRowBlockStructure* bs =
        CHECK_NOTNULL(A_->block_structure());
    const int num_col_blocks = bs->cols.size();

    num_cols_ = A_->num_cols();
    num_rows_ = A_->num_rows();
    num_eliminate_blocks_ = problem->num_eliminate_blocks;
    num_camera_blocks_ = num_col_blocks - num_eliminate_blocks_;
    options_.elimination_groups.push_back(num_eliminate_blocks_);
    options_.elimination_groups.push_back(
        A_->block_structure()->cols.size() - num_eliminate_blocks_);

    vector<int> blocks(num_col_blocks - num_eliminate_blocks_, 0);
    for (int i = num_eliminate_blocks_; i < num_col_blocks; ++i) {
      blocks[i - num_eliminate_blocks_] = bs->cols[i].size;
    }

    // The input matrix is a real jacobian and fairly poorly
    // conditioned. Setting D to a large constant makes the normal
    // equations better conditioned and makes the tests below better
    // conditioned.
    VectorRef(D_.get(), num_cols_).setConstant(10.0);

    schur_complement_.reset(new BlockRandomAccessDenseMatrix(blocks));
    Vector rhs(schur_complement_->num_rows());

    scoped_ptr<SchurEliminatorBase> eliminator;
    LinearSolver::Options eliminator_options;
    eliminator_options.elimination_groups = options_.elimination_groups;
    eliminator_options.num_threads = options_.num_threads;

    eliminator.reset(SchurEliminatorBase::Create(eliminator_options));
    eliminator->Init(num_eliminate_blocks_, bs);
    eliminator->Eliminate(A_.get(), b_.get(), D_.get(),
                          schur_complement_.get(), rhs.data());
  }


  AssertionResult IsSparsityStructureValid() {
    preconditioner_->InitStorage(*A_->block_structure());
    const HashSet<pair<int, int> >& cluster_pairs = get_cluster_pairs();
    const vector<int>& cluster_membership = get_cluster_membership();

    for (int i = 0; i < num_camera_blocks_; ++i) {
      for (int j = i; j < num_camera_blocks_; ++j) {
        if (cluster_pairs.count(make_pair(cluster_membership[i],
                                          cluster_membership[j]))) {
          if (!IsBlockPairInPreconditioner(i, j)) {
            return AssertionFailure()
                << "block pair (" << i << "," << j << "missing";
          }
        } else {
          if (IsBlockPairInPreconditioner(i, j)) {
            return AssertionFailure()
                << "block pair (" << i << "," << j << "should not be present";
          }
        }
      }
    }
    return AssertionSuccess();
  }

  AssertionResult PreconditionerValuesMatch() {
    preconditioner_->Update(*A_, D_.get());
    const HashSet<pair<int, int> >& cluster_pairs = get_cluster_pairs();
    const BlockRandomAccessSparseMatrix* m = get_m();
    Matrix preconditioner_matrix;
    m->matrix()->ToDenseMatrix(&preconditioner_matrix);
    ConstMatrixRef full_schur_complement(schur_complement_->values(),
                                         m->num_rows(),
                                         m->num_rows());
    const int num_clusters = get_num_clusters();
    const int kDiagonalBlockSize =
        kCameraSize * num_camera_blocks_ / num_clusters;

    for (int i = 0; i < num_clusters; ++i) {
      for (int j = i; j < num_clusters; ++j) {
        double diff = 0.0;
        if (cluster_pairs.count(make_pair(i, j))) {
          diff =
              (preconditioner_matrix.block(kDiagonalBlockSize * i,
                                           kDiagonalBlockSize * j,
                                           kDiagonalBlockSize,
                                           kDiagonalBlockSize) -
               full_schur_complement.block(kDiagonalBlockSize * i,
                                           kDiagonalBlockSize * j,
                                           kDiagonalBlockSize,
                                           kDiagonalBlockSize)).norm();
        } else {
          diff = preconditioner_matrix.block(kDiagonalBlockSize * i,
                                             kDiagonalBlockSize * j,
                                             kDiagonalBlockSize,
                                             kDiagonalBlockSize).norm();
        }
        if (diff > kTolerance) {
          return AssertionFailure()
              << "Preconditioner block " << i << " " << j << " differs "
              << "from expected value by " << diff;
        }
      }
    }
    return AssertionSuccess();
  }

  // Accessors
  int get_num_blocks() { return preconditioner_->num_blocks_; }

  int get_num_clusters() { return preconditioner_->num_clusters_; }
  int* get_mutable_num_clusters() { return &preconditioner_->num_clusters_; }

  const vector<int>& get_block_size() {
    return preconditioner_->block_size_; }

  vector<int>* get_mutable_block_size() {
    return &preconditioner_->block_size_; }

  const vector<int>& get_cluster_membership() {
    return preconditioner_->cluster_membership_;
  }

  vector<int>* get_mutable_cluster_membership() {
    return &preconditioner_->cluster_membership_;
  }

  const set<pair<int, int> >& get_block_pairs() {
    return preconditioner_->block_pairs_;
  }

  set<pair<int, int> >* get_mutable_block_pairs() {
    return &preconditioner_->block_pairs_;
  }

  const HashSet<pair<int, int> >& get_cluster_pairs() {
    return preconditioner_->cluster_pairs_;
  }

  HashSet<pair<int, int> >* get_mutable_cluster_pairs() {
    return &preconditioner_->cluster_pairs_;
  }

  bool IsBlockPairInPreconditioner(const int block1, const int block2) {
    return preconditioner_->IsBlockPairInPreconditioner(block1, block2);
  }

  bool IsBlockPairOffDiagonal(const int block1, const int block2) {
    return preconditioner_->IsBlockPairOffDiagonal(block1, block2);
  }

  const BlockRandomAccessSparseMatrix* get_m() {
    return preconditioner_->m_.get();
  }

  int num_rows_;
  int num_cols_;
  int num_eliminate_blocks_;
  int num_camera_blocks_;

  scoped_ptr<BlockSparseMatrix> A_;
  scoped_array<double> b_;
  scoped_array<double> D_;

  Preconditioner::Options options_;
  scoped_ptr<VisibilityBasedPreconditioner> preconditioner_;
  scoped_ptr<BlockRandomAccessDenseMatrix> schur_complement_;
};

#ifndef CERES_NO_PROTOCOL_BUFFERS
TEST_F(VisibilityBasedPreconditionerTest, OneClusterClusterJacobi) {
  options_.type = CLUSTER_JACOBI;
  preconditioner_.reset(
      new VisibilityBasedPreconditioner(*A_->block_structure(), options_));

  // Override the clustering to be a single clustering containing all
  // the cameras.
  vector<int>& cluster_membership = *get_mutable_cluster_membership();
  for (int i = 0; i < num_camera_blocks_; ++i) {
    cluster_membership[i] = 0;
  }

  *get_mutable_num_clusters() = 1;

  HashSet<pair<int, int> >& cluster_pairs = *get_mutable_cluster_pairs();
  cluster_pairs.clear();
  cluster_pairs.insert(make_pair(0, 0));

  EXPECT_TRUE(IsSparsityStructureValid());
  EXPECT_TRUE(PreconditionerValuesMatch());

  // Multiplication by the inverse of the preconditioner.
  const int num_rows = schur_complement_->num_rows();
  ConstMatrixRef full_schur_complement(schur_complement_->values(),
                                       num_rows,
                                       num_rows);
  Vector x(num_rows);
  Vector y(num_rows);
  Vector z(num_rows);

  for (int i = 0; i < num_rows; ++i) {
    x.setZero();
    y.setZero();
    z.setZero();
    x[i] = 1.0;
    preconditioner_->RightMultiply(x.data(), y.data());
    z = full_schur_complement
        .selfadjointView<Eigen::Upper>()
        .ldlt().solve(x);
    double max_relative_difference =
        ((y - z).array() / z.array()).matrix().lpNorm<Eigen::Infinity>();
    EXPECT_NEAR(max_relative_difference, 0.0, kTolerance);
  }
}



TEST_F(VisibilityBasedPreconditionerTest, ClusterJacobi) {
  options_.type = CLUSTER_JACOBI;
  preconditioner_.reset(
      new VisibilityBasedPreconditioner(*A_->block_structure(), options_));

  // Override the clustering to be equal number of cameras.
  vector<int>& cluster_membership = *get_mutable_cluster_membership();
  cluster_membership.resize(num_camera_blocks_);
  static const int kNumClusters = 3;

  for (int i = 0; i < num_camera_blocks_; ++i) {
    cluster_membership[i] = (i * kNumClusters) / num_camera_blocks_;
  }
  *get_mutable_num_clusters() = kNumClusters;

  HashSet<pair<int, int> >& cluster_pairs = *get_mutable_cluster_pairs();
  cluster_pairs.clear();
  for (int i = 0; i < kNumClusters; ++i) {
    cluster_pairs.insert(make_pair(i, i));
  }

  EXPECT_TRUE(IsSparsityStructureValid());
  EXPECT_TRUE(PreconditionerValuesMatch());
}


TEST_F(VisibilityBasedPreconditionerTest, ClusterTridiagonal) {
  options_.type = CLUSTER_TRIDIAGONAL;
  preconditioner_.reset(
      new VisibilityBasedPreconditioner(*A_->block_structure(), options_));
  static const int kNumClusters = 3;

  // Override the clustering to be 3 clusters.
  vector<int>& cluster_membership = *get_mutable_cluster_membership();
  cluster_membership.resize(num_camera_blocks_);
  for (int i = 0; i < num_camera_blocks_; ++i) {
    cluster_membership[i] = (i * kNumClusters) / num_camera_blocks_;
  }
  *get_mutable_num_clusters() = kNumClusters;

  // Spanning forest has structure 0-1 2
  HashSet<pair<int, int> >& cluster_pairs = *get_mutable_cluster_pairs();
  cluster_pairs.clear();
  for (int i = 0; i < kNumClusters; ++i) {
    cluster_pairs.insert(make_pair(i, i));
  }
  cluster_pairs.insert(make_pair(0, 1));

  EXPECT_TRUE(IsSparsityStructureValid());
  EXPECT_TRUE(PreconditionerValuesMatch());
}
#endif  // CERES_NO_PROTOCOL_BUFFERS

}  // namespace internal
}  // namespace ceres

#endif  // CERES_NO_SUITESPARSE
