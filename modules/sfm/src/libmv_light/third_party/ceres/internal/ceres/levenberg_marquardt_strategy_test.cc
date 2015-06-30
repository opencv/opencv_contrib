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

#include "ceres/internal/eigen.h"
#include "ceres/internal/scoped_ptr.h"
#include "ceres/levenberg_marquardt_strategy.h"
#include "ceres/linear_solver.h"
#include "ceres/trust_region_strategy.h"
#include "glog/logging.h"
#include "gmock/gmock.h"
#include "gmock/mock-log.h"
#include "gtest/gtest.h"

using testing::AllOf;
using testing::AnyNumber;
using testing::HasSubstr;
using testing::ScopedMockLog;
using testing::_;

namespace ceres {
namespace internal {

const double kTolerance = 1e-16;

// Linear solver that takes as input a vector and checks that the
// caller passes the same vector as LinearSolver::PerSolveOptions.D.
class RegularizationCheckingLinearSolver : public DenseSparseMatrixSolver {
 public:
  RegularizationCheckingLinearSolver(const int num_cols, const double* diagonal)
      : num_cols_(num_cols),
        diagonal_(diagonal) {
  }

  virtual ~RegularizationCheckingLinearSolver() {}

 private:
  virtual LinearSolver::Summary SolveImpl(
      DenseSparseMatrix* A,
      const double* b,
      const LinearSolver::PerSolveOptions& per_solve_options,
      double* x) {
    CHECK_NOTNULL(per_solve_options.D);
    for (int i = 0; i < num_cols_; ++i) {
      EXPECT_NEAR(per_solve_options.D[i], diagonal_[i], kTolerance)
          << i << " " << per_solve_options.D[i] << " " << diagonal_[i];
    }
    return LinearSolver::Summary();
  }

  const int num_cols_;
  const double* diagonal_;
};

TEST(LevenbergMarquardtStrategy, AcceptRejectStepRadiusScaling) {
  TrustRegionStrategy::Options options;
  options.initial_radius = 2.0;
  options.max_radius = 20.0;
  options.lm_min_diagonal = 1e-8;
  options.lm_max_diagonal = 1e8;

  // We need a non-null pointer here, so anything should do.
  scoped_ptr<LinearSolver> linear_solver(
      new RegularizationCheckingLinearSolver(0, NULL));
  options.linear_solver = linear_solver.get();

  LevenbergMarquardtStrategy lms(options);
  EXPECT_EQ(lms.Radius(), options.initial_radius);
  lms.StepRejected(0.0);
  EXPECT_EQ(lms.Radius(), 1.0);
  lms.StepRejected(-1.0);
  EXPECT_EQ(lms.Radius(), 0.25);
  lms.StepAccepted(1.0);
  EXPECT_EQ(lms.Radius(), 0.25 * 3.0);
  lms.StepAccepted(1.0);
  EXPECT_EQ(lms.Radius(), 0.25 * 3.0 * 3.0);
  lms.StepAccepted(0.25);
  EXPECT_EQ(lms.Radius(), 0.25 * 3.0 * 3.0 / 1.125);
  lms.StepAccepted(1.0);
  EXPECT_EQ(lms.Radius(), 0.25 * 3.0 * 3.0 / 1.125 * 3.0);
  lms.StepAccepted(1.0);
  EXPECT_EQ(lms.Radius(), 0.25 * 3.0 * 3.0 / 1.125 * 3.0 * 3.0);
  lms.StepAccepted(1.0);
  EXPECT_EQ(lms.Radius(), options.max_radius);
}

TEST(LevenbergMarquardtStrategy, CorrectDiagonalToLinearSolver) {
  Matrix jacobian(2, 3);
  jacobian.setZero();
  jacobian(0, 0) = 0.0;
  jacobian(0, 1) = 1.0;
  jacobian(1, 1) = 1.0;
  jacobian(0, 2) = 100.0;

  double residual = 1.0;
  double x[3];
  DenseSparseMatrix dsm(jacobian);

  TrustRegionStrategy::Options options;
  options.initial_radius = 2.0;
  options.max_radius = 20.0;
  options.lm_min_diagonal = 1e-2;
  options.lm_max_diagonal = 1e2;

  double diagonal[3];
  diagonal[0] = options.lm_min_diagonal;
  diagonal[1] = 2.0;
  diagonal[2] = options.lm_max_diagonal;
  for (int i = 0; i < 3; ++i) {
    diagonal[i] = sqrt(diagonal[i] / options.initial_radius);
  }

  RegularizationCheckingLinearSolver linear_solver(3, diagonal);
  options.linear_solver = &linear_solver;

  LevenbergMarquardtStrategy lms(options);
  TrustRegionStrategy::PerSolveOptions pso;

  {
    ScopedMockLog log;
    EXPECT_CALL(log, Log(_, _, _)).Times(AnyNumber());
    EXPECT_CALL(log, Log(WARNING, _,
                         HasSubstr("Failed to compute a finite step.")));

    TrustRegionStrategy::Summary summary =
        lms.ComputeStep(pso, &dsm, &residual, x);
    EXPECT_EQ(summary.termination_type, FAILURE);
  }
}

}  // namespace internal
}  // namespace ceres
