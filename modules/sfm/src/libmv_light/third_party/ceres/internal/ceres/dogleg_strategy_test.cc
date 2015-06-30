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
// Author: moll.markus@arcor.de (Markus Moll)

#include <limits>
#include "ceres/internal/eigen.h"
#include "ceres/internal/scoped_ptr.h"
#include "ceres/dense_qr_solver.h"
#include "ceres/dogleg_strategy.h"
#include "ceres/linear_solver.h"
#include "ceres/trust_region_strategy.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {
namespace {

class Fixture : public testing::Test {
 protected:
  scoped_ptr<DenseSparseMatrix> jacobian_;
  Vector residual_;
  Vector x_;
  TrustRegionStrategy::Options options_;
};

// A test problem where
//
//   J^T J = Q diag([1 2 4 8 16 32]) Q^T
//
// where Q is a randomly chosen orthonormal basis of R^6.
// The residual is chosen so that the minimum of the quadratic function is
// at (1, 1, 1, 1, 1, 1). It is therefore at a distance of sqrt(6) ~ 2.45
// from the origin.
class DoglegStrategyFixtureEllipse : public Fixture {
 protected:
  virtual void SetUp() {
    Matrix basis(6, 6);
    // The following lines exceed 80 characters for better readability.
    basis << -0.1046920933796121, -0.7449367449921986, -0.4190744502875876, -0.4480450716142566,  0.2375351607929440, -0.0363053418882862,
              0.4064975684355914,  0.2681113508511354, -0.7463625494601520, -0.0803264850508117, -0.4463149623021321,  0.0130224954867195,
             -0.5514387729089798,  0.1026621026168657, -0.5008316122125011,  0.5738122212666414,  0.2974664724007106,  0.1296020877535158,
              0.5037835370947156,  0.2668479925183712, -0.1051754618492798, -0.0272739396578799,  0.7947481647088278, -0.1776623363955670,
             -0.4005458426625444,  0.2939330589634109, -0.0682629380550051, -0.2895448882503687, -0.0457239396341685, -0.8139899477847840,
             -0.3247764582762654,  0.4528151365941945, -0.0276683863102816, -0.6155994592510784,  0.1489240599972848,  0.5362574892189350;

    Vector Ddiag(6);
    Ddiag << 1.0, 2.0, 4.0, 8.0, 16.0, 32.0;

    Matrix sqrtD = Ddiag.array().sqrt().matrix().asDiagonal();
    Matrix jacobian = sqrtD * basis;
    jacobian_.reset(new DenseSparseMatrix(jacobian));

    Vector minimum(6);
    minimum << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0;
    residual_ = -jacobian * minimum;

    x_.resize(6);
    x_.setZero();

    options_.lm_min_diagonal = 1.0;
    options_.lm_max_diagonal = 1.0;
  }
};

// A test problem where
//
//   J^T J = diag([1 2 4 8 16 32]) .
//
// The residual is chosen so that the minimum of the quadratic function is
// at (0, 0, 1, 0, 0, 0). It is therefore at a distance of 1 from the origin.
// The gradient at the origin points towards the global minimum.
class DoglegStrategyFixtureValley : public Fixture {
 protected:
  virtual void SetUp() {
    Vector Ddiag(6);
    Ddiag << 1.0, 2.0, 4.0, 8.0, 16.0, 32.0;

    Matrix jacobian = Ddiag.asDiagonal();
    jacobian_.reset(new DenseSparseMatrix(jacobian));

    Vector minimum(6);
    minimum << 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
    residual_ = -jacobian * minimum;

    x_.resize(6);
    x_.setZero();

    options_.lm_min_diagonal = 1.0;
    options_.lm_max_diagonal = 1.0;
  }
};

const double kTolerance = 1e-14;
const double kToleranceLoose = 1e-5;
const double kEpsilon = std::numeric_limits<double>::epsilon();

}  // namespace

// The DoglegStrategy must never return a step that is longer than the current
// trust region radius.
TEST_F(DoglegStrategyFixtureEllipse, TrustRegionObeyedTraditional) {
  scoped_ptr<LinearSolver> linear_solver(
      new DenseQRSolver(LinearSolver::Options()));
  options_.linear_solver = linear_solver.get();
  // The global minimum is at (1, 1, ..., 1), so the distance to it is
  // sqrt(6.0).  By restricting the trust region to a radius of 2.0,
  // we test if the trust region is actually obeyed.
  options_.dogleg_type = TRADITIONAL_DOGLEG;
  options_.initial_radius = 2.0;
  options_.max_radius = 2.0;

  DoglegStrategy strategy(options_);
  TrustRegionStrategy::PerSolveOptions pso;

  TrustRegionStrategy::Summary summary = strategy.ComputeStep(pso,
                                                              jacobian_.get(),
                                                              residual_.data(),
                                                              x_.data());

  EXPECT_NE(summary.termination_type, FAILURE);
  EXPECT_LE(x_.norm(), options_.initial_radius * (1.0 + 4.0 * kEpsilon));
}

TEST_F(DoglegStrategyFixtureEllipse, TrustRegionObeyedSubspace) {
  scoped_ptr<LinearSolver> linear_solver(
      new DenseQRSolver(LinearSolver::Options()));
  options_.linear_solver = linear_solver.get();
  options_.dogleg_type = SUBSPACE_DOGLEG;
  options_.initial_radius = 2.0;
  options_.max_radius = 2.0;

  DoglegStrategy strategy(options_);
  TrustRegionStrategy::PerSolveOptions pso;

  TrustRegionStrategy::Summary summary = strategy.ComputeStep(pso,
                                                              jacobian_.get(),
                                                              residual_.data(),
                                                              x_.data());

  EXPECT_NE(summary.termination_type, FAILURE);
  EXPECT_LE(x_.norm(), options_.initial_radius * (1.0 + 4.0 * kEpsilon));
}

TEST_F(DoglegStrategyFixtureEllipse, CorrectGaussNewtonStep) {
  scoped_ptr<LinearSolver> linear_solver(
      new DenseQRSolver(LinearSolver::Options()));
  options_.linear_solver = linear_solver.get();
  options_.dogleg_type = SUBSPACE_DOGLEG;
  options_.initial_radius = 10.0;
  options_.max_radius = 10.0;

  DoglegStrategy strategy(options_);
  TrustRegionStrategy::PerSolveOptions pso;

  TrustRegionStrategy::Summary summary = strategy.ComputeStep(pso,
                                                              jacobian_.get(),
                                                              residual_.data(),
                                                              x_.data());

  EXPECT_NE(summary.termination_type, FAILURE);
  EXPECT_NEAR(x_(0), 1.0, kToleranceLoose);
  EXPECT_NEAR(x_(1), 1.0, kToleranceLoose);
  EXPECT_NEAR(x_(2), 1.0, kToleranceLoose);
  EXPECT_NEAR(x_(3), 1.0, kToleranceLoose);
  EXPECT_NEAR(x_(4), 1.0, kToleranceLoose);
  EXPECT_NEAR(x_(5), 1.0, kToleranceLoose);
}

// Test if the subspace basis is a valid orthonormal basis of the space spanned
// by the gradient and the Gauss-Newton point.
TEST_F(DoglegStrategyFixtureEllipse, ValidSubspaceBasis) {
  scoped_ptr<LinearSolver> linear_solver(
      new DenseQRSolver(LinearSolver::Options()));
  options_.linear_solver = linear_solver.get();
  options_.dogleg_type = SUBSPACE_DOGLEG;
  options_.initial_radius = 2.0;
  options_.max_radius = 2.0;

  DoglegStrategy strategy(options_);
  TrustRegionStrategy::PerSolveOptions pso;

  strategy.ComputeStep(pso, jacobian_.get(), residual_.data(), x_.data());

  // Check if the basis is orthonormal.
  const Matrix basis = strategy.subspace_basis();
  EXPECT_NEAR(basis.col(0).norm(), 1.0, kTolerance);
  EXPECT_NEAR(basis.col(1).norm(), 1.0, kTolerance);
  EXPECT_NEAR(basis.col(0).dot(basis.col(1)), 0.0, kTolerance);

  // Check if the gradient projects onto itself.
  const Vector gradient = strategy.gradient();
  EXPECT_NEAR((gradient - basis*(basis.transpose()*gradient)).norm(),
              0.0,
              kTolerance);

  // Check if the Gauss-Newton point projects onto itself.
  const Vector gn = strategy.gauss_newton_step();
  EXPECT_NEAR((gn - basis*(basis.transpose()*gn)).norm(),
              0.0,
              kTolerance);
}

// Test if the step is correct if the gradient and the Gauss-Newton step point
// in the same direction and the Gauss-Newton step is outside the trust region,
// i.e. the trust region is active.
TEST_F(DoglegStrategyFixtureValley, CorrectStepLocalOptimumAlongGradient) {
  scoped_ptr<LinearSolver> linear_solver(
      new DenseQRSolver(LinearSolver::Options()));
  options_.linear_solver = linear_solver.get();
  options_.dogleg_type = SUBSPACE_DOGLEG;
  options_.initial_radius = 0.25;
  options_.max_radius = 0.25;

  DoglegStrategy strategy(options_);
  TrustRegionStrategy::PerSolveOptions pso;

  TrustRegionStrategy::Summary summary = strategy.ComputeStep(pso,
                                                              jacobian_.get(),
                                                              residual_.data(),
                                                              x_.data());

  EXPECT_NE(summary.termination_type, FAILURE);
  EXPECT_NEAR(x_(0), 0.0, kToleranceLoose);
  EXPECT_NEAR(x_(1), 0.0, kToleranceLoose);
  EXPECT_NEAR(x_(2), options_.initial_radius, kToleranceLoose);
  EXPECT_NEAR(x_(3), 0.0, kToleranceLoose);
  EXPECT_NEAR(x_(4), 0.0, kToleranceLoose);
  EXPECT_NEAR(x_(5), 0.0, kToleranceLoose);
}

// Test if the step is correct if the gradient and the Gauss-Newton step point
// in the same direction and the Gauss-Newton step is inside the trust region,
// i.e. the trust region is inactive.
TEST_F(DoglegStrategyFixtureValley, CorrectStepGlobalOptimumAlongGradient) {
  scoped_ptr<LinearSolver> linear_solver(
      new DenseQRSolver(LinearSolver::Options()));
  options_.linear_solver = linear_solver.get();
  options_.dogleg_type = SUBSPACE_DOGLEG;
  options_.initial_radius = 2.0;
  options_.max_radius = 2.0;

  DoglegStrategy strategy(options_);
  TrustRegionStrategy::PerSolveOptions pso;

  TrustRegionStrategy::Summary summary = strategy.ComputeStep(pso,
                                                              jacobian_.get(),
                                                              residual_.data(),
                                                              x_.data());

  EXPECT_NE(summary.termination_type, FAILURE);
  EXPECT_NEAR(x_(0), 0.0, kToleranceLoose);
  EXPECT_NEAR(x_(1), 0.0, kToleranceLoose);
  EXPECT_NEAR(x_(2), 1.0, kToleranceLoose);
  EXPECT_NEAR(x_(3), 0.0, kToleranceLoose);
  EXPECT_NEAR(x_(4), 0.0, kToleranceLoose);
  EXPECT_NEAR(x_(5), 0.0, kToleranceLoose);
}

}  // namespace internal
}  // namespace ceres
