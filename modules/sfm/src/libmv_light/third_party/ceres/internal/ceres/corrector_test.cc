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

#include "ceres/corrector.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include "gtest/gtest.h"
#include "ceres/random.h"
#include "ceres/internal/eigen.h"

namespace ceres {
namespace internal {

// If rho[1] is zero, the Corrector constructor should crash.
TEST(Corrector, ZeroGradientDeathTest) {
  const double kRho[] = {0.0, 0.0, 0.0};
  EXPECT_DEATH_IF_SUPPORTED({Corrector c(1.0, kRho);},
               ".*");
}

// If rho[1] is negative, the Corrector constructor should crash.
TEST(Corrector, NegativeGradientDeathTest) {
  const double kRho[] = {0.0, -0.1, 0.0};
  EXPECT_DEATH_IF_SUPPORTED({Corrector c(1.0, kRho);},
               ".*");
}

TEST(Corrector, ScalarCorrection) {
  double residuals = sqrt(3.0);
  double jacobian = 10.0;
  double sq_norm = residuals * residuals;

  const double kRho[] = {sq_norm, 0.1, -0.01};

  // In light of the rho'' < 0 clamping now implemented in
  // corrector.cc, alpha = 0 whenever rho'' < 0.
  const double kAlpha = 0.0;

  // Thus the expected value of the residual is
  // residual[i] * sqrt(kRho[1]) / (1.0 - kAlpha).
  const double kExpectedResidual =
      residuals * sqrt(kRho[1]) / (1 - kAlpha);

  // The jacobian in this case will be
  // sqrt(kRho[1]) * (1 - kAlpha) * jacobian.
  const double kExpectedJacobian = sqrt(kRho[1]) * (1 - kAlpha) * jacobian;

  Corrector c(sq_norm, kRho);
  c.CorrectJacobian(1.0, 1.0, &residuals, &jacobian);
  c.CorrectResiduals(1.0, &residuals);

  ASSERT_NEAR(residuals, kExpectedResidual, 1e-6);
  ASSERT_NEAR(kExpectedJacobian, jacobian, 1e-6);
}

TEST(Corrector, ScalarCorrectionZeroResidual) {
  double residuals = 0.0;
  double jacobian = 10.0;
  double sq_norm = residuals * residuals;

  const double kRho[] = {0.0, 0.1, -0.01};
  Corrector c(sq_norm, kRho);

  // The alpha equation is
  // 1/2 alpha^2 - alpha + 0.0 = 0.
  // i.e. alpha = 1.0 - sqrt(1.0).
  //      alpha = 0.0.
  // Thus the expected value of the residual is
  // residual[i] * sqrt(kRho[1])
  const double kExpectedResidual = residuals * sqrt(kRho[1]);

  // The jacobian in this case will be
  // sqrt(kRho[1]) * jacobian.
  const double kExpectedJacobian = sqrt(kRho[1]) * jacobian;

  c.CorrectJacobian(1, 1, &residuals, &jacobian);
  c.CorrectResiduals(1, &residuals);

  ASSERT_NEAR(residuals, kExpectedResidual, 1e-6);
  ASSERT_NEAR(kExpectedJacobian, jacobian, 1e-6);
}

// Scaling behaviour for one dimensional functions.
TEST(Corrector, ScalarCorrectionAlphaClamped) {
  double residuals = sqrt(3.0);
  double jacobian = 10.0;
  double sq_norm = residuals * residuals;

  const double kRho[] = {3, 0.1, -0.1};

  // rho[2] < 0 -> alpha = 0.0
  const double kAlpha = 0.0;

  // Thus the expected value of the residual is
  // residual[i] * sqrt(kRho[1]) / (1.0 - kAlpha).
  const double kExpectedResidual =
      residuals * sqrt(kRho[1]) / (1.0 - kAlpha);

  // The jacobian in this case will be scaled by
  // sqrt(rho[1]) * (1 - alpha) * J.
  const double kExpectedJacobian = sqrt(kRho[1]) *
      (1.0 - kAlpha) * jacobian;

  Corrector c(sq_norm, kRho);
  c.CorrectJacobian(1, 1, &residuals, &jacobian);
  c.CorrectResiduals(1, &residuals);

  ASSERT_NEAR(residuals, kExpectedResidual, 1e-6);
  ASSERT_NEAR(kExpectedJacobian, jacobian, 1e-6);
}

// Test that the corrected multidimensional residual and jacobians
// match the expected values and the resulting modified normal
// equations match the robustified gauss newton approximation.
TEST(Corrector, MultidimensionalGaussNewtonApproximation) {
  double residuals[3];
  double jacobian[2 * 3];
  double rho[3];

  // Eigen matrix references for linear algebra.
  MatrixRef jac(jacobian, 3, 2);
  VectorRef res(residuals, 3);

  // Ground truth values of the modified jacobian and residuals.
  Matrix g_jac(3, 2);
  Vector g_res(3);

  // Ground truth values of the robustified Gauss-Newton
  // approximation.
  Matrix g_hess(2, 2);
  Vector g_grad(2);

  // Corrected hessian and gradient implied by the modified jacobian
  // and hessians.
  Matrix c_hess(2, 2);
  Vector c_grad(2);

  srand(5);
  for (int iter = 0; iter < 10000; ++iter) {
    // Initialize the jacobian and residual.
    for (int i = 0; i < 2 * 3; ++i)
      jacobian[i] = RandDouble();
    for (int i = 0; i < 3; ++i)
      residuals[i] = RandDouble();

    const double sq_norm = res.dot(res);

    rho[0] = sq_norm;
    rho[1] = RandDouble();
    rho[2] = 2.0 * RandDouble() - 1.0;

    // If rho[2] > 0, then the curvature correction to the correction
    // and the gauss newton approximation will match. Otherwise, we
    // will clamp alpha to 0.

    const double kD = 1 + 2 * rho[2] / rho[1] * sq_norm;
    const double kAlpha = (rho[2] > 0.0) ? 1 - sqrt(kD) : 0.0;

    // Ground truth values.
    g_res = sqrt(rho[1]) / (1.0 - kAlpha) * res;
    g_jac = sqrt(rho[1]) * (jac - kAlpha / sq_norm *
                            res * res.transpose() * jac);

    g_grad = rho[1] * jac.transpose() * res;
    g_hess = rho[1] * jac.transpose() * jac +
        2.0 * rho[2] * jac.transpose() * res * res.transpose() * jac;

    Corrector c(sq_norm, rho);
    c.CorrectJacobian(3, 2, residuals, jacobian);
    c.CorrectResiduals(3, residuals);

    // Corrected gradient and hessian.
    c_grad  = jac.transpose() * res;
    c_hess = jac.transpose() * jac;

    ASSERT_NEAR((g_res - res).norm(), 0.0, 1e-10);
    ASSERT_NEAR((g_jac - jac).norm(), 0.0, 1e-10);

    ASSERT_NEAR((g_grad - c_grad).norm(), 0.0, 1e-10);
  }
}

TEST(Corrector, MultidimensionalGaussNewtonApproximationZeroResidual) {
  double residuals[3];
  double jacobian[2 * 3];
  double rho[3];

  // Eigen matrix references for linear algebra.
  MatrixRef jac(jacobian, 3, 2);
  VectorRef res(residuals, 3);

  // Ground truth values of the modified jacobian and residuals.
  Matrix g_jac(3, 2);
  Vector g_res(3);

  // Ground truth values of the robustified Gauss-Newton
  // approximation.
  Matrix g_hess(2, 2);
  Vector g_grad(2);

  // Corrected hessian and gradient implied by the modified jacobian
  // and hessians.
  Matrix c_hess(2, 2);
  Vector c_grad(2);

  srand(5);
  for (int iter = 0; iter < 10000; ++iter) {
    // Initialize the jacobian.
    for (int i = 0; i < 2 * 3; ++i)
      jacobian[i] = RandDouble();

    // Zero residuals
    res.setZero();

    const double sq_norm = res.dot(res);

    rho[0] = sq_norm;
    rho[1] = RandDouble();
    rho[2] = 2 * RandDouble() - 1.0;

    // Ground truth values.
    g_res = sqrt(rho[1]) * res;
    g_jac = sqrt(rho[1]) * jac;

    g_grad = rho[1] * jac.transpose() * res;
    g_hess = rho[1] * jac.transpose() * jac +
        2.0 * rho[2] * jac.transpose() * res * res.transpose() * jac;

    Corrector c(sq_norm, rho);
    c.CorrectJacobian(3, 2, residuals, jacobian);
    c.CorrectResiduals(3, residuals);

    // Corrected gradient and hessian.
    c_grad = jac.transpose() * res;
    c_hess = jac.transpose() * jac;

    ASSERT_NEAR((g_res - res).norm(), 0.0, 1e-10);
    ASSERT_NEAR((g_jac - jac).norm(), 0.0, 1e-10);

    ASSERT_NEAR((g_grad - c_grad).norm(), 0.0, 1e-10);
    ASSERT_NEAR((g_hess - c_hess).norm(), 0.0, 1e-10);
  }
}

}  // namespace internal
}  // namespace ceres
