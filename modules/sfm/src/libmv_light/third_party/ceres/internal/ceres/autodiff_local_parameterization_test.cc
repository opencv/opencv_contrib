// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2013 Google Inc. All rights reserved.
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

#include <cmath>
#include "ceres/autodiff_local_parameterization.h"
#include "ceres/fpclassify.h"
#include "ceres/local_parameterization.h"
#include "ceres/rotation.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

struct IdentityPlus {
  template <typename T>
  bool operator()(const T* x, const T* delta, T* x_plus_delta) const {
    for (int i = 0; i < 3; ++i) {
      x_plus_delta[i] = x[i] + delta[i];
    }
    return true;
  }
};


TEST(AutoDiffLocalParameterizationTest, IdentityParameterization) {
  AutoDiffLocalParameterization<IdentityPlus, 3, 3>
      parameterization;

  double x[3] = {1.0, 2.0, 3.0};
  double delta[3] = {0.0, 1.0, 2.0};
  double x_plus_delta[3] = {0.0, 0.0, 0.0};
  parameterization.Plus(x, delta, x_plus_delta);

  EXPECT_EQ(x_plus_delta[0], 1.0);
  EXPECT_EQ(x_plus_delta[1], 3.0);
  EXPECT_EQ(x_plus_delta[2], 5.0);

  double jacobian[9];
  parameterization.ComputeJacobian(x, jacobian);
  int k = 0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j, ++k) {
      EXPECT_EQ(jacobian[k], (i == j) ? 1.0 : 0.0);
    }
  }
}

struct QuaternionPlus {
  template<typename T>
  bool operator()(const T* x, const T* delta, T* x_plus_delta) const {
    const T squared_norm_delta =
        delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];

    T q_delta[4];
    if (squared_norm_delta > T(0.0)) {
      T norm_delta = sqrt(squared_norm_delta);
      const T sin_delta_by_delta = sin(norm_delta) / norm_delta;
      q_delta[0] = cos(norm_delta);
      q_delta[1] = sin_delta_by_delta * delta[0];
      q_delta[2] = sin_delta_by_delta * delta[1];
      q_delta[3] = sin_delta_by_delta * delta[2];
    } else {
      // We do not just use q_delta = [1,0,0,0] here because that is a
      // constant and when used for automatic differentiation will
      // lead to a zero derivative. Instead we take a first order
      // approximation and evaluate it at zero.
      q_delta[0] = T(1.0);
      q_delta[1] = delta[0];
      q_delta[2] = delta[1];
      q_delta[3] = delta[2];
    }

    QuaternionProduct(q_delta, x, x_plus_delta);
    return true;
  }
};

void QuaternionParameterizationTestHelper(const double* x,
                                          const double* delta) {
  const double kTolerance = 1e-14;
  double x_plus_delta_ref[4] = {0.0, 0.0, 0.0, 0.0};
  double jacobian_ref[12];


  QuaternionParameterization ref_parameterization;
  ref_parameterization.Plus(x, delta, x_plus_delta_ref);
  ref_parameterization.ComputeJacobian(x, jacobian_ref);

  double x_plus_delta[4] = {0.0, 0.0, 0.0, 0.0};
  double jacobian[12];
  AutoDiffLocalParameterization<QuaternionPlus, 4, 3> parameterization;
  parameterization.Plus(x, delta, x_plus_delta);
  parameterization.ComputeJacobian(x, jacobian);

  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(x_plus_delta[i], x_plus_delta_ref[i], kTolerance);
  }

  const double x_plus_delta_norm =
      sqrt(x_plus_delta[0] * x_plus_delta[0] +
           x_plus_delta[1] * x_plus_delta[1] +
           x_plus_delta[2] * x_plus_delta[2] +
           x_plus_delta[3] * x_plus_delta[3]);

  EXPECT_NEAR(x_plus_delta_norm, 1.0, kTolerance);

  for (int i = 0; i < 12; ++i) {
    EXPECT_TRUE(IsFinite(jacobian[i]));
    EXPECT_NEAR(jacobian[i], jacobian_ref[i], kTolerance)
        << "Jacobian mismatch: i = " << i
        << "\n Expected \n" << ConstMatrixRef(jacobian_ref, 4, 3)
        << "\n Actual \n" << ConstMatrixRef(jacobian, 4, 3);
  }
}

TEST(AutoDiffLocalParameterization, QuaternionParameterizationZeroTest) {
  double x[4] = {0.5, 0.5, 0.5, 0.5};
  double delta[3] = {0.0, 0.0, 0.0};
  QuaternionParameterizationTestHelper(x, delta);
}


TEST(AutoDiffLocalParameterization, QuaternionParameterizationNearZeroTest) {
  double x[4] = {0.52, 0.25, 0.15, 0.45};
  double norm_x = sqrt(x[0] * x[0] +
                       x[1] * x[1] +
                       x[2] * x[2] +
                       x[3] * x[3]);
  for (int i = 0; i < 4; ++i) {
    x[i] = x[i] / norm_x;
  }

  double delta[3] = {0.24, 0.15, 0.10};
  for (int i = 0; i < 3; ++i) {
    delta[i] = delta[i] * 1e-14;
  }

  QuaternionParameterizationTestHelper(x, delta);
}

TEST(AutoDiffLocalParameterization, QuaternionParameterizationNonZeroTest) {
  double x[4] = {0.52, 0.25, 0.15, 0.45};
  double norm_x = sqrt(x[0] * x[0] +
                       x[1] * x[1] +
                       x[2] * x[2] +
                       x[3] * x[3]);

  for (int i = 0; i < 4; ++i) {
    x[i] = x[i] / norm_x;
  }

  double delta[3] = {0.24, 0.15, 0.10};
  QuaternionParameterizationTestHelper(x, delta);
}

}  // namespace internal
}  // namespace ceres
