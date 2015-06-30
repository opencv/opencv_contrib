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

#include "ceres/numeric_diff_test_utils.h"

#include <algorithm>
#include <cmath>
#include "ceres/cost_function.h"
#include "ceres/internal/macros.h"
#include "ceres/test_util.h"
#include "ceres/types.h"
#include "gtest/gtest.h"


namespace ceres {
namespace internal {

bool EasyFunctor::operator()(const double* x1,
                             const double* x2,
                             double* residuals) const {
  residuals[0] = residuals[1] = residuals[2] = 0;
  for (int i = 0; i < 5; ++i) {
    residuals[0] += x1[i] * x2[i];
    residuals[2] += x2[i] * x2[i];
  }
  residuals[1] = residuals[0] * residuals[0];
  return true;
}

void EasyFunctor::ExpectCostFunctionEvaluationIsNearlyCorrect(
    const CostFunction& cost_function,
    NumericDiffMethod method) const {
  double x1[] = { 1.0, 2.0, 3.0, 4.0, 5.0 };
  double x2[] = { 9.0, 9.0, 5.0, 5.0, 1.0 };
  double *parameters[] = { &x1[0], &x2[0] };

  double dydx1[15];  // 3 x 5, row major.
  double dydx2[15];  // 3 x 5, row major.
  double *jacobians[2] = { &dydx1[0], &dydx2[0] };

  double residuals[3] = {-1e-100, -2e-100, -3e-100 };

  ASSERT_TRUE(cost_function.Evaluate(&parameters[0],
                                     &residuals[0],
                                     &jacobians[0]));

  EXPECT_EQ(residuals[0], 67);
  EXPECT_EQ(residuals[1], 4489);
  EXPECT_EQ(residuals[2], 213);

  const double tolerance = (method == CENTRAL)? 3e-9 : 2e-5;

  for (int i = 0; i < 5; ++i) {
    ExpectClose(x2[i],                    dydx1[5 * 0 + i], tolerance);  // y1
    ExpectClose(x1[i],                    dydx2[5 * 0 + i], tolerance);
    ExpectClose(2 * x2[i] * residuals[0], dydx1[5 * 1 + i], tolerance);  // y2
    ExpectClose(2 * x1[i] * residuals[0], dydx2[5 * 1 + i], tolerance);
    ExpectClose(0.0,                      dydx1[5 * 2 + i], tolerance);  // y3
    ExpectClose(2 * x2[i],                dydx2[5 * 2 + i], tolerance);
  }
}

bool TranscendentalFunctor::operator()(const double* x1,
                                       const double* x2,
                                       double* residuals) const {
  double x1x2 = 0;
  for (int i = 0; i < 5; ++i) {
    x1x2 += x1[i] * x2[i];
  }
  residuals[0] = sin(x1x2);
  residuals[1] = exp(-x1x2 / 10);
  return true;
}

void TranscendentalFunctor::ExpectCostFunctionEvaluationIsNearlyCorrect(
    const CostFunction& cost_function,
    NumericDiffMethod method) const {
  struct {
    double x1[5];
    double x2[5];
  } kTests[] = {
    { { 1.0, 2.0, 3.0, 4.0, 5.0 },  // No zeros.
      { 9.0, 9.0, 5.0, 5.0, 1.0 },
    },
    { { 0.0, 2.0, 3.0, 0.0, 5.0 },  // Some zeros x1.
      { 9.0, 9.0, 5.0, 5.0, 1.0 },
    },
    { { 1.0, 2.0, 3.0, 1.0, 5.0 },  // Some zeros x2.
      { 0.0, 9.0, 0.0, 5.0, 0.0 },
    },
    { { 0.0, 0.0, 0.0, 0.0, 0.0 },  // All zeros x1.
      { 9.0, 9.0, 5.0, 5.0, 1.0 },
    },
    { { 1.0, 2.0, 3.0, 4.0, 5.0 },  // All zeros x2.
      { 0.0, 0.0, 0.0, 0.0, 0.0 },
    },
    { { 0.0, 0.0, 0.0, 0.0, 0.0 },  // All zeros.
      { 0.0, 0.0, 0.0, 0.0, 0.0 },
    },
  };

  for (int k = 0; k < CERES_ARRAYSIZE(kTests); ++k) {
    double *x1 = &(kTests[k].x1[0]);
    double *x2 = &(kTests[k].x2[0]);
    double *parameters[] = { x1, x2 };

    double dydx1[10];
    double dydx2[10];
    double *jacobians[2] = { &dydx1[0], &dydx2[0] };

    double residuals[2];

    ASSERT_TRUE(cost_function.Evaluate(&parameters[0],
                                       &residuals[0],
                                       &jacobians[0]));
    double x1x2 = 0;
    for (int i = 0; i < 5; ++i) {
      x1x2 += x1[i] * x2[i];
    }

    const double tolerance = (method == CENTRAL)? 3e-9 : 2e-5;

    for (int i = 0; i < 5; ++i) {
      ExpectClose( x2[i] * cos(x1x2),              dydx1[5 * 0 + i], tolerance);
      ExpectClose( x1[i] * cos(x1x2),              dydx2[5 * 0 + i], tolerance);
      ExpectClose(-x2[i] * exp(-x1x2 / 10.) / 10., dydx1[5 * 1 + i], tolerance);
      ExpectClose(-x1[i] * exp(-x1x2 / 10.) / 10., dydx2[5 * 1 + i], tolerance);
    }
  }
}

}  // namespace internal
}  // namespace ceres
