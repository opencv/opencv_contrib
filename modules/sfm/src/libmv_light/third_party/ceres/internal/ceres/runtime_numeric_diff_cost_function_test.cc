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
// Author: keir@google.com (Keir Mierle)
//
// Based on the tests in numeric_diff_cost_function.cc.
//
// TODO(keir): See about code duplication.

#include "ceres/runtime_numeric_diff_cost_function.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>
#include "ceres/cost_function.h"
#include "ceres/internal/macros.h"
#include "ceres/internal/scoped_ptr.h"
#include "ceres/stringprintf.h"
#include "ceres/test_util.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

const double kRelativeEps = 1e-6;

// y1 = x1'x2      -> dy1/dx1 = x2,               dy1/dx2 = x1
// y2 = (x1'x2)^2  -> dy2/dx1 = 2 * x2 * (x1'x2), dy2/dx2 = 2 * x1 * (x1'x2)
// y3 = x2'x2      -> dy3/dx1 = 0,                dy3/dx2 = 2 * x2
class TestCostFunction : public CostFunction {
 public:
  TestCostFunction() {
    set_num_residuals(3);
    mutable_parameter_block_sizes()->push_back(5);  // x1.
    mutable_parameter_block_sizes()->push_back(5);  // x2.
  }
  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    (void) jacobians;  // Ignored.

    residuals[0] = residuals[1] = residuals[2] = 0;
    for (int i = 0; i < 5; ++i) {
      residuals[0] += parameters[0][i] * parameters[1][i];
      residuals[2] += parameters[1][i] * parameters[1][i];
    }
    residuals[1] = residuals[0] * residuals[0];
    return true;
  }
};

TEST(NumericDiffCostFunction, EasyCase) {
  // Try both central and forward difference.
  TestCostFunction term;
  scoped_ptr<CostFunction> cfs[2];
  cfs[0].reset(
      CreateRuntimeNumericDiffCostFunction(&term, CENTRAL, kRelativeEps));

  cfs[1].reset(
      CreateRuntimeNumericDiffCostFunction(&term, FORWARD, kRelativeEps));


  for (int c = 0; c < 2; ++c) {
    CostFunction *cost_function = cfs[c].get();

    double x1[] = { 1.0, 2.0, 3.0, 4.0, 5.0 };
    double x2[] = { 9.0, 9.0, 5.0, 5.0, 1.0 };
    double *parameters[] = { &x1[0], &x2[0] };

    double dydx1[15];  // 3 x 5, row major.
    double dydx2[15];  // 3 x 5, row major.
    double *jacobians[2] = { &dydx1[0], &dydx2[0] };

    double residuals[3] = {-1e-100, -2e-100, -3e-100 };

    ASSERT_TRUE(cost_function->Evaluate(&parameters[0],
                                        &residuals[0],
                                        &jacobians[0]));

    EXPECT_EQ(residuals[0], 67);
    EXPECT_EQ(residuals[1], 4489);
    EXPECT_EQ(residuals[2], 213);

    for (int i = 0; i < 5; ++i) {
      LOG(INFO) << "c = " << c << " i = " << i;
      const double kEps = c == 0 ? /* central */ 3e-9 : /* forward */ 2e-5;

      ExpectClose(x2[i],                    dydx1[5 * 0 + i], kEps);  // y1
      ExpectClose(x1[i],                    dydx2[5 * 0 + i], kEps);
      ExpectClose(2 * x2[i] * residuals[0], dydx1[5 * 1 + i], kEps);  // y2
      ExpectClose(2 * x1[i] * residuals[0], dydx2[5 * 1 + i], kEps);
      ExpectClose(0.0,                      dydx1[5 * 2 + i], kEps);  // y3
      ExpectClose(2 * x2[i],                dydx2[5 * 2 + i], kEps);
    }
  }
}

// y1 = sin(x1'x2)
// y2 = exp(-x1'x2 / 10)
//
// dy1/dx1 =  x2 * cos(x1'x2),            dy1/dx2 =  x1 * cos(x1'x2)
// dy2/dx1 = -x2 * exp(-x1'x2 / 10) / 10, dy2/dx2 = -x2 * exp(-x1'x2 / 10) / 10
class TranscendentalTestCostFunction : public CostFunction {
 public:
  TranscendentalTestCostFunction() {
    set_num_residuals(2);
    mutable_parameter_block_sizes()->push_back(5);  // x1.
    mutable_parameter_block_sizes()->push_back(5);  // x2.
  }
  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    (void) jacobians;  // Ignored.

    double x1x2 = 0;
    for (int i = 0; i < 5; ++i) {
      x1x2 += parameters[0][i] * parameters[1][i];
    }
    residuals[0] = sin(x1x2);
    residuals[1] = exp(-x1x2 / 10);
    return true;
  }
};

TEST(NumericDiffCostFunction, TransendentalOperationsInCostFunction) {
  // Try both central and forward difference.
  TranscendentalTestCostFunction term;
  scoped_ptr<CostFunction> cfs[2];
  cfs[0].reset(
      CreateRuntimeNumericDiffCostFunction(&term, CENTRAL, kRelativeEps));

  cfs[1].reset(
      CreateRuntimeNumericDiffCostFunction(&term, FORWARD, kRelativeEps));

  for (int c = 0; c < 2; ++c) {
    CostFunction *cost_function = cfs[c].get();

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

      ASSERT_TRUE(cost_function->Evaluate(&parameters[0],
                                          &residuals[0],
                                          &jacobians[0]));
      LOG(INFO) << "Ran evaluate for test k=" << k << " c=" << c;

      double x1x2 = 0;
      for (int i = 0; i < 5; ++i) {
        x1x2 += x1[i] * x2[i];
      }

      for (int i = 0; i < 5; ++i) {
        const double kEps = (c == 0 ? /* central */ 3e-9 : /* forward */ 2e-5);

        ExpectClose( x2[i] * cos(x1x2),              dydx1[5 * 0 + i], kEps);  // NOLINT
        ExpectClose( x1[i] * cos(x1x2),              dydx2[5 * 0 + i], kEps);  // NOLINT
        ExpectClose(-x2[i] * exp(-x1x2 / 10.) / 10., dydx1[5 * 1 + i], kEps);
        ExpectClose(-x1[i] * exp(-x1x2 / 10.) / 10., dydx2[5 * 1 + i], kEps);
      }
    }
  }
}

}  // namespace internal
}  // namespace ceres
