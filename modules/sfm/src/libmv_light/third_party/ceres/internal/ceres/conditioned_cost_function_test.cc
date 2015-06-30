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
// Author: wjr@google.com (William Rucklidge)
//
// Tests for the conditioned cost function.

#include "ceres/conditioned_cost_function.h"

#include "ceres/internal/eigen.h"
#include "ceres/normal_prior.h"
#include "ceres/types.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

// The size of the cost functions we build.
static const int kTestCostFunctionSize = 3;

// A simple cost function: return ax + b.
class LinearCostFunction : public CostFunction {
 public:
  LinearCostFunction(double a, double b) : a_(a), b_(b) {
    set_num_residuals(1);
    mutable_parameter_block_sizes()->push_back(1);
  }

  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    *residuals = **parameters * a_ + b_;
    if (jacobians && *jacobians) {
      **jacobians = a_;
    }

    return true;
  }

 private:
  const double a_, b_;
};

// Tests that ConditionedCostFunction does what it's supposed to.
TEST(CostFunctionTest, ConditionedCostFunction) {
  double v1[kTestCostFunctionSize], v2[kTestCostFunctionSize],
      jac[kTestCostFunctionSize * kTestCostFunctionSize],
      result[kTestCostFunctionSize];

  for (int i = 0; i < kTestCostFunctionSize; i++) {
    v1[i] = i;
    v2[i] = i * 10;
    // Seed a few garbage values in the Jacobian matrix, to make sure that
    // they're overwritten.
    jac[i * 2] = i * i;
    result[i] = i * i * i;
  }

  // Make a cost function that computes x - v2
  VectorRef v2_vector(v2, kTestCostFunctionSize, 1);
  Matrix identity(kTestCostFunctionSize, kTestCostFunctionSize);
  identity.setIdentity();
  NormalPrior* difference_cost_function = new NormalPrior(identity, v2_vector);

  vector<CostFunction*> conditioners;
  for (int i = 0; i < kTestCostFunctionSize; i++) {
    conditioners.push_back(new LinearCostFunction(i + 2, i * 7));
  }

  ConditionedCostFunction conditioned_cost_function(difference_cost_function,
                                                    conditioners,
                                                    TAKE_OWNERSHIP);
  EXPECT_EQ(difference_cost_function->num_residuals(),
            conditioned_cost_function.num_residuals());
  EXPECT_EQ(difference_cost_function->parameter_block_sizes(),
            conditioned_cost_function.parameter_block_sizes());

  double *parameters[1];
  parameters[0] = v1;
  double *jacs[1];
  jacs[0] = jac;

  conditioned_cost_function.Evaluate(parameters, result, jacs);
  for (int i = 0; i < kTestCostFunctionSize; i++) {
    EXPECT_DOUBLE_EQ((i + 2) * (v1[i] - v2[i]) + i * 7, result[i]);
  }

  for (int i = 0; i < kTestCostFunctionSize; i++) {
    for (int j = 0; j < kTestCostFunctionSize; j++) {
      double actual = jac[i * kTestCostFunctionSize + j];
      if (i != j) {
        EXPECT_DOUBLE_EQ(0, actual);
      } else {
        EXPECT_DOUBLE_EQ(i + 2, actual);
      }
    }
  }
}

}  // namespace internal
}  // namespace ceres
