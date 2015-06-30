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

#include <cmath>
#include <limits>
#include "gtest/gtest.h"
#include "ceres/parameter_block.h"
#include "ceres/residual_block.h"
#include "ceres/residual_block_utils.h"
#include "ceres/cost_function.h"
#include "ceres/internal/scoped_ptr.h"
#include "ceres/sized_cost_function.h"

namespace ceres {
namespace internal {

// Routine to check if ResidualBlock::Evaluate for unary CostFunction
// with one residual succeeds with true or dies.
void CheckEvaluation(const CostFunction& cost_function, bool is_good) {
  double x = 1.0;
  ParameterBlock parameter_block(&x, 1, -1);
  vector<ParameterBlock*> parameter_blocks;
  parameter_blocks.push_back(&parameter_block);

  ResidualBlock residual_block(&cost_function,
                               NULL,
                               parameter_blocks,
                               -1);

  scoped_array<double> scratch(
      new double[residual_block.NumScratchDoublesForEvaluate()]);

  double cost;
  double residuals;
  double jacobian;
  double* jacobians[] = { &jacobian };

  EXPECT_EQ(residual_block.Evaluate(true,
                                    &cost,
                                    &residuals,
                                    jacobians,
                                    scratch.get()), is_good);
}

// A CostFunction that behaves normaly, i.e., it computes numerically
// valid residuals and jacobians.
class GoodCostFunction: public SizedCostFunction<1, 1> {
 public:
  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    residuals[0] = 1;
    if (jacobians != NULL && jacobians[0] != NULL) {
      jacobians[0][0] = 0.0;
    }
    return true;
  }
};

// The following four CostFunctions simulate the different ways in
// which user code can cause ResidualBlock::Evaluate to fail.
class NoResidualUpdateCostFunction: public SizedCostFunction<1, 1> {
 public:
  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    // Forget to update the residuals.
    // residuals[0] = 1;
    if (jacobians != NULL && jacobians[0] != NULL) {
      jacobians[0][0] = 0.0;
    }
    return true;
  }
};

class NoJacobianUpdateCostFunction: public SizedCostFunction<1, 1> {
 public:
  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    residuals[0] = 1;
    if (jacobians != NULL && jacobians[0] != NULL) {
      // Forget to update the jacobians.
      // jacobians[0][0] = 0.0;
    }
    return true;
  }
};

class BadResidualCostFunction: public SizedCostFunction<1, 1> {
 public:
  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    residuals[0] = std::numeric_limits<double>::infinity();
    if (jacobians != NULL && jacobians[0] != NULL) {
      jacobians[0][0] = 0.0;
    }
    return true;
  }
};

class BadJacobianCostFunction: public SizedCostFunction<1, 1> {
 public:
  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    residuals[0] = 1.0;
    if (jacobians != NULL && jacobians[0] != NULL) {
      jacobians[0][0] = std::numeric_limits<double>::quiet_NaN();
    }
    return true;
  }
};

// Note: It is preferable to write the below test as:
//
//  CheckEvaluation(GoodCostFunction(), true);
//  CheckEvaluation(NoResidualUpdateCostFunction(), false);
//  CheckEvaluation(NoJacobianUpdateCostFunction(), false);
//  ...
//
// however, there is a bug in the version of GCC on Mac OS X we tested, which
// requires the objects get put into local variables instead of getting
// instantiated on the stack.
TEST(ResidualBlockUtils, CheckAllCombinationsOfBadness) {
  GoodCostFunction good_fun;
  CheckEvaluation(good_fun, true);
  NoResidualUpdateCostFunction no_residual;
  CheckEvaluation(no_residual, false);
  NoJacobianUpdateCostFunction no_jacobian;
  CheckEvaluation(no_jacobian, false);
  BadResidualCostFunction bad_residual;
  CheckEvaluation(bad_residual, false);
  BadJacobianCostFunction bad_jacobian;
  CheckEvaluation(bad_jacobian, false);
}

}  // namespace internal
}  // namespace ceres
