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

#include "ceres/autodiff_cost_function.h"

#include <cstddef>

#include "gtest/gtest.h"
#include "ceres/cost_function.h"

namespace ceres {
namespace internal {

class BinaryScalarCost {
 public:
  explicit BinaryScalarCost(double a): a_(a) {}
  template <typename T>
  bool operator()(const T* const x, const T* const y,
                  T* cost) const {
    cost[0] = x[0] * y[0] + x[1] * y[1]  - T(a_);
    return true;
  }
 private:
  double a_;
};

TEST(AutodiffCostFunction, BilinearDifferentiationTest) {
  CostFunction* cost_function  =
    new AutoDiffCostFunction<BinaryScalarCost, 1, 2, 2>(
        new BinaryScalarCost(1.0));

  double** parameters = new double*[2];
  parameters[0] = new double[2];
  parameters[1] = new double[2];

  parameters[0][0] = 1;
  parameters[0][1] = 2;

  parameters[1][0] = 3;
  parameters[1][1] = 4;

  double** jacobians = new double*[2];
  jacobians[0] = new double[2];
  jacobians[1] = new double[2];

  double residuals = 0.0;

  cost_function->Evaluate(parameters, &residuals, NULL);
  EXPECT_EQ(10.0, residuals);
  cost_function->Evaluate(parameters, &residuals, jacobians);

  EXPECT_EQ(3, jacobians[0][0]);
  EXPECT_EQ(4, jacobians[0][1]);
  EXPECT_EQ(1, jacobians[1][0]);
  EXPECT_EQ(2, jacobians[1][1]);

  delete[] jacobians[0];
  delete[] jacobians[1];
  delete[] parameters[0];
  delete[] parameters[1];
  delete[] jacobians;
  delete[] parameters;
  delete cost_function;
}

struct TenParameterCost {
  template <typename T>
  bool operator()(const T* const x0,
                  const T* const x1,
                  const T* const x2,
                  const T* const x3,
                  const T* const x4,
                  const T* const x5,
                  const T* const x6,
                  const T* const x7,
                  const T* const x8,
                  const T* const x9,
                  T* cost) const {
    cost[0] = *x0 + *x1 + *x2 + *x3 + *x4 + *x5 + *x6 + *x7 + *x8 + *x9;
    return true;
  }
};

TEST(AutodiffCostFunction, ManyParameterAutodiffInstantiates) {
  CostFunction* cost_function  =
      new AutoDiffCostFunction<
          TenParameterCost, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>(
              new TenParameterCost);

  double** parameters = new double*[10];
  double** jacobians = new double*[10];
  for (int i = 0; i < 10; ++i) {
    parameters[i] = new double[1];
    parameters[i][0] = i;
    jacobians[i] = new double[1];
  }

  double residuals = 0.0;

  cost_function->Evaluate(parameters, &residuals, NULL);
  EXPECT_EQ(45.0, residuals);

  cost_function->Evaluate(parameters, &residuals, jacobians);
  EXPECT_EQ(residuals, 45.0);
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(1.0, jacobians[i][0]);
  }

  for (int i = 0; i < 10; ++i) {
    delete[] jacobians[i];
    delete[] parameters[i];
  }
  delete[] jacobians;
  delete[] parameters;
  delete cost_function;
}

}  // namespace internal
}  // namespace ceres
