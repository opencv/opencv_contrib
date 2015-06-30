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

#include "ceres/residual_block.h"

#include "gtest/gtest.h"
#include "ceres/parameter_block.h"
#include "ceres/sized_cost_function.h"
#include "ceres/internal/eigen.h"
#include "ceres/local_parameterization.h"

namespace ceres {
namespace internal {

// Trivial cost function that accepts three arguments.
class TernaryCostFunction: public CostFunction {
 public:
  TernaryCostFunction(int num_residuals,
                      int16 parameter_block1_size,
                      int16 parameter_block2_size,
                      int16 parameter_block3_size) {
    set_num_residuals(num_residuals);
    mutable_parameter_block_sizes()->push_back(parameter_block1_size);
    mutable_parameter_block_sizes()->push_back(parameter_block2_size);
    mutable_parameter_block_sizes()->push_back(parameter_block3_size);
  }

  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    for (int i = 0; i < num_residuals(); ++i) {
      residuals[i] = i;
    }
    if (jacobians) {
      for (int k = 0; k < 3; ++k) {
        if (jacobians[k] != NULL) {
          MatrixRef jacobian(jacobians[k],
                             num_residuals(),
                             parameter_block_sizes()[k]);
          jacobian.setConstant(k);
        }
      }
    }
    return true;
  }
};

TEST(ResidualBlock, EvaluteWithNoLossFunctionOrLocalParameterizations) {
  double scratch[64];

  // Prepare the parameter blocks.
  double values_x[2];
  ParameterBlock x(values_x, 2, -1);

  double values_y[3];
  ParameterBlock y(values_y, 3, -1);

  double values_z[4];
  ParameterBlock z(values_z, 4, -1);

  vector<ParameterBlock*> parameters;
  parameters.push_back(&x);
  parameters.push_back(&y);
  parameters.push_back(&z);

  TernaryCostFunction cost_function(3, 2, 3, 4);

  // Create the object under tests.
  ResidualBlock residual_block(&cost_function, NULL, parameters, -1);

  // Verify getters.
  EXPECT_EQ(&cost_function, residual_block.cost_function());
  EXPECT_EQ(NULL, residual_block.loss_function());
  EXPECT_EQ(parameters[0], residual_block.parameter_blocks()[0]);
  EXPECT_EQ(parameters[1], residual_block.parameter_blocks()[1]);
  EXPECT_EQ(parameters[2], residual_block.parameter_blocks()[2]);
  EXPECT_EQ(3, residual_block.NumScratchDoublesForEvaluate());

  // Verify cost-only evaluation.
  double cost;
  residual_block.Evaluate(true, &cost, NULL, NULL, scratch);
  EXPECT_EQ(0.5 * (0*0 + 1*1 + 2*2), cost);

  // Verify cost and residual evaluation.
  double residuals[3];
  residual_block.Evaluate(true, &cost, residuals, NULL, scratch);
  EXPECT_EQ(0.5 * (0*0 + 1*1 + 2*2), cost);
  EXPECT_EQ(0.0, residuals[0]);
  EXPECT_EQ(1.0, residuals[1]);
  EXPECT_EQ(2.0, residuals[2]);

  // Verify cost, residual, and jacobian evaluation.
  cost = 0.0;
  VectorRef(residuals, 3).setConstant(0.0);

  Matrix jacobian_rx(3, 2);
  Matrix jacobian_ry(3, 3);
  Matrix jacobian_rz(3, 4);

  jacobian_rx.setConstant(-1.0);
  jacobian_ry.setConstant(-1.0);
  jacobian_rz.setConstant(-1.0);

  double *jacobian_ptrs[3] = {
    jacobian_rx.data(),
    jacobian_ry.data(),
    jacobian_rz.data()
  };

  residual_block.Evaluate(true, &cost, residuals, jacobian_ptrs, scratch);
  EXPECT_EQ(0.5 * (0*0 + 1*1 + 2*2), cost);
  EXPECT_EQ(0.0, residuals[0]);
  EXPECT_EQ(1.0, residuals[1]);
  EXPECT_EQ(2.0, residuals[2]);

  EXPECT_TRUE((jacobian_rx.array() == 0.0).all()) << "\n" << jacobian_rx;
  EXPECT_TRUE((jacobian_ry.array() == 1.0).all()) << "\n" << jacobian_ry;
  EXPECT_TRUE((jacobian_rz.array() == 2.0).all()) << "\n" << jacobian_rz;

  // Verify cost, residual, and partial jacobian evaluation.
  cost = 0.0;
  VectorRef(residuals, 3).setConstant(0.0);
  jacobian_rx.setConstant(-1.0);
  jacobian_ry.setConstant(-1.0);
  jacobian_rz.setConstant(-1.0);

  jacobian_ptrs[1] = NULL;  // Don't compute the jacobian for y.

  residual_block.Evaluate(true, &cost, residuals, jacobian_ptrs, scratch);
  EXPECT_EQ(0.5 * (0*0 + 1*1 + 2*2), cost);
  EXPECT_EQ(0.0, residuals[0]);
  EXPECT_EQ(1.0, residuals[1]);
  EXPECT_EQ(2.0, residuals[2]);

  EXPECT_TRUE((jacobian_rx.array() ==  0.0).all()) << "\n" << jacobian_rx;
  EXPECT_TRUE((jacobian_ry.array() == -1.0).all()) << "\n" << jacobian_ry;
  EXPECT_TRUE((jacobian_rz.array() ==  2.0).all()) << "\n" << jacobian_rz;
}

// Trivial cost function that accepts three arguments.
class LocallyParameterizedCostFunction: public SizedCostFunction<3, 2, 3, 4> {
 public:
  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    for (int i = 0; i < num_residuals(); ++i) {
      residuals[i] = i;
    }
    if (jacobians) {
      for (int k = 0; k < 3; ++k) {
        // The jacobians here are full sized, but they are transformed in the
        // evaluator into the "local" jacobian. In the tests, the "subset
        // constant" parameterization is used, which should pick out columns
        // from these jacobians. Put values in the jacobian that make this
        // obvious; in particular, make the jacobians like this:
        //
        //   0 1 2 3 4 ...
        //   0 1 2 3 4 ...
        //   0 1 2 3 4 ...
        //
        if (jacobians[k] != NULL) {
          MatrixRef jacobian(jacobians[k],
                             num_residuals(),
                             parameter_block_sizes()[k]);
          for (int j = 0; j < k + 2; ++j) {
            jacobian.col(j).setConstant(j);
          }
        }
      }
    }
    return true;
  }
};

TEST(ResidualBlock, EvaluteWithLocalParameterizations) {
  double scratch[64];

  // Prepare the parameter blocks.
  double values_x[2];
  ParameterBlock x(values_x, 2, -1);

  double values_y[3];
  ParameterBlock y(values_y, 3, -1);

  double values_z[4];
  ParameterBlock z(values_z, 4, -1);

  vector<ParameterBlock*> parameters;
  parameters.push_back(&x);
  parameters.push_back(&y);
  parameters.push_back(&z);

  // Make x have the first component fixed.
  vector<int> x_fixed;
  x_fixed.push_back(0);
  SubsetParameterization x_parameterization(2, x_fixed);
  x.SetParameterization(&x_parameterization);

  // Make z have the last and last component fixed.
  vector<int> z_fixed;
  z_fixed.push_back(2);
  SubsetParameterization z_parameterization(4, z_fixed);
  z.SetParameterization(&z_parameterization);

  LocallyParameterizedCostFunction cost_function;

  // Create the object under tests.
  ResidualBlock residual_block(&cost_function, NULL, parameters, -1);

  // Verify getters.
  EXPECT_EQ(&cost_function, residual_block.cost_function());
  EXPECT_EQ(NULL, residual_block.loss_function());
  EXPECT_EQ(parameters[0], residual_block.parameter_blocks()[0]);
  EXPECT_EQ(parameters[1], residual_block.parameter_blocks()[1]);
  EXPECT_EQ(parameters[2], residual_block.parameter_blocks()[2]);
  EXPECT_EQ(3*(2 + 4) + 3, residual_block.NumScratchDoublesForEvaluate());

  // Verify cost-only evaluation.
  double cost;
  residual_block.Evaluate(true, &cost, NULL, NULL, scratch);
  EXPECT_EQ(0.5 * (0*0 + 1*1 + 2*2), cost);

  // Verify cost and residual evaluation.
  double residuals[3];
  residual_block.Evaluate(true, &cost, residuals, NULL, scratch);
  EXPECT_EQ(0.5 * (0*0 + 1*1 + 2*2), cost);
  EXPECT_EQ(0.0, residuals[0]);
  EXPECT_EQ(1.0, residuals[1]);
  EXPECT_EQ(2.0, residuals[2]);

  // Verify cost, residual, and jacobian evaluation.
  cost = 0.0;
  VectorRef(residuals, 3).setConstant(0.0);

  Matrix jacobian_rx(3, 1);  // Since the first element is fixed.
  Matrix jacobian_ry(3, 3);
  Matrix jacobian_rz(3, 3);  // Since the third element is fixed.

  jacobian_rx.setConstant(-1.0);
  jacobian_ry.setConstant(-1.0);
  jacobian_rz.setConstant(-1.0);

  double *jacobian_ptrs[3] = {
    jacobian_rx.data(),
    jacobian_ry.data(),
    jacobian_rz.data()
  };

  residual_block.Evaluate(true, &cost, residuals, jacobian_ptrs, scratch);
  EXPECT_EQ(0.5 * (0*0 + 1*1 + 2*2), cost);
  EXPECT_EQ(0.0, residuals[0]);
  EXPECT_EQ(1.0, residuals[1]);
  EXPECT_EQ(2.0, residuals[2]);

  Matrix expected_jacobian_rx(3, 1);
  expected_jacobian_rx << 1.0, 1.0, 1.0;

  Matrix expected_jacobian_ry(3, 3);
  expected_jacobian_ry << 0.0, 1.0, 2.0,
                          0.0, 1.0, 2.0,
                          0.0, 1.0, 2.0;

  Matrix expected_jacobian_rz(3, 3);
  expected_jacobian_rz << 0.0, 1.0, /* 2.0, */ 3.0,  // 3rd parameter constant.
                          0.0, 1.0, /* 2.0, */ 3.0,
                          0.0, 1.0, /* 2.0, */ 3.0;

  EXPECT_EQ(expected_jacobian_rx, jacobian_rx)
      << "\nExpected:\n" << expected_jacobian_rx
      << "\nActual:\n"   << jacobian_rx;
  EXPECT_EQ(expected_jacobian_ry, jacobian_ry)
      << "\nExpected:\n" << expected_jacobian_ry
      << "\nActual:\n"   << jacobian_ry;
  EXPECT_EQ(expected_jacobian_rz, jacobian_rz)
      << "\nExpected:\n " << expected_jacobian_rz
      << "\nActual:\n"   << jacobian_rz;

  // Verify cost, residual, and partial jacobian evaluation.
  cost = 0.0;
  VectorRef(residuals, 3).setConstant(0.0);
  jacobian_rx.setConstant(-1.0);
  jacobian_ry.setConstant(-1.0);
  jacobian_rz.setConstant(-1.0);

  jacobian_ptrs[1] = NULL;  // Don't compute the jacobian for y.

  residual_block.Evaluate(true, &cost, residuals, jacobian_ptrs, scratch);
  EXPECT_EQ(0.5 * (0*0 + 1*1 + 2*2), cost);
  EXPECT_EQ(0.0, residuals[0]);
  EXPECT_EQ(1.0, residuals[1]);
  EXPECT_EQ(2.0, residuals[2]);

  EXPECT_EQ(expected_jacobian_rx, jacobian_rx);
  EXPECT_TRUE((jacobian_ry.array() == -1.0).all()) << "\n" << jacobian_ry;
  EXPECT_EQ(expected_jacobian_rz, jacobian_rz);
}

}  // namespace internal
}  // namespace ceres
