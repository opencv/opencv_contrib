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

#include "ceres/gradient_checking_cost_function.h"

#include <cmath>
#include <vector>
#include "ceres/cost_function.h"
#include "ceres/internal/scoped_ptr.h"
#include "ceres/local_parameterization.h"
#include "ceres/loss_function.h"
#include "ceres/parameter_block.h"
#include "ceres/problem_impl.h"
#include "ceres/program.h"
#include "ceres/random.h"
#include "ceres/residual_block.h"
#include "ceres/sized_cost_function.h"
#include "ceres/types.h"
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

// Pick a (non-quadratic) function whose derivative are easy:
//
//    f = exp(- a' x).
//   df = - f a.
//
// where 'a' is a vector of the same size as 'x'. In the block
// version, they are both block vectors, of course.
template<int bad_block = 1, int bad_variable = 2>
class TestTerm : public CostFunction {
 public:
  // The constructor of this function needs to know the number
  // of blocks desired, and the size of each block.
  TestTerm(int arity, int const *dim) : arity_(arity) {
    // Make 'arity' random vectors.
    a_.resize(arity_);
    for (int j = 0; j < arity_; ++j) {
      a_[j].resize(dim[j]);
      for (int u = 0; u < dim[j]; ++u) {
        a_[j][u] = 2.0 * RandDouble() - 1.0;
      }
    }

    for (int i = 0; i < arity_; i++) {
      mutable_parameter_block_sizes()->push_back(dim[i]);
    }
    set_num_residuals(1);
  }

  bool Evaluate(double const* const* parameters,
                double* residuals,
                double** jacobians) const {
    // Compute a . x.
    double ax = 0;
    for (int j = 0; j < arity_; ++j) {
      for (int u = 0; u < parameter_block_sizes()[j]; ++u) {
        ax += a_[j][u] * parameters[j][u];
      }
    }

    // This is the cost, but also appears as a factor
    // in the derivatives.
    double f = *residuals = exp(-ax);

    // Accumulate 1st order derivatives.
    if (jacobians) {
      for (int j = 0; j < arity_; ++j) {
        if (jacobians[j]) {
          for (int u = 0; u < parameter_block_sizes()[j]; ++u) {
            // See comments before class.
            jacobians[j][u] = - f * a_[j][u];

            if (bad_block == j && bad_variable == u) {
              // Whoopsiedoopsie! Deliberately introduce a faulty jacobian entry
              // like what happens when users make an error in their jacobian
              // computations. This should get detected.
              LOG(INFO) << "Poisoning jacobian for parameter block " << j
                        << ", row 0, column " << u;
              jacobians[j][u] += 500;
            }
          }
        }
      }
    }

    return true;
  }

 private:
  int arity_;
  vector<vector<double> > a_;
};

TEST(GradientCheckingCostFunction, ResidualsAndJacobiansArePreservedTest) {
  srand(5);

  // Test with 3 blocks of size 2, 3 and 4.
  int const arity = 3;
  int const dim[arity] = { 2, 3, 4 };

  // Make a random set of blocks.
  vector<double*> parameters(arity);
  for (int j = 0; j < arity; ++j) {
    parameters[j] = new double[dim[j]];
    for (int u = 0; u < dim[j]; ++u) {
      parameters[j][u] = 2.0 * RandDouble() - 1.0;
    }
  }

  double original_residual;
  double residual;
  vector<double*> original_jacobians(arity);
  vector<double*> jacobians(arity);

  for (int j = 0; j < arity; ++j) {
    // Since residual is one dimensional the jacobians have the same
    // size as the parameter blocks.
    jacobians[j] = new double[dim[j]];
    original_jacobians[j] = new double[dim[j]];
  }

  const double kRelativeStepSize = 1e-6;
  const double kRelativePrecision = 1e-4;

  TestTerm<-1, -1> term(arity, dim);
  scoped_ptr<CostFunction> gradient_checking_cost_function(
      CreateGradientCheckingCostFunction(&term,
                                         kRelativeStepSize,
                                         kRelativePrecision,
                                         "Ignored."));
  term.Evaluate(&parameters[0],
                &original_residual,
                &original_jacobians[0]);

  gradient_checking_cost_function->Evaluate(&parameters[0],
                                            &residual,
                                            &jacobians[0]);
  EXPECT_EQ(original_residual, residual);

  for (int j = 0; j < arity; j++) {
    for (int k = 0; k < dim[j]; ++k) {
      EXPECT_EQ(original_jacobians[j][k], jacobians[j][k]);
    }

    delete[] parameters[j];
    delete[] jacobians[j];
    delete[] original_jacobians[j];
  }
}

TEST(GradientCheckingCostFunction, SmokeTest) {
  srand(5);

  // Test with 3 blocks of size 2, 3 and 4.
  int const arity = 3;
  int const dim[arity] = { 2, 3, 4 };

  // Make a random set of blocks.
  vector<double*> parameters(arity);
  for (int j = 0; j < arity; ++j) {
    parameters[j] = new double[dim[j]];
    for (int u = 0; u < dim[j]; ++u) {
      parameters[j][u] = 2.0 * RandDouble() - 1.0;
    }
  }

  double residual;
  vector<double*> jacobians(arity);
  for (int j = 0; j < arity; ++j) {
    // Since residual is one dimensional the jacobians have the same size as the
    // parameter blocks.
    jacobians[j] = new double[dim[j]];
  }

  const double kRelativeStepSize = 1e-6;
  const double kRelativePrecision = 1e-4;

  // Should have one term that's bad, causing everything to get dumped.
  LOG(INFO) << "Bad gradient";
  {
    TestTerm<1, 2> term(arity, dim);
    scoped_ptr<CostFunction> gradient_checking_cost_function(
        CreateGradientCheckingCostFunction(&term,
                                           kRelativeStepSize,
                                           kRelativePrecision,
                                           "Fuzzy bananas"));

    ScopedMockLog log;
    EXPECT_CALL(log, Log(_, _, _)).Times(AnyNumber());
    EXPECT_CALL(log, Log(WARNING, _,
                         AllOf(HasSubstr("(1,0,2) Relative error worse than"),
                               HasSubstr("Fuzzy bananas"))));

    gradient_checking_cost_function->Evaluate(&parameters[0],
                                              &residual,
                                              &jacobians[0]);
  }

  // The gradient is correct, so no errors are reported.
  LOG(INFO) << "Good gradient";
  {
    TestTerm<-1, -1> term(arity, dim);
    scoped_ptr<CostFunction> gradient_checking_cost_function(
        CreateGradientCheckingCostFunction(&term,
                                           kRelativeStepSize,
                                           kRelativePrecision,
                                           "Ignored."));

    ScopedMockLog log;
    EXPECT_CALL(log, Log(_, _, _)).Times(0);

    gradient_checking_cost_function->Evaluate(&parameters[0],
                                              &residual,
                                              &jacobians[0]);
  }

  for (int j = 0; j < arity; j++) {
    delete[] parameters[j];
    delete[] jacobians[j];
  }
}

// The following three classes are for the purposes of defining
// function signatures. They have dummy Evaluate functions.

// Trivial cost function that accepts a single argument.
class UnaryCostFunction : public CostFunction {
 public:
  UnaryCostFunction(int num_residuals, int16 parameter_block_size) {
    set_num_residuals(num_residuals);
    mutable_parameter_block_sizes()->push_back(parameter_block_size);
  }
  virtual ~UnaryCostFunction() {}

  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    for (int i = 0; i < num_residuals(); ++i) {
      residuals[i] = 1;
    }
    return true;
  }
};

// Trivial cost function that accepts two arguments.
class BinaryCostFunction: public CostFunction {
 public:
  BinaryCostFunction(int num_residuals,
                     int16 parameter_block1_size,
                     int16 parameter_block2_size) {
    set_num_residuals(num_residuals);
    mutable_parameter_block_sizes()->push_back(parameter_block1_size);
    mutable_parameter_block_sizes()->push_back(parameter_block2_size);
  }

  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    for (int i = 0; i < num_residuals(); ++i) {
      residuals[i] = 2;
    }
    return true;
  }
};

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
      residuals[i] = 3;
    }
    return true;
  }
};

// Verify that the two ParameterBlocks are formed from the same user
// array and have the same LocalParameterization object.
void ParameterBlocksAreEquivalent(const ParameterBlock*  left,
                                  const ParameterBlock* right) {
  CHECK_NOTNULL(left);
  CHECK_NOTNULL(right);
  EXPECT_EQ(left->user_state(), right->user_state());
  EXPECT_EQ(left->Size(), right->Size());
  EXPECT_EQ(left->Size(), right->Size());
  EXPECT_EQ(left->LocalSize(), right->LocalSize());
  EXPECT_EQ(left->local_parameterization(), right->local_parameterization());
  EXPECT_EQ(left->IsConstant(), right->IsConstant());
}

TEST(GradientCheckingProblemImpl, ProblemDimensionsMatch) {
  // Parameter blocks with arbitrarily chosen initial values.
  double x[] = {1.0, 2.0, 3.0};
  double y[] = {4.0, 5.0, 6.0, 7.0};
  double z[] = {8.0, 9.0, 10.0, 11.0, 12.0};
  double w[] = {13.0, 14.0, 15.0, 16.0};

  ProblemImpl problem_impl;
  problem_impl.AddParameterBlock(x, 3);
  problem_impl.AddParameterBlock(y, 4);
  problem_impl.SetParameterBlockConstant(y);
  problem_impl.AddParameterBlock(z, 5);
  problem_impl.AddParameterBlock(w, 4, new QuaternionParameterization);
  problem_impl.AddResidualBlock(new UnaryCostFunction(2, 3), NULL, x);
  problem_impl.AddResidualBlock(new BinaryCostFunction(6, 5, 4) ,
                                NULL, z, y);
  problem_impl.AddResidualBlock(new BinaryCostFunction(3, 3, 5),
                                new TrivialLoss, x, z);
  problem_impl.AddResidualBlock(new BinaryCostFunction(7, 5, 3),
                                NULL, z, x);
  problem_impl.AddResidualBlock(new TernaryCostFunction(1, 5, 3, 4),
                                NULL, z, x, y);

  scoped_ptr<ProblemImpl> gradient_checking_problem_impl(
      CreateGradientCheckingProblemImpl(&problem_impl, 1.0, 1.0));

  // The dimensions of the two problems match.
  EXPECT_EQ(problem_impl.NumParameterBlocks(),
            gradient_checking_problem_impl->NumParameterBlocks());
  EXPECT_EQ(problem_impl.NumResidualBlocks(),
            gradient_checking_problem_impl->NumResidualBlocks());

  EXPECT_EQ(problem_impl.NumParameters(),
            gradient_checking_problem_impl->NumParameters());
  EXPECT_EQ(problem_impl.NumResiduals(),
            gradient_checking_problem_impl->NumResiduals());

  const Program& program = problem_impl.program();
  const Program& gradient_checking_program =
      gradient_checking_problem_impl->program();

  // Since we added the ParameterBlocks and ResidualBlocks explicitly,
  // they should be in the same order in the two programs. It is
  // possible that may change due to implementation changes to
  // Program. This is not exepected to be the case and writing code to
  // anticipate that possibility not worth the extra complexity in
  // this test.
  for (int i = 0; i < program.parameter_blocks().size(); ++i) {
    ParameterBlocksAreEquivalent(
        program.parameter_blocks()[i],
        gradient_checking_program.parameter_blocks()[i]);
  }

  for (int i = 0; i < program.residual_blocks().size(); ++i) {
    // Compare the sizes of the two ResidualBlocks.
    const ResidualBlock* original_residual_block =
        program.residual_blocks()[i];
    const ResidualBlock* new_residual_block =
        gradient_checking_program.residual_blocks()[i];
    EXPECT_EQ(original_residual_block->NumParameterBlocks(),
              new_residual_block->NumParameterBlocks());
    EXPECT_EQ(original_residual_block->NumResiduals(),
              new_residual_block->NumResiduals());
    EXPECT_EQ(original_residual_block->NumScratchDoublesForEvaluate(),
              new_residual_block->NumScratchDoublesForEvaluate());

    // Verify that the ParameterBlocks for the two residuals are equivalent.
    for (int j = 0; j < original_residual_block->NumParameterBlocks(); ++j) {
      ParameterBlocksAreEquivalent(
          original_residual_block->parameter_blocks()[j],
          new_residual_block->parameter_blocks()[j]);
    }
  }
}

}  // namespace internal
}  // namespace ceres
