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
//         keir@google.com (Keir Mierle)

#include "ceres/problem.h"
#include "ceres/problem_impl.h"

#include "ceres/casts.h"
#include "ceres/cost_function.h"
#include "ceres/crs_matrix.h"
#include "ceres/evaluator_test_utils.cc"
#include "ceres/internal/eigen.h"
#include "ceres/internal/scoped_ptr.h"
#include "ceres/local_parameterization.h"
#include "ceres/map_util.h"
#include "ceres/parameter_block.h"
#include "ceres/program.h"
#include "ceres/sized_cost_function.h"
#include "ceres/sparse_matrix.h"
#include "ceres/types.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

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

TEST(Problem, AddResidualWithNullCostFunctionDies) {
  double x[3], y[4], z[5];

  Problem problem;
  problem.AddParameterBlock(x, 3);
  problem.AddParameterBlock(y, 4);
  problem.AddParameterBlock(z, 5);

  EXPECT_DEATH_IF_SUPPORTED(problem.AddResidualBlock(NULL, NULL, x),
                            "'cost_function' Must be non NULL");
}

TEST(Problem, AddResidualWithIncorrectNumberOfParameterBlocksDies) {
  double x[3], y[4], z[5];

  Problem problem;
  problem.AddParameterBlock(x, 3);
  problem.AddParameterBlock(y, 4);
  problem.AddParameterBlock(z, 5);

  // UnaryCostFunction takes only one parameter, but two are passed.
  EXPECT_DEATH_IF_SUPPORTED(
      problem.AddResidualBlock(new UnaryCostFunction(2, 3), NULL, x, y),
      "parameter_blocks.size()");
}

TEST(Problem, AddResidualWithDifferentSizesOnTheSameVariableDies) {
  double x[3];

  Problem problem;
  problem.AddResidualBlock(new UnaryCostFunction(2, 3), NULL, x);
  EXPECT_DEATH_IF_SUPPORTED(problem.AddResidualBlock(
                                new UnaryCostFunction(
                                    2, 4 /* 4 != 3 */), NULL, x),
                            "different block sizes");
}

TEST(Problem, AddResidualWithDuplicateParametersDies) {
  double x[3], z[5];

  Problem problem;
  EXPECT_DEATH_IF_SUPPORTED(problem.AddResidualBlock(
                                new BinaryCostFunction(2, 3, 3), NULL, x, x),
                            "Duplicate parameter blocks");
  EXPECT_DEATH_IF_SUPPORTED(problem.AddResidualBlock(
                                new TernaryCostFunction(1, 5, 3, 5),
                                NULL, z, x, z),
                            "Duplicate parameter blocks");
}

TEST(Problem, AddResidualWithIncorrectSizesOfParameterBlockDies) {
  double x[3], y[4], z[5];

  Problem problem;
  problem.AddParameterBlock(x, 3);
  problem.AddParameterBlock(y, 4);
  problem.AddParameterBlock(z, 5);

  // The cost function expects the size of the second parameter, z, to be 4
  // instead of 5 as declared above. This is fatal.
  EXPECT_DEATH_IF_SUPPORTED(problem.AddResidualBlock(
      new BinaryCostFunction(2, 3, 4), NULL, x, z),
               "different block sizes");
}

TEST(Problem, AddResidualAddsDuplicatedParametersOnlyOnce) {
  double x[3], y[4], z[5];

  Problem problem;
  problem.AddResidualBlock(new UnaryCostFunction(2, 3), NULL, x);
  problem.AddResidualBlock(new UnaryCostFunction(2, 3), NULL, x);
  problem.AddResidualBlock(new UnaryCostFunction(2, 4), NULL, y);
  problem.AddResidualBlock(new UnaryCostFunction(2, 5), NULL, z);

  EXPECT_EQ(3, problem.NumParameterBlocks());
  EXPECT_EQ(12, problem.NumParameters());
}

TEST(Problem, AddParameterWithDifferentSizesOnTheSameVariableDies) {
  double x[3], y[4];

  Problem problem;
  problem.AddParameterBlock(x, 3);
  problem.AddParameterBlock(y, 4);

  EXPECT_DEATH_IF_SUPPORTED(problem.AddParameterBlock(x, 4),
                            "different block sizes");
}

static double *IntToPtr(int i) {
  return reinterpret_cast<double*>(sizeof(double) * i);  // NOLINT
}

TEST(Problem, AddParameterWithAliasedParametersDies) {
  // Layout is
  //
  //   0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17
  //                 [x] x  x  x  x          [y] y  y
  //         o==o==o                 o==o==o           o==o
  //               o--o--o     o--o--o     o--o  o--o--o
  //
  // Parameter block additions are tested as listed above; expected successful
  // ones marked with o==o and aliasing ones marked with o--o.

  Problem problem;
  problem.AddParameterBlock(IntToPtr(5),  5);  // x
  problem.AddParameterBlock(IntToPtr(13), 3);  // y

  EXPECT_DEATH_IF_SUPPORTED(problem.AddParameterBlock(IntToPtr( 4), 2),
                            "Aliasing detected");
  EXPECT_DEATH_IF_SUPPORTED(problem.AddParameterBlock(IntToPtr( 4), 3),
                            "Aliasing detected");
  EXPECT_DEATH_IF_SUPPORTED(problem.AddParameterBlock(IntToPtr( 4), 9),
                            "Aliasing detected");
  EXPECT_DEATH_IF_SUPPORTED(problem.AddParameterBlock(IntToPtr( 8), 3),
                            "Aliasing detected");
  EXPECT_DEATH_IF_SUPPORTED(problem.AddParameterBlock(IntToPtr(12), 2),
                            "Aliasing detected");
  EXPECT_DEATH_IF_SUPPORTED(problem.AddParameterBlock(IntToPtr(14), 3),
                            "Aliasing detected");

  // These ones should work.
  problem.AddParameterBlock(IntToPtr( 2), 3);
  problem.AddParameterBlock(IntToPtr(10), 3);
  problem.AddParameterBlock(IntToPtr(16), 2);

  ASSERT_EQ(5, problem.NumParameterBlocks());
}

TEST(Problem, AddParameterIgnoresDuplicateCalls) {
  double x[3], y[4];

  Problem problem;
  problem.AddParameterBlock(x, 3);
  problem.AddParameterBlock(y, 4);

  // Creating parameter blocks multiple times is ignored.
  problem.AddParameterBlock(x, 3);
  problem.AddResidualBlock(new UnaryCostFunction(2, 3), NULL, x);

  // ... even repeatedly.
  problem.AddParameterBlock(x, 3);
  problem.AddResidualBlock(new UnaryCostFunction(2, 3), NULL, x);

  // More parameters are fine.
  problem.AddParameterBlock(y, 4);
  problem.AddResidualBlock(new UnaryCostFunction(2, 4), NULL, y);

  EXPECT_EQ(2, problem.NumParameterBlocks());
  EXPECT_EQ(7, problem.NumParameters());
}

TEST(Problem, AddingParametersAndResidualsResultsInExpectedProblem) {
  double x[3], y[4], z[5], w[4];

  Problem problem;
  problem.AddParameterBlock(x, 3);
  EXPECT_EQ(1, problem.NumParameterBlocks());
  EXPECT_EQ(3, problem.NumParameters());

  problem.AddParameterBlock(y, 4);
  EXPECT_EQ(2, problem.NumParameterBlocks());
  EXPECT_EQ(7, problem.NumParameters());

  problem.AddParameterBlock(z, 5);
  EXPECT_EQ(3,  problem.NumParameterBlocks());
  EXPECT_EQ(12, problem.NumParameters());

  // Add a parameter that has a local parameterization.
  w[0] = 1.0; w[1] = 0.0; w[2] = 0.0; w[3] = 0.0;
  problem.AddParameterBlock(w, 4, new QuaternionParameterization);
  EXPECT_EQ(4,  problem.NumParameterBlocks());
  EXPECT_EQ(16, problem.NumParameters());

  problem.AddResidualBlock(new UnaryCostFunction(2, 3), NULL, x);
  problem.AddResidualBlock(new BinaryCostFunction(6, 5, 4) , NULL, z, y);
  problem.AddResidualBlock(new BinaryCostFunction(3, 3, 5), NULL, x, z);
  problem.AddResidualBlock(new BinaryCostFunction(7, 5, 3), NULL, z, x);
  problem.AddResidualBlock(new TernaryCostFunction(1, 5, 3, 4), NULL, z, x, y);

  const int total_residuals = 2 + 6 + 3 + 7 + 1;
  EXPECT_EQ(problem.NumResidualBlocks(), 5);
  EXPECT_EQ(problem.NumResiduals(), total_residuals);
}

class DestructorCountingCostFunction : public SizedCostFunction<3, 4, 5> {
 public:
  explicit DestructorCountingCostFunction(int *num_destructions)
      : num_destructions_(num_destructions) {}

  virtual ~DestructorCountingCostFunction() {
    *num_destructions_ += 1;
  }

  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    return true;
  }

 private:
  int* num_destructions_;
};

TEST(Problem, ReusedCostFunctionsAreOnlyDeletedOnce) {
  double y[4], z[5];
  int num_destructions = 0;

  // Add a cost function multiple times and check to make sure that
  // the destructor on the cost function is only called once.
  {
    Problem problem;
    problem.AddParameterBlock(y, 4);
    problem.AddParameterBlock(z, 5);

    CostFunction* cost = new DestructorCountingCostFunction(&num_destructions);
    problem.AddResidualBlock(cost, NULL, y, z);
    problem.AddResidualBlock(cost, NULL, y, z);
    problem.AddResidualBlock(cost, NULL, y, z);
    EXPECT_EQ(3, problem.NumResidualBlocks());
  }

  // Check that the destructor was called only once.
  CHECK_EQ(num_destructions, 1);
}

TEST(Problem, CostFunctionsAreDeletedEvenWithRemovals) {
  double y[4], z[5], w[4];
  int num_destructions = 0;
  {
    Problem problem;
    problem.AddParameterBlock(y, 4);
    problem.AddParameterBlock(z, 5);

    CostFunction* cost_yz =
        new DestructorCountingCostFunction(&num_destructions);
    CostFunction* cost_wz =
        new DestructorCountingCostFunction(&num_destructions);
    ResidualBlock* r_yz = problem.AddResidualBlock(cost_yz, NULL, y, z);
    ResidualBlock* r_wz = problem.AddResidualBlock(cost_wz, NULL, w, z);
    EXPECT_EQ(2, problem.NumResidualBlocks());

    // In the current implementation, the destructor shouldn't get run yet.
    problem.RemoveResidualBlock(r_yz);
    CHECK_EQ(num_destructions, 0);
    problem.RemoveResidualBlock(r_wz);
    CHECK_EQ(num_destructions, 0);

    EXPECT_EQ(0, problem.NumResidualBlocks());
  }
  CHECK_EQ(num_destructions, 2);
}

// Make the dynamic problem tests (e.g. for removing residual blocks)
// parameterized on whether the low-latency mode is enabled or not.
//
// This tests against ProblemImpl instead of Problem in order to inspect the
// state of the resulting Program; this is difficult with only the thin Problem
// interface.
struct DynamicProblem : public ::testing::TestWithParam<bool> {
  DynamicProblem() {
    Problem::Options options;
    options.enable_fast_parameter_block_removal = GetParam();
    problem.reset(new ProblemImpl(options));
  }

  ParameterBlock* GetParameterBlock(int block) {
    return problem->program().parameter_blocks()[block];
  }
  ResidualBlock* GetResidualBlock(int block) {
    return problem->program().residual_blocks()[block];
  }

  bool HasResidualBlock(ResidualBlock* residual_block) {
    return find(problem->program().residual_blocks().begin(),
                problem->program().residual_blocks().end(),
                residual_block) != problem->program().residual_blocks().end();
  }

  // The next block of functions until the end are only for testing the
  // residual block removals.
  void ExpectParameterBlockContainsResidualBlock(
      double* values,
      ResidualBlock* residual_block) {
    ParameterBlock* parameter_block =
        FindOrDie(problem->parameter_map(), values);
    EXPECT_TRUE(ContainsKey(*(parameter_block->mutable_residual_blocks()),
                            residual_block));
  }

  void ExpectSize(double* values, int size) {
    ParameterBlock* parameter_block =
        FindOrDie(problem->parameter_map(), values);
    EXPECT_EQ(size, parameter_block->mutable_residual_blocks()->size());
  }

  // Degenerate case.
  void ExpectParameterBlockContains(double* values) {
    ExpectSize(values, 0);
  }

  void ExpectParameterBlockContains(double* values,
                                    ResidualBlock* r1) {
    ExpectSize(values, 1);
    ExpectParameterBlockContainsResidualBlock(values, r1);
  }

  void ExpectParameterBlockContains(double* values,
                                    ResidualBlock* r1,
                                    ResidualBlock* r2) {
    ExpectSize(values, 2);
    ExpectParameterBlockContainsResidualBlock(values, r1);
    ExpectParameterBlockContainsResidualBlock(values, r2);
  }

  void ExpectParameterBlockContains(double* values,
                                    ResidualBlock* r1,
                                    ResidualBlock* r2,
                                    ResidualBlock* r3) {
    ExpectSize(values, 3);
    ExpectParameterBlockContainsResidualBlock(values, r1);
    ExpectParameterBlockContainsResidualBlock(values, r2);
    ExpectParameterBlockContainsResidualBlock(values, r3);
  }

  void ExpectParameterBlockContains(double* values,
                                    ResidualBlock* r1,
                                    ResidualBlock* r2,
                                    ResidualBlock* r3,
                                    ResidualBlock* r4) {
    ExpectSize(values, 4);
    ExpectParameterBlockContainsResidualBlock(values, r1);
    ExpectParameterBlockContainsResidualBlock(values, r2);
    ExpectParameterBlockContainsResidualBlock(values, r3);
    ExpectParameterBlockContainsResidualBlock(values, r4);
  }

  scoped_ptr<ProblemImpl> problem;
  double y[4], z[5], w[3];
};

TEST(Problem, SetParameterBlockConstantWithUnknownPtrDies) {
  double x[3];
  double y[2];

  Problem problem;
  problem.AddParameterBlock(x, 3);

  EXPECT_DEATH_IF_SUPPORTED(problem.SetParameterBlockConstant(y),
                            "Parameter block not found:");
}

TEST(Problem, SetParameterBlockVariableWithUnknownPtrDies) {
  double x[3];
  double y[2];

  Problem problem;
  problem.AddParameterBlock(x, 3);

  EXPECT_DEATH_IF_SUPPORTED(problem.SetParameterBlockVariable(y),
                            "Parameter block not found:");
}

TEST(Problem, SetLocalParameterizationWithUnknownPtrDies) {
  double x[3];
  double y[2];

  Problem problem;
  problem.AddParameterBlock(x, 3);

  EXPECT_DEATH_IF_SUPPORTED(
      problem.SetParameterization(y, new IdentityParameterization(3)),
      "Parameter block not found:");
}

TEST(Problem, RemoveParameterBlockWithUnknownPtrDies) {
  double x[3];
  double y[2];

  Problem problem;
  problem.AddParameterBlock(x, 3);

  EXPECT_DEATH_IF_SUPPORTED(
      problem.RemoveParameterBlock(y), "Parameter block not found:");
}

TEST(Problem, ParameterBlockQueryTest) {
  double x[3];
  double y[4];
  Problem problem;
  problem.AddParameterBlock(x, 3);
  problem.AddParameterBlock(y, 4);

  vector<int> constant_parameters;
  constant_parameters.push_back(0);
  problem.SetParameterization(
      x,
      new SubsetParameterization(3, constant_parameters));
  EXPECT_EQ(problem.ParameterBlockSize(x), 3);
  EXPECT_EQ(problem.ParameterBlockLocalSize(x), 2);
  EXPECT_EQ(problem.ParameterBlockLocalSize(y), 4);

  vector<double*> parameter_blocks;
  problem.GetParameterBlocks(&parameter_blocks);
  EXPECT_EQ(parameter_blocks.size(), 2);
  EXPECT_NE(parameter_blocks[0], parameter_blocks[1]);
  EXPECT_TRUE(parameter_blocks[0] == x || parameter_blocks[0] == y);
  EXPECT_TRUE(parameter_blocks[1] == x || parameter_blocks[1] == y);

  problem.RemoveParameterBlock(x);
  problem.GetParameterBlocks(&parameter_blocks);
  EXPECT_EQ(parameter_blocks.size(), 1);
  EXPECT_TRUE(parameter_blocks[0] == y);
}

TEST_P(DynamicProblem, RemoveParameterBlockWithNoResiduals) {
  problem->AddParameterBlock(y, 4);
  problem->AddParameterBlock(z, 5);
  problem->AddParameterBlock(w, 3);
  ASSERT_EQ(3, problem->NumParameterBlocks());
  ASSERT_EQ(0, problem->NumResidualBlocks());
  EXPECT_EQ(y, GetParameterBlock(0)->user_state());
  EXPECT_EQ(z, GetParameterBlock(1)->user_state());
  EXPECT_EQ(w, GetParameterBlock(2)->user_state());

  // w is at the end, which might break the swapping logic so try adding and
  // removing it.
  problem->RemoveParameterBlock(w);
  ASSERT_EQ(2, problem->NumParameterBlocks());
  ASSERT_EQ(0, problem->NumResidualBlocks());
  EXPECT_EQ(y, GetParameterBlock(0)->user_state());
  EXPECT_EQ(z, GetParameterBlock(1)->user_state());
  problem->AddParameterBlock(w, 3);
  ASSERT_EQ(3, problem->NumParameterBlocks());
  ASSERT_EQ(0, problem->NumResidualBlocks());
  EXPECT_EQ(y, GetParameterBlock(0)->user_state());
  EXPECT_EQ(z, GetParameterBlock(1)->user_state());
  EXPECT_EQ(w, GetParameterBlock(2)->user_state());

  // Now remove z, which is in the middle, and add it back.
  problem->RemoveParameterBlock(z);
  ASSERT_EQ(2, problem->NumParameterBlocks());
  ASSERT_EQ(0, problem->NumResidualBlocks());
  EXPECT_EQ(y, GetParameterBlock(0)->user_state());
  EXPECT_EQ(w, GetParameterBlock(1)->user_state());
  problem->AddParameterBlock(z, 5);
  ASSERT_EQ(3, problem->NumParameterBlocks());
  ASSERT_EQ(0, problem->NumResidualBlocks());
  EXPECT_EQ(y, GetParameterBlock(0)->user_state());
  EXPECT_EQ(w, GetParameterBlock(1)->user_state());
  EXPECT_EQ(z, GetParameterBlock(2)->user_state());

  // Now remove everything.
  // y
  problem->RemoveParameterBlock(y);
  ASSERT_EQ(2, problem->NumParameterBlocks());
  ASSERT_EQ(0, problem->NumResidualBlocks());
  EXPECT_EQ(z, GetParameterBlock(0)->user_state());
  EXPECT_EQ(w, GetParameterBlock(1)->user_state());

  // z
  problem->RemoveParameterBlock(z);
  ASSERT_EQ(1, problem->NumParameterBlocks());
  ASSERT_EQ(0, problem->NumResidualBlocks());
  EXPECT_EQ(w, GetParameterBlock(0)->user_state());

  // w
  problem->RemoveParameterBlock(w);
  EXPECT_EQ(0, problem->NumParameterBlocks());
  EXPECT_EQ(0, problem->NumResidualBlocks());
}

TEST_P(DynamicProblem, RemoveParameterBlockWithResiduals) {
  problem->AddParameterBlock(y, 4);
  problem->AddParameterBlock(z, 5);
  problem->AddParameterBlock(w, 3);
  ASSERT_EQ(3, problem->NumParameterBlocks());
  ASSERT_EQ(0, problem->NumResidualBlocks());
  EXPECT_EQ(y, GetParameterBlock(0)->user_state());
  EXPECT_EQ(z, GetParameterBlock(1)->user_state());
  EXPECT_EQ(w, GetParameterBlock(2)->user_state());

  // Add all combinations of cost functions.
  CostFunction* cost_yzw = new TernaryCostFunction(1, 4, 5, 3);
  CostFunction* cost_yz  = new BinaryCostFunction (1, 4, 5);
  CostFunction* cost_yw  = new BinaryCostFunction (1, 4, 3);
  CostFunction* cost_zw  = new BinaryCostFunction (1, 5, 3);
  CostFunction* cost_y   = new UnaryCostFunction  (1, 4);
  CostFunction* cost_z   = new UnaryCostFunction  (1, 5);
  CostFunction* cost_w   = new UnaryCostFunction  (1, 3);

  ResidualBlock* r_yzw = problem->AddResidualBlock(cost_yzw, NULL, y, z, w);
  ResidualBlock* r_yz  = problem->AddResidualBlock(cost_yz,  NULL, y, z);
  ResidualBlock* r_yw  = problem->AddResidualBlock(cost_yw,  NULL, y, w);
  ResidualBlock* r_zw  = problem->AddResidualBlock(cost_zw,  NULL, z, w);
  ResidualBlock* r_y   = problem->AddResidualBlock(cost_y,   NULL, y);
  ResidualBlock* r_z   = problem->AddResidualBlock(cost_z,   NULL, z);
  ResidualBlock* r_w   = problem->AddResidualBlock(cost_w,   NULL, w);

  EXPECT_EQ(3, problem->NumParameterBlocks());
  EXPECT_EQ(7, problem->NumResidualBlocks());

  // Remove w, which should remove r_yzw, r_yw, r_zw, r_w.
  problem->RemoveParameterBlock(w);
  ASSERT_EQ(2, problem->NumParameterBlocks());
  ASSERT_EQ(3, problem->NumResidualBlocks());

  ASSERT_FALSE(HasResidualBlock(r_yzw));
  ASSERT_TRUE (HasResidualBlock(r_yz ));
  ASSERT_FALSE(HasResidualBlock(r_yw ));
  ASSERT_FALSE(HasResidualBlock(r_zw ));
  ASSERT_TRUE (HasResidualBlock(r_y  ));
  ASSERT_TRUE (HasResidualBlock(r_z  ));
  ASSERT_FALSE(HasResidualBlock(r_w  ));

  // Remove z, which will remove almost everything else.
  problem->RemoveParameterBlock(z);
  ASSERT_EQ(1, problem->NumParameterBlocks());
  ASSERT_EQ(1, problem->NumResidualBlocks());

  ASSERT_FALSE(HasResidualBlock(r_yzw));
  ASSERT_FALSE(HasResidualBlock(r_yz ));
  ASSERT_FALSE(HasResidualBlock(r_yw ));
  ASSERT_FALSE(HasResidualBlock(r_zw ));
  ASSERT_TRUE (HasResidualBlock(r_y  ));
  ASSERT_FALSE(HasResidualBlock(r_z  ));
  ASSERT_FALSE(HasResidualBlock(r_w  ));

  // Remove y; all gone.
  problem->RemoveParameterBlock(y);
  EXPECT_EQ(0, problem->NumParameterBlocks());
  EXPECT_EQ(0, problem->NumResidualBlocks());
}

TEST_P(DynamicProblem, RemoveResidualBlock) {
  problem->AddParameterBlock(y, 4);
  problem->AddParameterBlock(z, 5);
  problem->AddParameterBlock(w, 3);

  // Add all combinations of cost functions.
  CostFunction* cost_yzw = new TernaryCostFunction(1, 4, 5, 3);
  CostFunction* cost_yz  = new BinaryCostFunction (1, 4, 5);
  CostFunction* cost_yw  = new BinaryCostFunction (1, 4, 3);
  CostFunction* cost_zw  = new BinaryCostFunction (1, 5, 3);
  CostFunction* cost_y   = new UnaryCostFunction  (1, 4);
  CostFunction* cost_z   = new UnaryCostFunction  (1, 5);
  CostFunction* cost_w   = new UnaryCostFunction  (1, 3);

  ResidualBlock* r_yzw = problem->AddResidualBlock(cost_yzw, NULL, y, z, w);
  ResidualBlock* r_yz  = problem->AddResidualBlock(cost_yz,  NULL, y, z);
  ResidualBlock* r_yw  = problem->AddResidualBlock(cost_yw,  NULL, y, w);
  ResidualBlock* r_zw  = problem->AddResidualBlock(cost_zw,  NULL, z, w);
  ResidualBlock* r_y   = problem->AddResidualBlock(cost_y,   NULL, y);
  ResidualBlock* r_z   = problem->AddResidualBlock(cost_z,   NULL, z);
  ResidualBlock* r_w   = problem->AddResidualBlock(cost_w,   NULL, w);

  if (GetParam()) {
    // In this test parameterization, there should be back-pointers from the
    // parameter blocks to the residual blocks.
    ExpectParameterBlockContains(y, r_yzw, r_yz, r_yw, r_y);
    ExpectParameterBlockContains(z, r_yzw, r_yz, r_zw, r_z);
    ExpectParameterBlockContains(w, r_yzw, r_yw, r_zw, r_w);
  } else {
    // Otherwise, nothing.
    EXPECT_TRUE(GetParameterBlock(0)->mutable_residual_blocks() == NULL);
    EXPECT_TRUE(GetParameterBlock(1)->mutable_residual_blocks() == NULL);
    EXPECT_TRUE(GetParameterBlock(2)->mutable_residual_blocks() == NULL);
  }
  EXPECT_EQ(3, problem->NumParameterBlocks());
  EXPECT_EQ(7, problem->NumResidualBlocks());

  // Remove each residual and check the state after each removal.

  // Remove r_yzw.
  problem->RemoveResidualBlock(r_yzw);
  ASSERT_EQ(3, problem->NumParameterBlocks());
  ASSERT_EQ(6, problem->NumResidualBlocks());
  if (GetParam()) {
    ExpectParameterBlockContains(y, r_yz, r_yw, r_y);
    ExpectParameterBlockContains(z, r_yz, r_zw, r_z);
    ExpectParameterBlockContains(w, r_yw, r_zw, r_w);
  }
  ASSERT_TRUE (HasResidualBlock(r_yz ));
  ASSERT_TRUE (HasResidualBlock(r_yw ));
  ASSERT_TRUE (HasResidualBlock(r_zw ));
  ASSERT_TRUE (HasResidualBlock(r_y  ));
  ASSERT_TRUE (HasResidualBlock(r_z  ));
  ASSERT_TRUE (HasResidualBlock(r_w  ));

  // Remove r_yw.
  problem->RemoveResidualBlock(r_yw);
  ASSERT_EQ(3, problem->NumParameterBlocks());
  ASSERT_EQ(5, problem->NumResidualBlocks());
  if (GetParam()) {
    ExpectParameterBlockContains(y, r_yz, r_y);
    ExpectParameterBlockContains(z, r_yz, r_zw, r_z);
    ExpectParameterBlockContains(w, r_zw, r_w);
  }
  ASSERT_TRUE (HasResidualBlock(r_yz ));
  ASSERT_TRUE (HasResidualBlock(r_zw ));
  ASSERT_TRUE (HasResidualBlock(r_y  ));
  ASSERT_TRUE (HasResidualBlock(r_z  ));
  ASSERT_TRUE (HasResidualBlock(r_w  ));

  // Remove r_zw.
  problem->RemoveResidualBlock(r_zw);
  ASSERT_EQ(3, problem->NumParameterBlocks());
  ASSERT_EQ(4, problem->NumResidualBlocks());
  if (GetParam()) {
    ExpectParameterBlockContains(y, r_yz, r_y);
    ExpectParameterBlockContains(z, r_yz, r_z);
    ExpectParameterBlockContains(w, r_w);
  }
  ASSERT_TRUE (HasResidualBlock(r_yz ));
  ASSERT_TRUE (HasResidualBlock(r_y  ));
  ASSERT_TRUE (HasResidualBlock(r_z  ));
  ASSERT_TRUE (HasResidualBlock(r_w  ));

  // Remove r_w.
  problem->RemoveResidualBlock(r_w);
  ASSERT_EQ(3, problem->NumParameterBlocks());
  ASSERT_EQ(3, problem->NumResidualBlocks());
  if (GetParam()) {
    ExpectParameterBlockContains(y, r_yz, r_y);
    ExpectParameterBlockContains(z, r_yz, r_z);
    ExpectParameterBlockContains(w);
  }
  ASSERT_TRUE (HasResidualBlock(r_yz ));
  ASSERT_TRUE (HasResidualBlock(r_y  ));
  ASSERT_TRUE (HasResidualBlock(r_z  ));

  // Remove r_yz.
  problem->RemoveResidualBlock(r_yz);
  ASSERT_EQ(3, problem->NumParameterBlocks());
  ASSERT_EQ(2, problem->NumResidualBlocks());
  if (GetParam()) {
    ExpectParameterBlockContains(y, r_y);
    ExpectParameterBlockContains(z, r_z);
    ExpectParameterBlockContains(w);
  }
  ASSERT_TRUE (HasResidualBlock(r_y  ));
  ASSERT_TRUE (HasResidualBlock(r_z  ));

  // Remove the last two.
  problem->RemoveResidualBlock(r_z);
  problem->RemoveResidualBlock(r_y);
  ASSERT_EQ(3, problem->NumParameterBlocks());
  ASSERT_EQ(0, problem->NumResidualBlocks());
  if (GetParam()) {
    ExpectParameterBlockContains(y);
    ExpectParameterBlockContains(z);
    ExpectParameterBlockContains(w);
  }
}

INSTANTIATE_TEST_CASE_P(OptionsInstantiation,
                        DynamicProblem,
                        ::testing::Values(true, false));

// Test for Problem::Evaluate

// r_i = i - (j + 1) * x_ij^2
template <int kNumResiduals, int kNumParameterBlocks>
class QuadraticCostFunction : public CostFunction {
 public:
  QuadraticCostFunction() {
    CHECK_GT(kNumResiduals, 0);
    CHECK_GT(kNumParameterBlocks, 0);
    set_num_residuals(kNumResiduals);
    for (int i = 0; i < kNumParameterBlocks; ++i) {
      mutable_parameter_block_sizes()->push_back(kNumResiduals);
    }
  }

  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    for (int i = 0; i < kNumResiduals; ++i) {
      residuals[i] = i;
      for (int j = 0; j < kNumParameterBlocks; ++j) {
        residuals[i] -= (j + 1.0) * parameters[j][i] * parameters[j][i];
      }
    }

    if (jacobians == NULL) {
      return true;
    }

    for (int j = 0; j < kNumParameterBlocks; ++j) {
      if (jacobians[j] != NULL) {
        MatrixRef(jacobians[j], kNumResiduals, kNumResiduals) =
            (-2.0 * (j + 1.0) *
             ConstVectorRef(parameters[j], kNumResiduals)).asDiagonal();
      }
    }

    return true;
  }
};

// Convert a CRSMatrix to a dense Eigen matrix.
void CRSToDenseMatrix(const CRSMatrix& input, Matrix* output) {
  Matrix& m = *CHECK_NOTNULL(output);
  m.resize(input.num_rows, input.num_cols);
  m.setZero();
  for (int row = 0; row < input.num_rows; ++row) {
    for (int j = input.rows[row]; j < input.rows[row + 1]; ++j) {
      const int col = input.cols[j];
      m(row, col) = input.values[j];
    }
  }
}

class ProblemEvaluateTest : public ::testing::Test {
 protected:
  void SetUp() {
    for (int i = 0; i < 6; ++i) {
      parameters_[i] = static_cast<double>(i + 1);
    }

    parameter_blocks_.push_back(parameters_);
    parameter_blocks_.push_back(parameters_ + 2);
    parameter_blocks_.push_back(parameters_ + 4);


    CostFunction* cost_function = new QuadraticCostFunction<2, 2>;

    // f(x, y)
    residual_blocks_.push_back(
        problem_.AddResidualBlock(cost_function,
                                  NULL,
                                  parameters_,
                                  parameters_ + 2));
    // g(y, z)
    residual_blocks_.push_back(
        problem_.AddResidualBlock(cost_function,
                                  NULL, parameters_ + 2,
                                  parameters_ + 4));
    // h(z, x)
    residual_blocks_.push_back(
        problem_.AddResidualBlock(cost_function,
                                  NULL,
                                  parameters_ + 4,
                                  parameters_));
  }



  void EvaluateAndCompare(const Problem::EvaluateOptions& options,
                          const int expected_num_rows,
                          const int expected_num_cols,
                          const double expected_cost,
                          const double* expected_residuals,
                          const double* expected_gradient,
                          const double* expected_jacobian) {
    double cost;
    vector<double> residuals;
    vector<double> gradient;
    CRSMatrix jacobian;

    EXPECT_TRUE(
        problem_.Evaluate(options,
                          &cost,
                          expected_residuals != NULL ? &residuals : NULL,
                          expected_gradient != NULL ? &gradient : NULL,
                          expected_jacobian != NULL ? &jacobian : NULL));

    if (expected_residuals != NULL) {
      EXPECT_EQ(residuals.size(), expected_num_rows);
    }

    if (expected_gradient != NULL) {
      EXPECT_EQ(gradient.size(), expected_num_cols);
    }

    if (expected_jacobian != NULL) {
      EXPECT_EQ(jacobian.num_rows, expected_num_rows);
      EXPECT_EQ(jacobian.num_cols, expected_num_cols);
    }

    Matrix dense_jacobian;
    if (expected_jacobian != NULL) {
      CRSToDenseMatrix(jacobian, &dense_jacobian);
    }

    CompareEvaluations(expected_num_rows,
                       expected_num_cols,
                       expected_cost,
                       expected_residuals,
                       expected_gradient,
                       expected_jacobian,
                       cost,
                       residuals.size() > 0 ? &residuals[0] : NULL,
                       gradient.size() > 0 ? &gradient[0] : NULL,
                       dense_jacobian.data());
  }

  void CheckAllEvaluationCombinations(const Problem::EvaluateOptions& options,
                                      const ExpectedEvaluation& expected) {
    for (int i = 0; i < 8; ++i) {
      EvaluateAndCompare(options,
                         expected.num_rows,
                         expected.num_cols,
                         expected.cost,
                         (i & 1) ? expected.residuals : NULL,
                         (i & 2) ? expected.gradient  : NULL,
                         (i & 4) ? expected.jacobian  : NULL);
    }
  }

  ProblemImpl problem_;
  double parameters_[6];
  vector<double*> parameter_blocks_;
  vector<ResidualBlockId> residual_blocks_;
};


TEST_F(ProblemEvaluateTest, MultipleParameterAndResidualBlocks) {
  ExpectedEvaluation expected = {
    // Rows/columns
    6, 6,
    // Cost
    7607.0,
    // Residuals
    { -19.0, -35.0,  // f
      -59.0, -87.0,  // g
      -27.0, -43.0   // h
    },
    // Gradient
    {  146.0,  484.0,   // x
       582.0, 1256.0,   // y
      1450.0, 2604.0,   // z
    },
    // Jacobian
    //                       x             y             z
    { /* f(x, y) */ -2.0,  0.0, -12.0,   0.0,   0.0,   0.0,
                     0.0, -4.0,   0.0, -16.0,   0.0,   0.0,
      /* g(y, z) */  0.0,  0.0,  -6.0,   0.0, -20.0,   0.0,
                     0.0,  0.0,   0.0,  -8.0,   0.0, -24.0,
      /* h(z, x) */ -4.0,  0.0,   0.0,   0.0, -10.0,   0.0,
                     0.0, -8.0,   0.0,   0.0,   0.0, -12.0
    }
  };

  CheckAllEvaluationCombinations(Problem::EvaluateOptions(), expected);
}

TEST_F(ProblemEvaluateTest, ParameterAndResidualBlocksPassedInOptions) {
  ExpectedEvaluation expected = {
    // Rows/columns
    6, 6,
    // Cost
    7607.0,
    // Residuals
    { -19.0, -35.0,  // f
      -59.0, -87.0,  // g
      -27.0, -43.0   // h
    },
    // Gradient
    {  146.0,  484.0,   // x
       582.0, 1256.0,   // y
      1450.0, 2604.0,   // z
    },
    // Jacobian
    //                       x             y             z
    { /* f(x, y) */ -2.0,  0.0, -12.0,   0.0,   0.0,   0.0,
                     0.0, -4.0,   0.0, -16.0,   0.0,   0.0,
      /* g(y, z) */  0.0,  0.0,  -6.0,   0.0, -20.0,   0.0,
                     0.0,  0.0,   0.0,  -8.0,   0.0, -24.0,
      /* h(z, x) */ -4.0,  0.0,   0.0,   0.0, -10.0,   0.0,
                     0.0, -8.0,   0.0,   0.0,   0.0, -12.0
    }
  };

  Problem::EvaluateOptions evaluate_options;
  evaluate_options.parameter_blocks = parameter_blocks_;
  evaluate_options.residual_blocks = residual_blocks_;
  CheckAllEvaluationCombinations(evaluate_options, expected);
}

TEST_F(ProblemEvaluateTest, ReorderedResidualBlocks) {
  ExpectedEvaluation expected = {
    // Rows/columns
    6, 6,
    // Cost
    7607.0,
    // Residuals
    { -19.0, -35.0,  // f
      -27.0, -43.0,  // h
      -59.0, -87.0   // g
    },
    // Gradient
    {  146.0,  484.0,   // x
       582.0, 1256.0,   // y
      1450.0, 2604.0,   // z
    },
    // Jacobian
    //                       x             y             z
    { /* f(x, y) */ -2.0,  0.0, -12.0,   0.0,   0.0,   0.0,
                     0.0, -4.0,   0.0, -16.0,   0.0,   0.0,
      /* h(z, x) */ -4.0,  0.0,   0.0,   0.0, -10.0,   0.0,
                     0.0, -8.0,   0.0,   0.0,   0.0, -12.0,
      /* g(y, z) */  0.0,  0.0,  -6.0,   0.0, -20.0,   0.0,
                     0.0,  0.0,   0.0,  -8.0,   0.0, -24.0
    }
  };

  Problem::EvaluateOptions evaluate_options;
  evaluate_options.parameter_blocks = parameter_blocks_;

  // f, h, g
  evaluate_options.residual_blocks.push_back(residual_blocks_[0]);
  evaluate_options.residual_blocks.push_back(residual_blocks_[2]);
  evaluate_options.residual_blocks.push_back(residual_blocks_[1]);

  CheckAllEvaluationCombinations(evaluate_options, expected);
}

TEST_F(ProblemEvaluateTest, ReorderedResidualBlocksAndReorderedParameterBlocks) {
  ExpectedEvaluation expected = {
    // Rows/columns
    6, 6,
    // Cost
    7607.0,
    // Residuals
    { -19.0, -35.0,  // f
      -27.0, -43.0,  // h
      -59.0, -87.0   // g
    },
    // Gradient
    {  1450.0, 2604.0,   // z
        582.0, 1256.0,   // y
        146.0,  484.0,   // x
    },
     // Jacobian
    //                       z             y             x
    { /* f(x, y) */   0.0,   0.0, -12.0,   0.0,  -2.0,   0.0,
                      0.0,   0.0,   0.0, -16.0,   0.0,  -4.0,
      /* h(z, x) */ -10.0,   0.0,   0.0,   0.0,  -4.0,   0.0,
                      0.0, -12.0,   0.0,   0.0,   0.0,  -8.0,
      /* g(y, z) */ -20.0,   0.0,  -6.0,   0.0,   0.0,   0.0,
                      0.0, -24.0,   0.0,  -8.0,   0.0,   0.0
    }
  };

  Problem::EvaluateOptions evaluate_options;
  // z, y, x
  evaluate_options.parameter_blocks.push_back(parameter_blocks_[2]);
  evaluate_options.parameter_blocks.push_back(parameter_blocks_[1]);
  evaluate_options.parameter_blocks.push_back(parameter_blocks_[0]);

  // f, h, g
  evaluate_options.residual_blocks.push_back(residual_blocks_[0]);
  evaluate_options.residual_blocks.push_back(residual_blocks_[2]);
  evaluate_options.residual_blocks.push_back(residual_blocks_[1]);

  CheckAllEvaluationCombinations(evaluate_options, expected);
}

TEST_F(ProblemEvaluateTest, ConstantParameterBlock) {
  ExpectedEvaluation expected = {
    // Rows/columns
    6, 6,
    // Cost
    7607.0,
    // Residuals
    { -19.0, -35.0,  // f
      -59.0, -87.0,  // g
      -27.0, -43.0   // h
    },

    // Gradient
    {  146.0,  484.0,  // x
         0.0,    0.0,  // y
      1450.0, 2604.0,  // z
    },

    // Jacobian
    //                       x             y             z
    { /* f(x, y) */ -2.0,  0.0,   0.0,   0.0,   0.0,   0.0,
                     0.0, -4.0,   0.0,   0.0,   0.0,   0.0,
      /* g(y, z) */  0.0,  0.0,   0.0,   0.0, -20.0,   0.0,
                     0.0,  0.0,   0.0,   0.0,   0.0, -24.0,
      /* h(z, x) */ -4.0,  0.0,   0.0,   0.0, -10.0,   0.0,
                     0.0, -8.0,   0.0,   0.0,   0.0, -12.0
    }
  };

  problem_.SetParameterBlockConstant(parameters_ + 2);
  CheckAllEvaluationCombinations(Problem::EvaluateOptions(), expected);
}

TEST_F(ProblemEvaluateTest, ExcludedAResidualBlock) {
  ExpectedEvaluation expected = {
    // Rows/columns
    4, 6,
    // Cost
    2082.0,
    // Residuals
    { -19.0, -35.0,  // f
      -27.0, -43.0   // h
    },
    // Gradient
    {  146.0,  484.0,   // x
       228.0,  560.0,   // y
       270.0,  516.0,   // z
    },
    // Jacobian
    //                       x             y             z
    { /* f(x, y) */ -2.0,  0.0, -12.0,   0.0,   0.0,   0.0,
                     0.0, -4.0,   0.0, -16.0,   0.0,   0.0,
      /* h(z, x) */ -4.0,  0.0,   0.0,   0.0, -10.0,   0.0,
                     0.0, -8.0,   0.0,   0.0,   0.0, -12.0
    }
  };

  Problem::EvaluateOptions evaluate_options;
  evaluate_options.residual_blocks.push_back(residual_blocks_[0]);
  evaluate_options.residual_blocks.push_back(residual_blocks_[2]);

  CheckAllEvaluationCombinations(evaluate_options, expected);
}

TEST_F(ProblemEvaluateTest, ExcludedParameterBlock) {
  ExpectedEvaluation expected = {
    // Rows/columns
    6, 4,
    // Cost
    7607.0,
    // Residuals
    { -19.0, -35.0,  // f
      -59.0, -87.0,  // g
      -27.0, -43.0   // h
    },

    // Gradient
    {  146.0,  484.0,  // x
      1450.0, 2604.0,  // z
    },

    // Jacobian
    //                       x             z
    { /* f(x, y) */ -2.0,  0.0,   0.0,   0.0,
                     0.0, -4.0,   0.0,   0.0,
      /* g(y, z) */  0.0,  0.0, -20.0,   0.0,
                     0.0,  0.0,   0.0, -24.0,
      /* h(z, x) */ -4.0,  0.0, -10.0,   0.0,
                     0.0, -8.0,   0.0, -12.0
    }
  };

  Problem::EvaluateOptions evaluate_options;
  // x, z
  evaluate_options.parameter_blocks.push_back(parameter_blocks_[0]);
  evaluate_options.parameter_blocks.push_back(parameter_blocks_[2]);
  evaluate_options.residual_blocks = residual_blocks_;
  CheckAllEvaluationCombinations(evaluate_options, expected);
}

TEST_F(ProblemEvaluateTest, ExcludedParameterBlockAndExcludedResidualBlock) {
  ExpectedEvaluation expected = {
    // Rows/columns
    4, 4,
    // Cost
    6318.0,
    // Residuals
    { -19.0, -35.0,  // f
      -59.0, -87.0,  // g
    },

    // Gradient
    {   38.0,  140.0,  // x
      1180.0, 2088.0,  // z
    },

    // Jacobian
    //                       x             z
    { /* f(x, y) */ -2.0,  0.0,   0.0,   0.0,
                     0.0, -4.0,   0.0,   0.0,
      /* g(y, z) */  0.0,  0.0, -20.0,   0.0,
                     0.0,  0.0,   0.0, -24.0,
    }
  };

  Problem::EvaluateOptions evaluate_options;
  // x, z
  evaluate_options.parameter_blocks.push_back(parameter_blocks_[0]);
  evaluate_options.parameter_blocks.push_back(parameter_blocks_[2]);
  evaluate_options.residual_blocks.push_back(residual_blocks_[0]);
  evaluate_options.residual_blocks.push_back(residual_blocks_[1]);

  CheckAllEvaluationCombinations(evaluate_options, expected);
}

TEST_F(ProblemEvaluateTest, LocalParameterization) {
  ExpectedEvaluation expected = {
    // Rows/columns
    6, 5,
    // Cost
    7607.0,
    // Residuals
    { -19.0, -35.0,  // f
      -59.0, -87.0,  // g
      -27.0, -43.0   // h
    },
    // Gradient
    {  146.0,  484.0,  // x
      1256.0,          // y with SubsetParameterization
      1450.0, 2604.0,  // z
    },
    // Jacobian
    //                       x      y             z
    { /* f(x, y) */ -2.0,  0.0,   0.0,   0.0,   0.0,
                     0.0, -4.0, -16.0,   0.0,   0.0,
      /* g(y, z) */  0.0,  0.0,   0.0, -20.0,   0.0,
                     0.0,  0.0,  -8.0,   0.0, -24.0,
      /* h(z, x) */ -4.0,  0.0,   0.0, -10.0,   0.0,
                     0.0, -8.0,   0.0,   0.0, -12.0
    }
  };

  vector<int> constant_parameters;
  constant_parameters.push_back(0);
  problem_.SetParameterization(parameters_ + 2,
                               new SubsetParameterization(2,
                                                          constant_parameters));

  CheckAllEvaluationCombinations(Problem::EvaluateOptions(), expected);
}

}  // namespace internal
}  // namespace ceres
