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

#include "gtest/gtest.h"
#include "ceres/autodiff_cost_function.h"
#include "ceres/linear_solver.h"
#include "ceres/ordered_groups.h"
#include "ceres/parameter_block.h"
#include "ceres/problem_impl.h"
#include "ceres/program.h"
#include "ceres/residual_block.h"
#include "ceres/solver_impl.h"
#include "ceres/sized_cost_function.h"

namespace ceres {
namespace internal {

// A cost function that sipmply returns its argument.
class UnaryIdentityCostFunction : public SizedCostFunction<1, 1> {
 public:
  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    residuals[0] = parameters[0][0];
    if (jacobians != NULL && jacobians[0] != NULL) {
      jacobians[0][0] = 1.0;
    }
    return true;
  }
};

// Templated base class for the CostFunction signatures.
template <int kNumResiduals, int N0, int N1, int N2>
class MockCostFunctionBase : public
SizedCostFunction<kNumResiduals, N0, N1, N2> {
 public:
  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    // Do nothing. This is never called.
    return true;
  }
};

class UnaryCostFunction : public MockCostFunctionBase<2, 1, 0, 0> {};
class BinaryCostFunction : public MockCostFunctionBase<2, 1, 1, 0> {};
class TernaryCostFunction : public MockCostFunctionBase<2, 1, 1, 1> {};

TEST(SolverImpl, RemoveFixedBlocksNothingConstant) {
  ProblemImpl problem;
  double x;
  double y;
  double z;

  problem.AddParameterBlock(&x, 1);
  problem.AddParameterBlock(&y, 1);
  problem.AddParameterBlock(&z, 1);
  problem.AddResidualBlock(new UnaryCostFunction(), NULL, &x);
  problem.AddResidualBlock(new BinaryCostFunction(), NULL, &x, &y);
  problem.AddResidualBlock(new TernaryCostFunction(), NULL, &x, &y, &z);

  string error;
  {
    ParameterBlockOrdering ordering;
    ordering.AddElementToGroup(&x, 0);
    ordering.AddElementToGroup(&y, 0);
    ordering.AddElementToGroup(&z, 0);

    Program program(*problem.mutable_program());
    EXPECT_TRUE(SolverImpl::RemoveFixedBlocksFromProgram(&program,
                                                         &ordering,
                                                         NULL,
                                                         &error));
    EXPECT_EQ(program.NumParameterBlocks(), 3);
    EXPECT_EQ(program.NumResidualBlocks(), 3);
    EXPECT_EQ(ordering.NumElements(), 3);
  }
}

TEST(SolverImpl, RemoveFixedBlocksAllParameterBlocksConstant) {
  ProblemImpl problem;
  double x;

  problem.AddParameterBlock(&x, 1);
  problem.AddResidualBlock(new UnaryCostFunction(), NULL, &x);
  problem.SetParameterBlockConstant(&x);

  ParameterBlockOrdering ordering;
  ordering.AddElementToGroup(&x, 0);

  Program program(problem.program());
  string error;
  EXPECT_TRUE(SolverImpl::RemoveFixedBlocksFromProgram(&program,
                                                       &ordering,
                                                       NULL,
                                                       &error));
  EXPECT_EQ(program.NumParameterBlocks(), 0);
  EXPECT_EQ(program.NumResidualBlocks(), 0);
  EXPECT_EQ(ordering.NumElements(), 0);
}

TEST(SolverImpl, RemoveFixedBlocksNoResidualBlocks) {
  ProblemImpl problem;
  double x;
  double y;
  double z;

  problem.AddParameterBlock(&x, 1);
  problem.AddParameterBlock(&y, 1);
  problem.AddParameterBlock(&z, 1);

  ParameterBlockOrdering ordering;
  ordering.AddElementToGroup(&x, 0);
  ordering.AddElementToGroup(&y, 0);
  ordering.AddElementToGroup(&z, 0);


  Program program(problem.program());
  string error;
  EXPECT_TRUE(SolverImpl::RemoveFixedBlocksFromProgram(&program,
                                                       &ordering,
                                                       NULL,
                                                       &error));
  EXPECT_EQ(program.NumParameterBlocks(), 0);
  EXPECT_EQ(program.NumResidualBlocks(), 0);
  EXPECT_EQ(ordering.NumElements(), 0);
}

TEST(SolverImpl, RemoveFixedBlocksOneParameterBlockConstant) {
  ProblemImpl problem;
  double x;
  double y;
  double z;

  problem.AddParameterBlock(&x, 1);
  problem.AddParameterBlock(&y, 1);
  problem.AddParameterBlock(&z, 1);

  ParameterBlockOrdering ordering;
  ordering.AddElementToGroup(&x, 0);
  ordering.AddElementToGroup(&y, 0);
  ordering.AddElementToGroup(&z, 0);

  problem.AddResidualBlock(new UnaryCostFunction(), NULL, &x);
  problem.AddResidualBlock(new BinaryCostFunction(), NULL, &x, &y);
  problem.SetParameterBlockConstant(&x);


  Program program(problem.program());
  string error;
  EXPECT_TRUE(SolverImpl::RemoveFixedBlocksFromProgram(&program,
                                                       &ordering,
                                                       NULL,
                                                       &error));
  EXPECT_EQ(program.NumParameterBlocks(), 1);
  EXPECT_EQ(program.NumResidualBlocks(), 1);
  EXPECT_EQ(ordering.NumElements(), 1);
}

TEST(SolverImpl, RemoveFixedBlocksNumEliminateBlocks) {
  ProblemImpl problem;
  double x;
  double y;
  double z;

  problem.AddParameterBlock(&x, 1);
  problem.AddParameterBlock(&y, 1);
  problem.AddParameterBlock(&z, 1);
  problem.AddResidualBlock(new UnaryCostFunction(), NULL, &x);
  problem.AddResidualBlock(new TernaryCostFunction(), NULL, &x, &y, &z);
  problem.AddResidualBlock(new BinaryCostFunction(), NULL, &x, &y);
  problem.SetParameterBlockConstant(&x);

  ParameterBlockOrdering ordering;
  ordering.AddElementToGroup(&x, 0);
  ordering.AddElementToGroup(&y, 0);
  ordering.AddElementToGroup(&z, 1);

  Program program(problem.program());
  string error;
  EXPECT_TRUE(SolverImpl::RemoveFixedBlocksFromProgram(&program,
                                                       &ordering,
                                                       NULL,
                                                       &error));
  EXPECT_EQ(program.NumParameterBlocks(), 2);
  EXPECT_EQ(program.NumResidualBlocks(), 2);
  EXPECT_EQ(ordering.NumElements(), 2);
  EXPECT_EQ(ordering.GroupId(&y), 0);
  EXPECT_EQ(ordering.GroupId(&z), 1);
}

TEST(SolverImpl, RemoveFixedBlocksFixedCost) {
  ProblemImpl problem;
  double x = 1.23;
  double y = 4.56;
  double z = 7.89;

  problem.AddParameterBlock(&x, 1);
  problem.AddParameterBlock(&y, 1);
  problem.AddParameterBlock(&z, 1);
  problem.AddResidualBlock(new UnaryIdentityCostFunction(), NULL, &x);
  problem.AddResidualBlock(new TernaryCostFunction(), NULL, &x, &y, &z);
  problem.AddResidualBlock(new BinaryCostFunction(), NULL, &x, &y);
  problem.SetParameterBlockConstant(&x);

  ParameterBlockOrdering ordering;
  ordering.AddElementToGroup(&x, 0);
  ordering.AddElementToGroup(&y, 0);
  ordering.AddElementToGroup(&z, 1);

  double fixed_cost = 0.0;
  Program program(problem.program());

  double expected_fixed_cost;
  ResidualBlock *expected_removed_block = program.residual_blocks()[0];
  scoped_array<double> scratch(
      new double[expected_removed_block->NumScratchDoublesForEvaluate()]);
  expected_removed_block->Evaluate(true,
                                   &expected_fixed_cost,
                                   NULL,
                                   NULL,
                                   scratch.get());

  string error;
  EXPECT_TRUE(SolverImpl::RemoveFixedBlocksFromProgram(&program,
                                                       &ordering,
                                                       &fixed_cost,
                                                       &error));
  EXPECT_EQ(program.NumParameterBlocks(), 2);
  EXPECT_EQ(program.NumResidualBlocks(), 2);
  EXPECT_EQ(ordering.NumElements(), 2);
  EXPECT_EQ(ordering.GroupId(&y), 0);
  EXPECT_EQ(ordering.GroupId(&z), 1);
  EXPECT_DOUBLE_EQ(fixed_cost, expected_fixed_cost);
}

TEST(SolverImpl, ReorderResidualBlockNormalFunction) {
  ProblemImpl problem;
  double x;
  double y;
  double z;

  problem.AddParameterBlock(&x, 1);
  problem.AddParameterBlock(&y, 1);
  problem.AddParameterBlock(&z, 1);

  problem.AddResidualBlock(new UnaryCostFunction(), NULL, &x);
  problem.AddResidualBlock(new BinaryCostFunction(), NULL, &z, &x);
  problem.AddResidualBlock(new BinaryCostFunction(), NULL, &z, &y);
  problem.AddResidualBlock(new UnaryCostFunction(), NULL, &z);
  problem.AddResidualBlock(new BinaryCostFunction(), NULL, &x, &y);
  problem.AddResidualBlock(new UnaryCostFunction(), NULL, &y);

  ParameterBlockOrdering* ordering = new ParameterBlockOrdering;
  ordering->AddElementToGroup(&x, 0);
  ordering->AddElementToGroup(&y, 0);
  ordering->AddElementToGroup(&z, 1);

  Solver::Options options;
  options.linear_solver_type = DENSE_SCHUR;
  options.linear_solver_ordering = ordering;

  const vector<ResidualBlock*>& residual_blocks =
      problem.program().residual_blocks();

  vector<ResidualBlock*> expected_residual_blocks;

  // This is a bit fragile, but it serves the purpose. We know the
  // bucketing algorithm that the reordering function uses, so we
  // expect the order for residual blocks for each e_block to be
  // filled in reverse.
  expected_residual_blocks.push_back(residual_blocks[4]);
  expected_residual_blocks.push_back(residual_blocks[1]);
  expected_residual_blocks.push_back(residual_blocks[0]);
  expected_residual_blocks.push_back(residual_blocks[5]);
  expected_residual_blocks.push_back(residual_blocks[2]);
  expected_residual_blocks.push_back(residual_blocks[3]);

  Program* program = problem.mutable_program();
  program->SetParameterOffsetsAndIndex();

  string error;
  EXPECT_TRUE(SolverImpl::LexicographicallyOrderResidualBlocks(
                  2,
                  problem.mutable_program(),
                  &error));
  EXPECT_EQ(residual_blocks.size(), expected_residual_blocks.size());
  for (int i = 0; i < expected_residual_blocks.size(); ++i) {
    EXPECT_EQ(residual_blocks[i], expected_residual_blocks[i]);
  }
}

TEST(SolverImpl, ReorderResidualBlockNormalFunctionWithFixedBlocks) {
  ProblemImpl problem;
  double x;
  double y;
  double z;

  problem.AddParameterBlock(&x, 1);
  problem.AddParameterBlock(&y, 1);
  problem.AddParameterBlock(&z, 1);

  // Set one parameter block constant.
  problem.SetParameterBlockConstant(&z);

  // Mark residuals for x's row block with "x" for readability.
  problem.AddResidualBlock(new UnaryCostFunction(), NULL, &x);       // 0 x
  problem.AddResidualBlock(new BinaryCostFunction(), NULL, &z, &x);  // 1 x
  problem.AddResidualBlock(new BinaryCostFunction(), NULL, &z, &y);  // 2
  problem.AddResidualBlock(new BinaryCostFunction(), NULL, &z, &y);  // 3
  problem.AddResidualBlock(new BinaryCostFunction(), NULL, &x, &z);  // 4 x
  problem.AddResidualBlock(new BinaryCostFunction(), NULL, &z, &y);  // 5
  problem.AddResidualBlock(new BinaryCostFunction(), NULL, &x, &z);  // 6 x
  problem.AddResidualBlock(new UnaryCostFunction(), NULL, &y);       // 7

  ParameterBlockOrdering* ordering = new ParameterBlockOrdering;
  ordering->AddElementToGroup(&x, 0);
  ordering->AddElementToGroup(&z, 0);
  ordering->AddElementToGroup(&y, 1);

  Solver::Options options;
  options.linear_solver_type = DENSE_SCHUR;
  options.linear_solver_ordering = ordering;

  // Create the reduced program. This should remove the fixed block "z",
  // marking the index to -1 at the same time. x and y also get indices.
  string error;
  scoped_ptr<Program> reduced_program(
      SolverImpl::CreateReducedProgram(&options, &problem, NULL, &error));

  const vector<ResidualBlock*>& residual_blocks =
      problem.program().residual_blocks();

  // This is a bit fragile, but it serves the purpose. We know the
  // bucketing algorithm that the reordering function uses, so we
  // expect the order for residual blocks for each e_block to be
  // filled in reverse.

  vector<ResidualBlock*> expected_residual_blocks;

  // Row block for residuals involving "x". These are marked "x" in the block
  // of code calling AddResidual() above.
  expected_residual_blocks.push_back(residual_blocks[6]);
  expected_residual_blocks.push_back(residual_blocks[4]);
  expected_residual_blocks.push_back(residual_blocks[1]);
  expected_residual_blocks.push_back(residual_blocks[0]);

  // Row block for residuals involving "y".
  expected_residual_blocks.push_back(residual_blocks[7]);
  expected_residual_blocks.push_back(residual_blocks[5]);
  expected_residual_blocks.push_back(residual_blocks[3]);
  expected_residual_blocks.push_back(residual_blocks[2]);

  EXPECT_EQ(reduced_program->residual_blocks().size(),
            expected_residual_blocks.size());
  for (int i = 0; i < expected_residual_blocks.size(); ++i) {
    EXPECT_EQ(reduced_program->residual_blocks()[i],
              expected_residual_blocks[i]);
  }
}

TEST(SolverImpl, AutomaticSchurReorderingRespectsConstantBlocks) {
  ProblemImpl problem;
  double x;
  double y;
  double z;

  problem.AddParameterBlock(&x, 1);
  problem.AddParameterBlock(&y, 1);
  problem.AddParameterBlock(&z, 1);

  // Set one parameter block constant.
  problem.SetParameterBlockConstant(&z);

  problem.AddResidualBlock(new UnaryCostFunction(), NULL, &x);
  problem.AddResidualBlock(new BinaryCostFunction(), NULL, &z, &x);
  problem.AddResidualBlock(new BinaryCostFunction(), NULL, &z, &y);
  problem.AddResidualBlock(new BinaryCostFunction(), NULL, &z, &y);
  problem.AddResidualBlock(new BinaryCostFunction(), NULL, &x, &z);
  problem.AddResidualBlock(new BinaryCostFunction(), NULL, &z, &y);
  problem.AddResidualBlock(new BinaryCostFunction(), NULL, &x, &z);
  problem.AddResidualBlock(new UnaryCostFunction(), NULL, &y);
  problem.AddResidualBlock(new UnaryCostFunction(), NULL, &z);

  ParameterBlockOrdering* ordering = new ParameterBlockOrdering;
  ordering->AddElementToGroup(&x, 0);
  ordering->AddElementToGroup(&z, 0);
  ordering->AddElementToGroup(&y, 0);

  Solver::Options options;
  options.linear_solver_type = DENSE_SCHUR;
  options.linear_solver_ordering = ordering;

  string error;
  scoped_ptr<Program> reduced_program(
      SolverImpl::CreateReducedProgram(&options, &problem, NULL, &error));

  const vector<ResidualBlock*>& residual_blocks =
      reduced_program->residual_blocks();
  const vector<ParameterBlock*>& parameter_blocks =
      reduced_program->parameter_blocks();

  const vector<ResidualBlock*>& original_residual_blocks =
      problem.program().residual_blocks();

  EXPECT_EQ(residual_blocks.size(), 8);
  EXPECT_EQ(reduced_program->parameter_blocks().size(), 2);

  // Verify that right parmeter block and the residual blocks have
  // been removed.
  for (int i = 0; i < 8; ++i) {
    EXPECT_NE(residual_blocks[i], original_residual_blocks.back());
  }
  for (int i = 0; i < 2; ++i) {
    EXPECT_NE(parameter_blocks[i]->mutable_user_state(), &z);
  }
}

TEST(SolverImpl, ApplyUserOrderingOrderingTooSmall) {
  ProblemImpl problem;
  double x;
  double y;
  double z;

  problem.AddParameterBlock(&x, 1);
  problem.AddParameterBlock(&y, 1);
  problem.AddParameterBlock(&z, 1);

  ParameterBlockOrdering ordering;
  ordering.AddElementToGroup(&x, 0);
  ordering.AddElementToGroup(&y, 1);

  Program program(problem.program());
  string error;
  EXPECT_FALSE(SolverImpl::ApplyUserOrdering(problem.parameter_map(),
                                             &ordering,
                                             &program,
                                             &error));
}

TEST(SolverImpl, ApplyUserOrderingNormal) {
  ProblemImpl problem;
  double x;
  double y;
  double z;

  problem.AddParameterBlock(&x, 1);
  problem.AddParameterBlock(&y, 1);
  problem.AddParameterBlock(&z, 1);

  ParameterBlockOrdering ordering;
  ordering.AddElementToGroup(&x, 0);
  ordering.AddElementToGroup(&y, 2);
  ordering.AddElementToGroup(&z, 1);

  Program* program = problem.mutable_program();
  string error;

  EXPECT_TRUE(SolverImpl::ApplyUserOrdering(problem.parameter_map(),
                                            &ordering,
                                            program,
                                            &error));
  const vector<ParameterBlock*>& parameter_blocks = program->parameter_blocks();

  EXPECT_EQ(parameter_blocks.size(), 3);
  EXPECT_EQ(parameter_blocks[0]->user_state(), &x);
  EXPECT_EQ(parameter_blocks[1]->user_state(), &z);
  EXPECT_EQ(parameter_blocks[2]->user_state(), &y);
}

#if defined(CERES_NO_SUITESPARSE) && defined(CERES_NO_CXSPARSE)
TEST(SolverImpl, CreateLinearSolverNoSuiteSparse) {
  Solver::Options options;
  options.linear_solver_type = SPARSE_NORMAL_CHOLESKY;
  // CreateLinearSolver assumes a non-empty ordering.
  options.linear_solver_ordering = new ParameterBlockOrdering;
  string error;
  EXPECT_FALSE(SolverImpl::CreateLinearSolver(&options, &error));
}
#endif

TEST(SolverImpl, CreateLinearSolverNegativeMaxNumIterations) {
  Solver::Options options;
  options.linear_solver_type = DENSE_QR;
  options.linear_solver_max_num_iterations = -1;
  // CreateLinearSolver assumes a non-empty ordering.
  options.linear_solver_ordering = new ParameterBlockOrdering;
  string error;
  EXPECT_EQ(SolverImpl::CreateLinearSolver(&options, &error),
            static_cast<LinearSolver*>(NULL));
}

TEST(SolverImpl, CreateLinearSolverNegativeMinNumIterations) {
  Solver::Options options;
  options.linear_solver_type = DENSE_QR;
  options.linear_solver_min_num_iterations = -1;
  // CreateLinearSolver assumes a non-empty ordering.
  options.linear_solver_ordering = new ParameterBlockOrdering;
  string error;
  EXPECT_EQ(SolverImpl::CreateLinearSolver(&options, &error),
            static_cast<LinearSolver*>(NULL));
}

TEST(SolverImpl, CreateLinearSolverMaxLessThanMinIterations) {
  Solver::Options options;
  options.linear_solver_type = DENSE_QR;
  options.linear_solver_min_num_iterations = 10;
  options.linear_solver_max_num_iterations = 5;
  options.linear_solver_ordering = new ParameterBlockOrdering;
  string error;
  EXPECT_EQ(SolverImpl::CreateLinearSolver(&options, &error),
            static_cast<LinearSolver*>(NULL));
}

TEST(SolverImpl, CreateLinearSolverDenseSchurMultipleThreads) {
  Solver::Options options;
  options.linear_solver_type = DENSE_SCHUR;
  options.num_linear_solver_threads = 2;
  // The Schur type solvers can only be created with the Ordering
  // contains at least one elimination group.
  options.linear_solver_ordering = new ParameterBlockOrdering;
  double x;
  double y;
  options.linear_solver_ordering->AddElementToGroup(&x, 0);
  options.linear_solver_ordering->AddElementToGroup(&y, 0);

  string error;
  scoped_ptr<LinearSolver> solver(
      SolverImpl::CreateLinearSolver(&options, &error));
  EXPECT_TRUE(solver != NULL);
  EXPECT_EQ(options.linear_solver_type, DENSE_SCHUR);
  EXPECT_EQ(options.num_linear_solver_threads, 2);
}

TEST(SolverImpl, CreateIterativeLinearSolverForDogleg) {
  Solver::Options options;
  options.trust_region_strategy_type = DOGLEG;
  // CreateLinearSolver assumes a non-empty ordering.
  options.linear_solver_ordering = new ParameterBlockOrdering;
  string error;
  options.linear_solver_type = ITERATIVE_SCHUR;
  EXPECT_EQ(SolverImpl::CreateLinearSolver(&options, &error),
            static_cast<LinearSolver*>(NULL));

  options.linear_solver_type = CGNR;
  EXPECT_EQ(SolverImpl::CreateLinearSolver(&options, &error),
            static_cast<LinearSolver*>(NULL));
}

TEST(SolverImpl, CreateLinearSolverNormalOperation) {
  Solver::Options options;
  scoped_ptr<LinearSolver> solver;
  options.linear_solver_type = DENSE_QR;
  // CreateLinearSolver assumes a non-empty ordering.
  options.linear_solver_ordering = new ParameterBlockOrdering;
  string error;
  solver.reset(SolverImpl::CreateLinearSolver(&options, &error));
  EXPECT_EQ(options.linear_solver_type, DENSE_QR);
  EXPECT_TRUE(solver.get() != NULL);

  options.linear_solver_type = DENSE_NORMAL_CHOLESKY;
  solver.reset(SolverImpl::CreateLinearSolver(&options, &error));
  EXPECT_EQ(options.linear_solver_type, DENSE_NORMAL_CHOLESKY);
  EXPECT_TRUE(solver.get() != NULL);

#ifndef CERES_NO_SUITESPARSE
  options.linear_solver_type = SPARSE_NORMAL_CHOLESKY;
  options.sparse_linear_algebra_library = SUITE_SPARSE;
  solver.reset(SolverImpl::CreateLinearSolver(&options, &error));
  EXPECT_EQ(options.linear_solver_type, SPARSE_NORMAL_CHOLESKY);
  EXPECT_TRUE(solver.get() != NULL);
#endif

#ifndef CERES_NO_CXSPARSE
  options.linear_solver_type = SPARSE_NORMAL_CHOLESKY;
  options.sparse_linear_algebra_library = CX_SPARSE;
  solver.reset(SolverImpl::CreateLinearSolver(&options, &error));
  EXPECT_EQ(options.linear_solver_type, SPARSE_NORMAL_CHOLESKY);
  EXPECT_TRUE(solver.get() != NULL);
#endif

  double x;
  double y;
  options.linear_solver_ordering->AddElementToGroup(&x, 0);
  options.linear_solver_ordering->AddElementToGroup(&y, 0);

  options.linear_solver_type = DENSE_SCHUR;
  solver.reset(SolverImpl::CreateLinearSolver(&options, &error));
  EXPECT_EQ(options.linear_solver_type, DENSE_SCHUR);
  EXPECT_TRUE(solver.get() != NULL);

  options.linear_solver_type = SPARSE_SCHUR;
  solver.reset(SolverImpl::CreateLinearSolver(&options, &error));

#if defined(CERES_NO_SUITESPARSE) && defined(CERES_NO_CXSPARSE)
  EXPECT_TRUE(SolverImpl::CreateLinearSolver(&options, &error) == NULL);
#else
  EXPECT_TRUE(solver.get() != NULL);
  EXPECT_EQ(options.linear_solver_type, SPARSE_SCHUR);
#endif

  options.linear_solver_type = ITERATIVE_SCHUR;
  solver.reset(SolverImpl::CreateLinearSolver(&options, &error));
  EXPECT_EQ(options.linear_solver_type, ITERATIVE_SCHUR);
  EXPECT_TRUE(solver.get() != NULL);
}

struct QuadraticCostFunction {
  template <typename T> bool operator()(const T* const x,
                                        T* residual) const {
    residual[0] = T(5.0) - *x;
    return true;
  }
};

struct RememberingCallback : public IterationCallback {
  explicit RememberingCallback(double *x) : calls(0), x(x) {}
  virtual ~RememberingCallback() {}
  virtual CallbackReturnType operator()(const IterationSummary& summary) {
    x_values.push_back(*x);
    return SOLVER_CONTINUE;
  }
  int calls;
  double *x;
  vector<double> x_values;
};

TEST(SolverImpl, UpdateStateEveryIterationOption) {
  double x = 50.0;
  const double original_x = x;

  scoped_ptr<CostFunction> cost_function(
      new AutoDiffCostFunction<QuadraticCostFunction, 1, 1>(
          new QuadraticCostFunction));

  Problem::Options problem_options;
  problem_options.cost_function_ownership = DO_NOT_TAKE_OWNERSHIP;
  ProblemImpl problem(problem_options);
  problem.AddResidualBlock(cost_function.get(), NULL, &x);

  Solver::Options options;
  options.linear_solver_type = DENSE_QR;

  RememberingCallback callback(&x);
  options.callbacks.push_back(&callback);

  Solver::Summary summary;

  int num_iterations;

  // First try: no updating.
  SolverImpl::Solve(options, &problem, &summary);
  num_iterations = summary.num_successful_steps +
                   summary.num_unsuccessful_steps;
  EXPECT_GT(num_iterations, 1);
  for (int i = 0; i < callback.x_values.size(); ++i) {
    EXPECT_EQ(50.0, callback.x_values[i]);
  }

  // Second try: with updating
  x = 50.0;
  options.update_state_every_iteration = true;
  callback.x_values.clear();
  SolverImpl::Solve(options, &problem, &summary);
  num_iterations = summary.num_successful_steps +
                   summary.num_unsuccessful_steps;
  EXPECT_GT(num_iterations, 1);
  EXPECT_EQ(original_x, callback.x_values[0]);
  EXPECT_NE(original_x, callback.x_values[1]);
}

// The parameters must be in separate blocks so that they can be individually
// set constant or not.
struct Quadratic4DCostFunction {
  template <typename T> bool operator()(const T* const x,
                                        const T* const y,
                                        const T* const z,
                                        const T* const w,
                                        T* residual) const {
    // A 4-dimension axis-aligned quadratic.
    residual[0] = T(10.0) - *x +
                  T(20.0) - *y +
                  T(30.0) - *z +
                  T(40.0) - *w;
    return true;
  }
};

TEST(SolverImpl, ConstantParameterBlocksDoNotChangeAndStateInvariantKept) {
  double x = 50.0;
  double y = 50.0;
  double z = 50.0;
  double w = 50.0;
  const double original_x = 50.0;
  const double original_y = 50.0;
  const double original_z = 50.0;
  const double original_w = 50.0;

  scoped_ptr<CostFunction> cost_function(
      new AutoDiffCostFunction<Quadratic4DCostFunction, 1, 1, 1, 1, 1>(
          new Quadratic4DCostFunction));

  Problem::Options problem_options;
  problem_options.cost_function_ownership = DO_NOT_TAKE_OWNERSHIP;

  ProblemImpl problem(problem_options);
  problem.AddResidualBlock(cost_function.get(), NULL, &x, &y, &z, &w);
  problem.SetParameterBlockConstant(&x);
  problem.SetParameterBlockConstant(&w);

  Solver::Options options;
  options.linear_solver_type = DENSE_QR;

  Solver::Summary summary;
  SolverImpl::Solve(options, &problem, &summary);

  // Verify only the non-constant parameters were mutated.
  EXPECT_EQ(original_x, x);
  EXPECT_NE(original_y, y);
  EXPECT_NE(original_z, z);
  EXPECT_EQ(original_w, w);

  // Check that the parameter block state pointers are pointing back at the
  // user state, instead of inside a random temporary vector made by Solve().
  EXPECT_EQ(&x, problem.program().parameter_blocks()[0]->state());
  EXPECT_EQ(&y, problem.program().parameter_blocks()[1]->state());
  EXPECT_EQ(&z, problem.program().parameter_blocks()[2]->state());
  EXPECT_EQ(&w, problem.program().parameter_blocks()[3]->state());
}

TEST(SolverImpl, NoParameterBlocks) {
  ProblemImpl problem_impl;
  Solver::Options options;
  Solver::Summary summary;
  SolverImpl::Solve(options, &problem_impl, &summary);
  EXPECT_EQ(summary.termination_type, DID_NOT_RUN);
  EXPECT_EQ(summary.error, "Problem contains no parameter blocks.");
}

TEST(SolverImpl, NoResiduals) {
  ProblemImpl problem_impl;
  Solver::Options options;
  Solver::Summary summary;
  double x = 1;
  problem_impl.AddParameterBlock(&x, 1);
  SolverImpl::Solve(options, &problem_impl, &summary);
  EXPECT_EQ(summary.termination_type, DID_NOT_RUN);
  EXPECT_EQ(summary.error, "Problem contains no residual blocks.");
}


TEST(SolverImpl, ProblemIsConstant) {
  ProblemImpl problem_impl;
  Solver::Options options;
  Solver::Summary summary;
  double x = 1;
  problem_impl.AddResidualBlock(new UnaryIdentityCostFunction, NULL, &x);
  problem_impl.SetParameterBlockConstant(&x);
  SolverImpl::Solve(options, &problem_impl, &summary);
  EXPECT_EQ(summary.termination_type, FUNCTION_TOLERANCE);
  EXPECT_EQ(summary.initial_cost, 1.0 / 2.0);
  EXPECT_EQ(summary.final_cost, 1.0 / 2.0);
}

TEST(SolverImpl, AlternateLinearSolverForSchurTypeLinearSolver) {
  Solver::Options options;

  options.linear_solver_type = DENSE_QR;
  SolverImpl::AlternateLinearSolverForSchurTypeLinearSolver(&options);
  EXPECT_EQ(options.linear_solver_type, DENSE_QR);

  options.linear_solver_type = DENSE_NORMAL_CHOLESKY;
  SolverImpl::AlternateLinearSolverForSchurTypeLinearSolver(&options);
  EXPECT_EQ(options.linear_solver_type, DENSE_NORMAL_CHOLESKY);

  options.linear_solver_type = SPARSE_NORMAL_CHOLESKY;
  SolverImpl::AlternateLinearSolverForSchurTypeLinearSolver(&options);
  EXPECT_EQ(options.linear_solver_type, SPARSE_NORMAL_CHOLESKY);

  options.linear_solver_type = CGNR;
  SolverImpl::AlternateLinearSolverForSchurTypeLinearSolver(&options);
  EXPECT_EQ(options.linear_solver_type, CGNR);

  options.linear_solver_type = DENSE_SCHUR;
  SolverImpl::AlternateLinearSolverForSchurTypeLinearSolver(&options);
  EXPECT_EQ(options.linear_solver_type, DENSE_QR);

  options.linear_solver_type = SPARSE_SCHUR;
  SolverImpl::AlternateLinearSolverForSchurTypeLinearSolver(&options);
  EXPECT_EQ(options.linear_solver_type, SPARSE_NORMAL_CHOLESKY);

  options.linear_solver_type = ITERATIVE_SCHUR;
  options.preconditioner_type = IDENTITY;
  SolverImpl::AlternateLinearSolverForSchurTypeLinearSolver(&options);
  EXPECT_EQ(options.linear_solver_type, CGNR);
  EXPECT_EQ(options.preconditioner_type, IDENTITY);

  options.linear_solver_type = ITERATIVE_SCHUR;
  options.preconditioner_type = JACOBI;
  SolverImpl::AlternateLinearSolverForSchurTypeLinearSolver(&options);
  EXPECT_EQ(options.linear_solver_type, CGNR);
  EXPECT_EQ(options.preconditioner_type, JACOBI);

  options.linear_solver_type = ITERATIVE_SCHUR;
  options.preconditioner_type = SCHUR_JACOBI;
  SolverImpl::AlternateLinearSolverForSchurTypeLinearSolver(&options);
  EXPECT_EQ(options.linear_solver_type, CGNR);
  EXPECT_EQ(options.preconditioner_type, JACOBI);

  options.linear_solver_type = ITERATIVE_SCHUR;
  options.preconditioner_type = CLUSTER_JACOBI;
  SolverImpl::AlternateLinearSolverForSchurTypeLinearSolver(&options);
  EXPECT_EQ(options.linear_solver_type, CGNR);
  EXPECT_EQ(options.preconditioner_type, JACOBI);

  options.linear_solver_type = ITERATIVE_SCHUR;
  options.preconditioner_type = CLUSTER_TRIDIAGONAL;
  SolverImpl::AlternateLinearSolverForSchurTypeLinearSolver(&options);
  EXPECT_EQ(options.linear_solver_type, CGNR);
  EXPECT_EQ(options.preconditioner_type, JACOBI);
}
}  // namespace internal
}  // namespace ceres
