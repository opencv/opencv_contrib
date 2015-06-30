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
// Tests shared across evaluators. The tests try all combinations of linear
// solver and num_eliminate_blocks (for schur-based solvers).

#include "ceres/evaluator.h"

#include "ceres/casts.h"
#include "ceres/cost_function.h"
#include "ceres/crs_matrix.h"
#include "ceres/evaluator_test_utils.h"
#include "ceres/internal/eigen.h"
#include "ceres/internal/scoped_ptr.h"
#include "ceres/local_parameterization.h"
#include "ceres/problem_impl.h"
#include "ceres/program.h"
#include "ceres/sized_cost_function.h"
#include "ceres/sparse_matrix.h"
#include "ceres/types.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

// TODO(keir): Consider pushing this into a common test utils file.
template<int kFactor, int kNumResiduals,
         int N0 = 0, int N1 = 0, int N2 = 0, bool kSucceeds = true>
class ParameterIgnoringCostFunction
    : public SizedCostFunction<kNumResiduals, N0, N1, N2> {
  typedef SizedCostFunction<kNumResiduals, N0, N1, N2> Base;
 public:
  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    for (int i = 0; i < Base::num_residuals(); ++i) {
      residuals[i] = i + 1;
    }
    if (jacobians) {
      for (int k = 0; k < Base::parameter_block_sizes().size(); ++k) {
        // The jacobians here are full sized, but they are transformed in the
        // evaluator into the "local" jacobian. In the tests, the "subset
        // constant" parameterization is used, which should pick out columns
        // from these jacobians. Put values in the jacobian that make this
        // obvious; in particular, make the jacobians like this:
        //
        //   1 2 3 4 ...
        //   1 2 3 4 ...   .*  kFactor
        //   1 2 3 4 ...
        //
        // where the multiplication by kFactor makes it easier to distinguish
        // between Jacobians of different residuals for the same parameter.
        if (jacobians[k] != NULL) {
          MatrixRef jacobian(jacobians[k],
                             Base::num_residuals(),
                             Base::parameter_block_sizes()[k]);
          for (int j = 0; j < Base::parameter_block_sizes()[k]; ++j) {
            jacobian.col(j).setConstant(kFactor * (j + 1));
          }
        }
      }
    }
    return kSucceeds;
  }
};

struct EvaluatorTest
    : public ::testing::TestWithParam<pair<LinearSolverType, int> > {
  Evaluator* CreateEvaluator(Program* program) {
    // This program is straight from the ProblemImpl, and so has no index/offset
    // yet; compute it here as required by the evalutor implementations.
    program->SetParameterOffsetsAndIndex();

    VLOG(1) << "Creating evaluator with type: " << GetParam().first
            << " and num_eliminate_blocks: " << GetParam().second;
    Evaluator::Options options;
    options.linear_solver_type = GetParam().first;
    options.num_eliminate_blocks = GetParam().second;
    string error;
    return Evaluator::Create(options, program, &error);
  }

  void EvaluateAndCompare(ProblemImpl *problem,
                          int expected_num_rows,
                          int expected_num_cols,
                          double expected_cost,
                          const double* expected_residuals,
                          const double* expected_gradient,
                          const double* expected_jacobian) {
    scoped_ptr<Evaluator> evaluator(
        CreateEvaluator(problem->mutable_program()));
    int num_residuals = expected_num_rows;
    int num_parameters = expected_num_cols;

    double cost = -1;

    Vector residuals(num_residuals);
    residuals.setConstant(-2000);

    Vector gradient(num_parameters);
    gradient.setConstant(-3000);

    scoped_ptr<SparseMatrix> jacobian(evaluator->CreateJacobian());

    ASSERT_EQ(expected_num_rows, evaluator->NumResiduals());
    ASSERT_EQ(expected_num_cols, evaluator->NumEffectiveParameters());
    ASSERT_EQ(expected_num_rows, jacobian->num_rows());
    ASSERT_EQ(expected_num_cols, jacobian->num_cols());

    vector<double> state(evaluator->NumParameters());

    ASSERT_TRUE(evaluator->Evaluate(
          &state[0],
          &cost,
          expected_residuals != NULL ? &residuals[0]  : NULL,
          expected_gradient  != NULL ? &gradient[0]   : NULL,
          expected_jacobian  != NULL ? jacobian.get() : NULL));

    Matrix actual_jacobian;
    if (expected_jacobian != NULL) {
      jacobian->ToDenseMatrix(&actual_jacobian);
    }

    CompareEvaluations(expected_num_rows,
                       expected_num_cols,
                       expected_cost,
                       expected_residuals,
                       expected_gradient,
                       expected_jacobian,
                       cost,
                       &residuals[0],
                       &gradient[0],
                       actual_jacobian.data());
  }

  // Try all combinations of parameters for the evaluator.
  void CheckAllEvaluationCombinations(const ExpectedEvaluation &expected) {
    for (int i = 0; i < 8; ++i) {
      EvaluateAndCompare(&problem,
                         expected.num_rows,
                         expected.num_cols,
                         expected.cost,
                         (i & 1) ? expected.residuals : NULL,
                         (i & 2) ? expected.gradient  : NULL,
                         (i & 4) ? expected.jacobian  : NULL);
    }
  }

  // The values are ignored completely by the cost function.
  double x[2];
  double y[3];
  double z[4];

  ProblemImpl problem;
};

void SetSparseMatrixConstant(SparseMatrix* sparse_matrix, double value) {
  VectorRef(sparse_matrix->mutable_values(),
            sparse_matrix->num_nonzeros()).setConstant(value);
}

TEST_P(EvaluatorTest, SingleResidualProblem) {
  problem.AddResidualBlock(new ParameterIgnoringCostFunction<1, 3, 2, 3, 4>,
                           NULL,
                           x, y, z);

  ExpectedEvaluation expected = {
    // Rows/columns
    3, 9,
    // Cost
    7.0,
    // Residuals
    { 1.0, 2.0, 3.0 },
    // Gradient
    { 6.0, 12.0,              // x
      6.0, 12.0, 18.0,        // y
      6.0, 12.0, 18.0, 24.0,  // z
    },
    // Jacobian
    //   x          y             z
    { 1, 2,   1, 2, 3,   1, 2, 3, 4,
      1, 2,   1, 2, 3,   1, 2, 3, 4,
      1, 2,   1, 2, 3,   1, 2, 3, 4
    }
  };
  CheckAllEvaluationCombinations(expected);
}

TEST_P(EvaluatorTest, SingleResidualProblemWithPermutedParameters) {
  // Add the parameters in explicit order to force the ordering in the program.
  problem.AddParameterBlock(x,  2);
  problem.AddParameterBlock(y,  3);
  problem.AddParameterBlock(z,  4);

  // Then use a cost function which is similar to the others, but swap around
  // the ordering of the parameters to the cost function. This shouldn't affect
  // the jacobian evaluation, but requires explicit handling in the evaluators.
  // At one point the compressed row evaluator had a bug that went undetected
  // for a long time, since by chance most users added parameters to the problem
  // in the same order that they occured as parameters to a cost function.
  problem.AddResidualBlock(new ParameterIgnoringCostFunction<1, 3, 4, 3, 2>,
                           NULL,
                           z, y, x);

  ExpectedEvaluation expected = {
    // Rows/columns
    3, 9,
    // Cost
    7.0,
    // Residuals
    { 1.0, 2.0, 3.0 },
    // Gradient
    { 6.0, 12.0,              // x
      6.0, 12.0, 18.0,        // y
      6.0, 12.0, 18.0, 24.0,  // z
    },
    // Jacobian
    //   x          y             z
    { 1, 2,   1, 2, 3,   1, 2, 3, 4,
      1, 2,   1, 2, 3,   1, 2, 3, 4,
      1, 2,   1, 2, 3,   1, 2, 3, 4
    }
  };
  CheckAllEvaluationCombinations(expected);
}

TEST_P(EvaluatorTest, SingleResidualProblemWithNuisanceParameters) {
  // These parameters are not used.
  double a[2];
  double b[1];
  double c[1];
  double d[3];

  // Add the parameters in a mixed order so the Jacobian is "checkered" with the
  // values from the other parameters.
  problem.AddParameterBlock(a, 2);
  problem.AddParameterBlock(x, 2);
  problem.AddParameterBlock(b, 1);
  problem.AddParameterBlock(y, 3);
  problem.AddParameterBlock(c, 1);
  problem.AddParameterBlock(z, 4);
  problem.AddParameterBlock(d, 3);

  problem.AddResidualBlock(new ParameterIgnoringCostFunction<1, 3, 2, 3, 4>,
                           NULL,
                           x, y, z);

  ExpectedEvaluation expected = {
    // Rows/columns
    3, 16,
    // Cost
    7.0,
    // Residuals
    { 1.0, 2.0, 3.0 },
    // Gradient
    { 0.0, 0.0,               // a
      6.0, 12.0,              // x
      0.0,                    // b
      6.0, 12.0, 18.0,        // y
      0.0,                    // c
      6.0, 12.0, 18.0, 24.0,  // z
      0.0, 0.0, 0.0,          // d
    },
    // Jacobian
    //   a        x     b           y     c              z           d
    { 0, 0,    1, 2,    0,    1, 2, 3,    0,    1, 2, 3, 4,    0, 0, 0,
      0, 0,    1, 2,    0,    1, 2, 3,    0,    1, 2, 3, 4,    0, 0, 0,
      0, 0,    1, 2,    0,    1, 2, 3,    0,    1, 2, 3, 4,    0, 0, 0
    }
  };
  CheckAllEvaluationCombinations(expected);
}

TEST_P(EvaluatorTest, MultipleResidualProblem) {
  // Add the parameters in explicit order to force the ordering in the program.
  problem.AddParameterBlock(x,  2);
  problem.AddParameterBlock(y,  3);
  problem.AddParameterBlock(z,  4);

  // f(x, y) in R^2
  problem.AddResidualBlock(new ParameterIgnoringCostFunction<1, 2, 2, 3>,
                           NULL,
                           x, y);

  // g(x, z) in R^3
  problem.AddResidualBlock(new ParameterIgnoringCostFunction<2, 3, 2, 4>,
                           NULL,
                           x, z);

  // h(y, z) in R^4
  problem.AddResidualBlock(new ParameterIgnoringCostFunction<3, 4, 3, 4>,
                           NULL,
                           y, z);

  ExpectedEvaluation expected = {
    // Rows/columns
    9, 9,
    // Cost
    // f       g           h
    (  1 + 4 + 1 + 4 + 9 + 1 + 4 + 9 + 16) / 2.0,
    // Residuals
    { 1.0, 2.0,           // f
      1.0, 2.0, 3.0,      // g
      1.0, 2.0, 3.0, 4.0  // h
    },
    // Gradient
    { 15.0, 30.0,               // x
      33.0, 66.0, 99.0,         // y
      42.0, 84.0, 126.0, 168.0  // z
    },
    // Jacobian
    //                x        y           z
    {   /* f(x, y) */ 1, 2,    1, 2, 3,    0, 0, 0, 0,
                      1, 2,    1, 2, 3,    0, 0, 0, 0,

        /* g(x, z) */ 2, 4,    0, 0, 0,    2, 4, 6, 8,
                      2, 4,    0, 0, 0,    2, 4, 6, 8,
                      2, 4,    0, 0, 0,    2, 4, 6, 8,

        /* h(y, z) */ 0, 0,    3, 6, 9,    3, 6, 9, 12,
                      0, 0,    3, 6, 9,    3, 6, 9, 12,
                      0, 0,    3, 6, 9,    3, 6, 9, 12,
                      0, 0,    3, 6, 9,    3, 6, 9, 12
    }
  };
  CheckAllEvaluationCombinations(expected);
}

TEST_P(EvaluatorTest, MultipleResidualsWithLocalParameterizations) {
  // Add the parameters in explicit order to force the ordering in the program.
  problem.AddParameterBlock(x,  2);

  // Fix y's first dimension.
  vector<int> y_fixed;
  y_fixed.push_back(0);
  problem.AddParameterBlock(y, 3, new SubsetParameterization(3, y_fixed));

  // Fix z's second dimension.
  vector<int> z_fixed;
  z_fixed.push_back(1);
  problem.AddParameterBlock(z, 4, new SubsetParameterization(4, z_fixed));

  // f(x, y) in R^2
  problem.AddResidualBlock(new ParameterIgnoringCostFunction<1, 2, 2, 3>,
                           NULL,
                           x, y);

  // g(x, z) in R^3
  problem.AddResidualBlock(new ParameterIgnoringCostFunction<2, 3, 2, 4>,
                           NULL,
                           x, z);

  // h(y, z) in R^4
  problem.AddResidualBlock(new ParameterIgnoringCostFunction<3, 4, 3, 4>,
                           NULL,
                           y, z);

  ExpectedEvaluation expected = {
    // Rows/columns
    9, 7,
    // Cost
    // f       g           h
    (  1 + 4 + 1 + 4 + 9 + 1 + 4 + 9 + 16) / 2.0,
    // Residuals
    { 1.0, 2.0,           // f
      1.0, 2.0, 3.0,      // g
      1.0, 2.0, 3.0, 4.0  // h
    },
    // Gradient
    { 15.0, 30.0,         // x
      66.0, 99.0,         // y
      42.0, 126.0, 168.0  // z
    },
    // Jacobian
    //                x        y           z
    {   /* f(x, y) */ 1, 2,    2, 3,    0, 0, 0,
                      1, 2,    2, 3,    0, 0, 0,

        /* g(x, z) */ 2, 4,    0, 0,    2, 6, 8,
                      2, 4,    0, 0,    2, 6, 8,
                      2, 4,    0, 0,    2, 6, 8,

        /* h(y, z) */ 0, 0,    6, 9,    3, 9, 12,
                      0, 0,    6, 9,    3, 9, 12,
                      0, 0,    6, 9,    3, 9, 12,
                      0, 0,    6, 9,    3, 9, 12
    }
  };
  CheckAllEvaluationCombinations(expected);
}

TEST_P(EvaluatorTest, MultipleResidualProblemWithSomeConstantParameters) {
  // The values are ignored completely by the cost function.
  double x[2];
  double y[3];
  double z[4];

  // Add the parameters in explicit order to force the ordering in the program.
  problem.AddParameterBlock(x,  2);
  problem.AddParameterBlock(y,  3);
  problem.AddParameterBlock(z,  4);

  // f(x, y) in R^2
  problem.AddResidualBlock(new ParameterIgnoringCostFunction<1, 2, 2, 3>,
                           NULL,
                           x, y);

  // g(x, z) in R^3
  problem.AddResidualBlock(new ParameterIgnoringCostFunction<2, 3, 2, 4>,
                           NULL,
                           x, z);

  // h(y, z) in R^4
  problem.AddResidualBlock(new ParameterIgnoringCostFunction<3, 4, 3, 4>,
                           NULL,
                           y, z);

  // For this test, "z" is constant.
  problem.SetParameterBlockConstant(z);

  // Create the reduced program which is missing the fixed "z" variable.
  // Normally, the preprocessing of the program that happens in solver_impl
  // takes care of this, but we don't want to invoke the solver here.
  Program reduced_program;
  vector<ParameterBlock*>* parameter_blocks =
      problem.mutable_program()->mutable_parameter_blocks();

  // "z" is the last parameter; save it for later and pop it off temporarily.
  // Note that "z" will still get read during evaluation, so it cannot be
  // deleted at this point.
  ParameterBlock* parameter_block_z = parameter_blocks->back();
  parameter_blocks->pop_back();

  ExpectedEvaluation expected = {
    // Rows/columns
    9, 5,
    // Cost
    // f       g           h
    (  1 + 4 + 1 + 4 + 9 + 1 + 4 + 9 + 16) / 2.0,
    // Residuals
    { 1.0, 2.0,           // f
      1.0, 2.0, 3.0,      // g
      1.0, 2.0, 3.0, 4.0  // h
    },
    // Gradient
    { 15.0, 30.0,        // x
      33.0, 66.0, 99.0,  // y
    },
    // Jacobian
    //                x        y
    {   /* f(x, y) */ 1, 2,    1, 2, 3,
                      1, 2,    1, 2, 3,

        /* g(x, z) */ 2, 4,    0, 0, 0,
                      2, 4,    0, 0, 0,
                      2, 4,    0, 0, 0,

        /* h(y, z) */ 0, 0,    3, 6, 9,
                      0, 0,    3, 6, 9,
                      0, 0,    3, 6, 9,
                      0, 0,    3, 6, 9
    }
  };
  CheckAllEvaluationCombinations(expected);

  // Restore parameter block z, so it will get freed in a consistent way.
  parameter_blocks->push_back(parameter_block_z);
}

TEST_P(EvaluatorTest, EvaluatorAbortsForResidualsThatFailToEvaluate) {
  // Switch the return value to failure.
  problem.AddResidualBlock(
      new ParameterIgnoringCostFunction<20, 3, 2, 3, 4, false>, NULL, x, y, z);

  // The values are ignored.
  double state[9];

  scoped_ptr<Evaluator> evaluator(CreateEvaluator(problem.mutable_program()));
  scoped_ptr<SparseMatrix> jacobian(evaluator->CreateJacobian());
  double cost;
  EXPECT_FALSE(evaluator->Evaluate(state, &cost, NULL, NULL, NULL));
}

// In the pairs, the first argument is the linear solver type, and the second
// argument is num_eliminate_blocks. Changing the num_eliminate_blocks only
// makes sense for the schur-based solvers.
//
// Try all values of num_eliminate_blocks that make sense given that in the
// tests a maximum of 4 parameter blocks are present.
INSTANTIATE_TEST_CASE_P(
    LinearSolvers,
    EvaluatorTest,
    ::testing::Values(make_pair(DENSE_QR, 0),
                      make_pair(DENSE_SCHUR, 0),
                      make_pair(DENSE_SCHUR, 1),
                      make_pair(DENSE_SCHUR, 2),
                      make_pair(DENSE_SCHUR, 3),
                      make_pair(DENSE_SCHUR, 4),
                      make_pair(SPARSE_SCHUR, 0),
                      make_pair(SPARSE_SCHUR, 1),
                      make_pair(SPARSE_SCHUR, 2),
                      make_pair(SPARSE_SCHUR, 3),
                      make_pair(SPARSE_SCHUR, 4),
                      make_pair(ITERATIVE_SCHUR, 0),
                      make_pair(ITERATIVE_SCHUR, 1),
                      make_pair(ITERATIVE_SCHUR, 2),
                      make_pair(ITERATIVE_SCHUR, 3),
                      make_pair(ITERATIVE_SCHUR, 4),
                      make_pair(SPARSE_NORMAL_CHOLESKY, 0)));

// Simple cost function used to check if the evaluator is sensitive to
// state changes.
class ParameterSensitiveCostFunction : public SizedCostFunction<2, 2> {
 public:
  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    double x1 = parameters[0][0];
    double x2 = parameters[0][1];
    residuals[0] = x1 * x1;
    residuals[1] = x2 * x2;

    if (jacobians != NULL) {
      double* jacobian = jacobians[0];
      if (jacobian != NULL) {
        jacobian[0] = 2.0 * x1;
        jacobian[1] = 0.0;
        jacobian[2] = 0.0;
        jacobian[3] = 2.0 * x2;
      }
    }
    return true;
  }
};

TEST(Evaluator, EvaluatorRespectsParameterChanges) {
  ProblemImpl problem;

  double x[2];
  x[0] = 1.0;
  x[1] = 1.0;

  problem.AddResidualBlock(new ParameterSensitiveCostFunction(), NULL, x);
  Program* program = problem.mutable_program();
  program->SetParameterOffsetsAndIndex();

  Evaluator::Options options;
  options.linear_solver_type = DENSE_QR;
  options.num_eliminate_blocks = 0;
  string error;
  scoped_ptr<Evaluator> evaluator(Evaluator::Create(options, program, &error));
  scoped_ptr<SparseMatrix> jacobian(evaluator->CreateJacobian());

  ASSERT_EQ(2, jacobian->num_rows());
  ASSERT_EQ(2, jacobian->num_cols());

  double state[2];
  state[0] = 2.0;
  state[1] = 3.0;

  // The original state of a residual block comes from the user's
  // state. So the original state is 1.0, 1.0, and the only way we get
  // the 2.0, 3.0 results in the following tests is if it respects the
  // values in the state vector.

  // Cost only; no residuals and no jacobian.
  {
    double cost = -1;
    ASSERT_TRUE(evaluator->Evaluate(state, &cost, NULL, NULL, NULL));
    EXPECT_EQ(48.5, cost);
  }

  // Cost and residuals, no jacobian.
  {
    double cost = -1;
    double residuals[2] = { -2, -2 };
    ASSERT_TRUE(evaluator->Evaluate(state, &cost, residuals, NULL, NULL));
    EXPECT_EQ(48.5, cost);
    EXPECT_EQ(4, residuals[0]);
    EXPECT_EQ(9, residuals[1]);
  }

  // Cost, residuals, and jacobian.
  {
    double cost = -1;
    double residuals[2] = { -2, -2};
    SetSparseMatrixConstant(jacobian.get(), -1);
    ASSERT_TRUE(evaluator->Evaluate(state,
                                    &cost,
                                    residuals,
                                    NULL,
                                    jacobian.get()));
    EXPECT_EQ(48.5, cost);
    EXPECT_EQ(4, residuals[0]);
    EXPECT_EQ(9, residuals[1]);
    Matrix actual_jacobian;
    jacobian->ToDenseMatrix(&actual_jacobian);

    Matrix expected_jacobian(2, 2);
    expected_jacobian
        << 2 * state[0], 0,
           0, 2 * state[1];

    EXPECT_TRUE((actual_jacobian.array() == expected_jacobian.array()).all())
        << "Actual:\n" << actual_jacobian
        << "\nExpected:\n" << expected_jacobian;
  }
}

}  // namespace internal
}  // namespace ceres
