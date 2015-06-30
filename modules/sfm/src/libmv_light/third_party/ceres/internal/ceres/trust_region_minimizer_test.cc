// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2012 Google Inc. All rights reserved.
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
//         sameeragarwal@google.com (Sameer Agarwal)
//
// This tests the TrustRegionMinimizer loop using a direct Evaluator
// implementation, rather than having a test that goes through all the
// Program and Problem machinery.

#include <cmath>
#include "ceres/cost_function.h"
#include "ceres/dense_qr_solver.h"
#include "ceres/dense_sparse_matrix.h"
#include "ceres/evaluator.h"
#include "ceres/internal/port.h"
#include "ceres/linear_solver.h"
#include "ceres/minimizer.h"
#include "ceres/problem.h"
#include "ceres/trust_region_minimizer.h"
#include "ceres/trust_region_strategy.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

// Templated Evaluator for Powell's function. The template parameters
// indicate which of the four variables/columns of the jacobian are
// active. This is equivalent to constructing a problem and using the
// SubsetLocalParameterization. This allows us to test the support for
// the Evaluator::Plus operation besides checking for the basic
// performance of the trust region algorithm.
template <bool col1, bool col2, bool col3, bool col4>
class PowellEvaluator2 : public Evaluator {
 public:
  PowellEvaluator2()
      : num_active_cols_(
          (col1 ? 1 : 0) +
          (col2 ? 1 : 0) +
          (col3 ? 1 : 0) +
          (col4 ? 1 : 0)) {
    VLOG(1) << "Columns: "
            << col1 << " "
            << col2 << " "
            << col3 << " "
            << col4;
  }

  virtual ~PowellEvaluator2() {}

  // Implementation of Evaluator interface.
  virtual SparseMatrix* CreateJacobian() const {
    CHECK(col1 || col2 || col3 || col4);
    DenseSparseMatrix* dense_jacobian =
        new DenseSparseMatrix(NumResiduals(), NumEffectiveParameters());
    dense_jacobian->SetZero();
    return dense_jacobian;
  }

  virtual bool Evaluate(const Evaluator::EvaluateOptions& evaluate_options,
                        const double* state,
                        double* cost,
                        double* residuals,
                        double* /* gradient */,
                        SparseMatrix* jacobian) {
    double x1 = state[0];
    double x2 = state[1];
    double x3 = state[2];
    double x4 = state[3];

    VLOG(1) << "State: "
            << "x1=" << x1 << ", "
            << "x2=" << x2 << ", "
            << "x3=" << x3 << ", "
            << "x4=" << x4 << ".";

    double f1 = x1 + 10.0 * x2;
    double f2 = sqrt(5.0) * (x3 - x4);
    double f3 = pow(x2 - 2.0 * x3, 2.0);
    double f4 = sqrt(10.0) * pow(x1 - x4, 2.0);

    VLOG(1) << "Function: "
            << "f1=" << f1 << ", "
            << "f2=" << f2 << ", "
            << "f3=" << f3 << ", "
            << "f4=" << f4 << ".";

    *cost = (f1*f1 + f2*f2 + f3*f3 + f4*f4) / 2.0;

    VLOG(1) << "Cost: " << *cost;

    if (residuals != NULL) {
      residuals[0] = f1;
      residuals[1] = f2;
      residuals[2] = f3;
      residuals[3] = f4;
    }

    if (jacobian != NULL) {
      DenseSparseMatrix* dense_jacobian;
      dense_jacobian = down_cast<DenseSparseMatrix*>(jacobian);
      dense_jacobian->SetZero();

      ColMajorMatrixRef jacobian_matrix = dense_jacobian->mutable_matrix();
      CHECK_EQ(jacobian_matrix.cols(), num_active_cols_);

      int column_index = 0;
      if (col1) {
        jacobian_matrix.col(column_index++) <<
            1.0,
            0.0,
            0.0,
            sqrt(10.0) * 2.0 * (x1 - x4) * (1.0 - x4);
      }
      if (col2) {
        jacobian_matrix.col(column_index++) <<
            10.0,
            0.0,
            2.0*(x2 - 2.0*x3)*(1.0 - 2.0*x3),
            0.0;
      }

      if (col3) {
        jacobian_matrix.col(column_index++) <<
            0.0,
            sqrt(5.0),
            2.0*(x2 - 2.0*x3)*(x2 - 2.0),
            0.0;
      }

      if (col4) {
        jacobian_matrix.col(column_index++) <<
            0.0,
            -sqrt(5.0),
            0.0,
            sqrt(10.0) * 2.0 * (x1 - x4) * (x1 - 1.0);
      }
      VLOG(1) << "\n" << jacobian_matrix;
    }
    return true;
  }

  virtual bool Plus(const double* state,
                    const double* delta,
                    double* state_plus_delta) const {
    int delta_index = 0;
    state_plus_delta[0] = (col1  ? state[0] + delta[delta_index++] : state[0]);
    state_plus_delta[1] = (col2  ? state[1] + delta[delta_index++] : state[1]);
    state_plus_delta[2] = (col3  ? state[2] + delta[delta_index++] : state[2]);
    state_plus_delta[3] = (col4  ? state[3] + delta[delta_index++] : state[3]);
    return true;
  }

  virtual int NumEffectiveParameters() const { return num_active_cols_; }
  virtual int NumParameters()          const { return 4; }
  virtual int NumResiduals()           const { return 4; }

 private:
  const int num_active_cols_;
};

// Templated function to hold a subset of the columns fixed and check
// if the solver converges to the optimal values or not.
template<bool col1, bool col2, bool col3, bool col4>
void IsTrustRegionSolveSuccessful(TrustRegionStrategyType strategy_type) {
  Solver::Options solver_options;
  LinearSolver::Options linear_solver_options;
  DenseQRSolver linear_solver(linear_solver_options);

  double parameters[4] = { 3, -1, 0, 1.0 };

  // If the column is inactive, then set its value to the optimal
  // value.
  parameters[0] = (col1 ? parameters[0] : 0.0);
  parameters[1] = (col2 ? parameters[1] : 0.0);
  parameters[2] = (col3 ? parameters[2] : 0.0);
  parameters[3] = (col4 ? parameters[3] : 0.0);

  PowellEvaluator2<col1, col2, col3, col4> powell_evaluator;
  scoped_ptr<SparseMatrix> jacobian(powell_evaluator.CreateJacobian());

  Minimizer::Options minimizer_options(solver_options);
  minimizer_options.gradient_tolerance = 1e-26;
  minimizer_options.function_tolerance = 1e-26;
  minimizer_options.parameter_tolerance = 1e-26;
  minimizer_options.evaluator = &powell_evaluator;
  minimizer_options.jacobian = jacobian.get();

  TrustRegionStrategy::Options trust_region_strategy_options;
  trust_region_strategy_options.trust_region_strategy_type = strategy_type;
  trust_region_strategy_options.linear_solver = &linear_solver;
  trust_region_strategy_options.initial_radius = 1e4;
  trust_region_strategy_options.max_radius = 1e20;
  trust_region_strategy_options.lm_min_diagonal = 1e-6;
  trust_region_strategy_options.lm_max_diagonal = 1e32;
  scoped_ptr<TrustRegionStrategy> strategy(
      TrustRegionStrategy::Create(trust_region_strategy_options));
  minimizer_options.trust_region_strategy = strategy.get();

  TrustRegionMinimizer minimizer;
  Solver::Summary summary;
  minimizer.Minimize(minimizer_options, parameters, &summary);

  // The minimum is at x1 = x2 = x3 = x4 = 0.
  EXPECT_NEAR(0.0, parameters[0], 0.001);
  EXPECT_NEAR(0.0, parameters[1], 0.001);
  EXPECT_NEAR(0.0, parameters[2], 0.001);
  EXPECT_NEAR(0.0, parameters[3], 0.001);
};

TEST(TrustRegionMinimizer, PowellsSingularFunctionUsingLevenbergMarquardt) {
  // This case is excluded because this has a local minimum and does
  // not find the optimum. This should not affect the correctness of
  // this test since we are testing all the other 14 combinations of
  // column activations.
  //
  //   IsSolveSuccessful<true, true, false, true>();

  const TrustRegionStrategyType kStrategy = LEVENBERG_MARQUARDT;
  IsTrustRegionSolveSuccessful<true,  true,  true,  true >(kStrategy);
  IsTrustRegionSolveSuccessful<true,  true,  true,  false>(kStrategy);
  IsTrustRegionSolveSuccessful<true,  false, true,  true >(kStrategy);
  IsTrustRegionSolveSuccessful<false, true,  true,  true >(kStrategy);
  IsTrustRegionSolveSuccessful<true,  true,  false, false>(kStrategy);
  IsTrustRegionSolveSuccessful<true,  false, true,  false>(kStrategy);
  IsTrustRegionSolveSuccessful<false, true,  true,  false>(kStrategy);
  IsTrustRegionSolveSuccessful<true,  false, false, true >(kStrategy);
  IsTrustRegionSolveSuccessful<false, true,  false, true >(kStrategy);
  IsTrustRegionSolveSuccessful<false, false, true,  true >(kStrategy);
  IsTrustRegionSolveSuccessful<true,  false, false, false>(kStrategy);
  IsTrustRegionSolveSuccessful<false, true,  false, false>(kStrategy);
  IsTrustRegionSolveSuccessful<false, false, true,  false>(kStrategy);
  IsTrustRegionSolveSuccessful<false, false, false, true >(kStrategy);
}

TEST(TrustRegionMinimizer, PowellsSingularFunctionUsingDogleg) {
  // The following two cases are excluded because they encounter a
  // local minimum.
  //
  //  IsTrustRegionSolveSuccessful<true, true, false, true >(kStrategy);
  //  IsTrustRegionSolveSuccessful<true,  true,  true,  true >(kStrategy);

  const TrustRegionStrategyType kStrategy = DOGLEG;
  IsTrustRegionSolveSuccessful<true,  true,  true,  false>(kStrategy);
  IsTrustRegionSolveSuccessful<true,  false, true,  true >(kStrategy);
  IsTrustRegionSolveSuccessful<false, true,  true,  true >(kStrategy);
  IsTrustRegionSolveSuccessful<true,  true,  false, false>(kStrategy);
  IsTrustRegionSolveSuccessful<true,  false, true,  false>(kStrategy);
  IsTrustRegionSolveSuccessful<false, true,  true,  false>(kStrategy);
  IsTrustRegionSolveSuccessful<true,  false, false, true >(kStrategy);
  IsTrustRegionSolveSuccessful<false, true,  false, true >(kStrategy);
  IsTrustRegionSolveSuccessful<false, false, true,  true >(kStrategy);
  IsTrustRegionSolveSuccessful<true,  false, false, false>(kStrategy);
  IsTrustRegionSolveSuccessful<false, true,  false, false>(kStrategy);
  IsTrustRegionSolveSuccessful<false, false, true,  false>(kStrategy);
  IsTrustRegionSolveSuccessful<false, false, false, true >(kStrategy);
}


class CurveCostFunction : public CostFunction {
 public:
  CurveCostFunction(int num_vertices, double target_length)
      : num_vertices_(num_vertices), target_length_(target_length) {
    set_num_residuals(1);
    for (int i = 0; i < num_vertices_; ++i) {
      mutable_parameter_block_sizes()->push_back(2);
    }
  }

  bool Evaluate(double const* const* parameters,
                double* residuals,
                double** jacobians) const {
    residuals[0] = target_length_;

    for (int i = 0; i < num_vertices_; ++i) {
      int prev = (num_vertices_ + i - 1) % num_vertices_;
      double length = 0.0;
      for (int dim = 0; dim < 2; dim++) {
        const double diff = parameters[prev][dim] - parameters[i][dim];
        length += diff * diff;
      }
      residuals[0] -= sqrt(length);
    }

    if (jacobians == NULL) {
      return true;
    }

    for (int i = 0; i < num_vertices_; ++i) {
      if (jacobians[i] != NULL) {
        int prev = (num_vertices_ + i - 1) % num_vertices_;
        int next = (i + 1) % num_vertices_;

        double u[2], v[2];
        double norm_u = 0., norm_v = 0.;
        for (int dim = 0; dim < 2; dim++) {
          u[dim] = parameters[i][dim] - parameters[prev][dim];
          norm_u += u[dim] * u[dim];
          v[dim] = parameters[next][dim] - parameters[i][dim];
          norm_v += v[dim] * v[dim];
        }

        norm_u = sqrt(norm_u);
        norm_v = sqrt(norm_v);

        for (int dim = 0; dim < 2; dim++) {
          jacobians[i][dim] = 0.;

          if (norm_u > std::numeric_limits< double >::min()) {
            jacobians[i][dim] -= u[dim] / norm_u;
          }

          if (norm_v > std::numeric_limits< double >::min()) {
            jacobians[i][dim] += v[dim] / norm_v;
          }
        }
      }
    }

    return true;
  }

 private:
  int     num_vertices_;
  double  target_length_;
};

TEST(TrustRegionMinimizer, JacobiScalingTest) {
  int N = 6;
  std::vector< double* > y(N);
  const double pi = 3.1415926535897932384626433;
  for (int i = 0; i < N; i++) {
    double theta = i * 2. * pi/ static_cast< double >(N);
    y[i] = new double[2];
    y[i][0] = cos(theta);
    y[i][1] = sin(theta);
  }

  Problem problem;
  problem.AddResidualBlock(new CurveCostFunction(N, 10.), NULL, y);
  Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  Solver::Summary summary;
  Solve(options, &problem, &summary);
  EXPECT_LE(summary.final_cost, 1e-10);

  for (int i = 0; i < N; i++) {
    delete []y[i];
  }
}

}  // namespace internal
}  // namespace ceres
