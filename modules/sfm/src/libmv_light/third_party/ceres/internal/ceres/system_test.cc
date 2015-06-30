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
//         sameeragarwal@google.com (Sameer Agarwal)
//
// System level tests for Ceres. The current suite of two tests. The
// first test is a small test based on Powell's Function. It is a
// scalar problem with 4 variables. The second problem is a bundle
// adjustment problem with 16 cameras and two thousand cameras. The
// first problem is to test the sanity test the factorization based
// solvers. The second problem is used to test the various
// combinations of solvers, orderings, preconditioners and
// multithreading.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "ceres/autodiff_cost_function.h"
#include "ceres/ordered_groups.h"
#include "ceres/problem.h"
#include "ceres/rotation.h"
#include "ceres/solver.h"
#include "ceres/stringprintf.h"
#include "ceres/test_util.h"
#include "ceres/types.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

const bool kAutomaticOrdering = true;
const bool kUserOrdering = false;

// Struct used for configuring the solver.
struct SolverConfig {
  SolverConfig(LinearSolverType linear_solver_type,
               SparseLinearAlgebraLibraryType sparse_linear_algebra_library,
               bool use_automatic_ordering)
      : linear_solver_type(linear_solver_type),
        sparse_linear_algebra_library(sparse_linear_algebra_library),
        use_automatic_ordering(use_automatic_ordering),
        preconditioner_type(IDENTITY),
        num_threads(1) {
  }

  SolverConfig(LinearSolverType linear_solver_type,
               SparseLinearAlgebraLibraryType sparse_linear_algebra_library,
               bool use_automatic_ordering,
               PreconditionerType preconditioner_type)
      : linear_solver_type(linear_solver_type),
        sparse_linear_algebra_library(sparse_linear_algebra_library),
        use_automatic_ordering(use_automatic_ordering),
        preconditioner_type(preconditioner_type),
        num_threads(1) {
  }

  string ToString() const {
    return StringPrintf(
        "(%s, %s, %s, %s, %d)",
        LinearSolverTypeToString(linear_solver_type),
        SparseLinearAlgebraLibraryTypeToString(sparse_linear_algebra_library),
        use_automatic_ordering ? "AUTOMATIC" : "USER",
        PreconditionerTypeToString(preconditioner_type),
        num_threads);
  }

  LinearSolverType linear_solver_type;
  SparseLinearAlgebraLibraryType sparse_linear_algebra_library;
  bool use_automatic_ordering;
  PreconditionerType preconditioner_type;
  int num_threads;
};

// Templated function that given a set of solver configurations,
// instantiates a new copy of SystemTestProblem for each configuration
// and solves it. The solutions are expected to have residuals with
// coordinate-wise maximum absolute difference less than or equal to
// max_abs_difference.
//
// The template parameter SystemTestProblem is expected to implement
// the following interface.
//
//   class SystemTestProblem {
//     public:
//       SystemTestProblem();
//       Problem* mutable_problem();
//       Solver::Options* mutable_solver_options();
//   };
template <typename SystemTestProblem>
void RunSolversAndCheckTheyMatch(const vector<SolverConfig>& configurations,
                                 const double max_abs_difference) {
  int num_configurations = configurations.size();
  vector<SystemTestProblem*> problems;
  vector<vector<double> > final_residuals(num_configurations);

  for (int i = 0; i < num_configurations; ++i) {
    SystemTestProblem* system_test_problem = new SystemTestProblem();

    const SolverConfig& config = configurations[i];

    Solver::Options& options = *(system_test_problem->mutable_solver_options());
    options.linear_solver_type = config.linear_solver_type;
    options.sparse_linear_algebra_library =
        config.sparse_linear_algebra_library;
    options.preconditioner_type = config.preconditioner_type;
    options.num_threads = config.num_threads;
    options.num_linear_solver_threads = config.num_threads;

    if (config.use_automatic_ordering) {
      delete options.linear_solver_ordering;
      options.linear_solver_ordering = NULL;
    }

    LOG(INFO) << "Running solver configuration: "
              << config.ToString();

    Solver::Summary summary;
    Solve(options,
          system_test_problem->mutable_problem(),
          &summary);

    system_test_problem
        ->mutable_problem()
        ->Evaluate(Problem::EvaluateOptions(),
                   NULL,
                   &final_residuals[i],
                   NULL,
                   NULL);

    CHECK_NE(summary.termination_type, ceres::NUMERICAL_FAILURE)
        << "Solver configuration " << i << " failed.";
    problems.push_back(system_test_problem);

    // Compare the resulting solutions to each other. Arbitrarily take
    // SPARSE_NORMAL_CHOLESKY as the golden solve. We compare
    // solutions by comparing their residual vectors. We do not
    // compare parameter vectors because it is much more brittle and
    // error prone to do so, since the same problem can have nearly
    // the same residuals at two completely different positions in
    // parameter space.
    if (i > 0) {
      const vector<double>& reference_residuals = final_residuals[0];
      const vector<double>& current_residuals = final_residuals[i];

      for (int j = 0; j < reference_residuals.size(); ++j) {
        EXPECT_NEAR(current_residuals[j],
                    reference_residuals[j],
                    max_abs_difference)
            << "Not close enough residual:" << j
            << " reference " << reference_residuals[j]
            << " current " << current_residuals[j];
      }
    }
  }

  for (int i = 0; i < num_configurations; ++i) {
    delete problems[i];
  }
}

// This class implements the SystemTestProblem interface and provides
// access to an implementation of Powell's singular function.
//
//   F = 1/2 (f1^2 + f2^2 + f3^2 + f4^2)
//
//   f1 = x1 + 10*x2;
//   f2 = sqrt(5) * (x3 - x4)
//   f3 = (x2 - 2*x3)^2
//   f4 = sqrt(10) * (x1 - x4)^2
//
// The starting values are x1 = 3, x2 = -1, x3 = 0, x4 = 1.
// The minimum is 0 at (x1, x2, x3, x4) = 0.
//
// From: Testing Unconstrained Optimization Software by Jorge J. More, Burton S.
// Garbow and Kenneth E. Hillstrom in ACM Transactions on Mathematical Software,
// Vol 7(1), March 1981.
class PowellsFunction {
 public:
  PowellsFunction() {
    x_[0] =  3.0;
    x_[1] = -1.0;
    x_[2] =  0.0;
    x_[3] =  1.0;

    problem_.AddResidualBlock(
        new AutoDiffCostFunction<F1, 1, 1, 1>(new F1), NULL, &x_[0], &x_[1]);
    problem_.AddResidualBlock(
        new AutoDiffCostFunction<F2, 1, 1, 1>(new F2), NULL, &x_[2], &x_[3]);
    problem_.AddResidualBlock(
        new AutoDiffCostFunction<F3, 1, 1, 1>(new F3), NULL, &x_[1], &x_[2]);
    problem_.AddResidualBlock(
        new AutoDiffCostFunction<F4, 1, 1, 1>(new F4), NULL, &x_[0], &x_[3]);

    options_.max_num_iterations = 10;
  }

  Problem* mutable_problem() { return &problem_; }
  Solver::Options* mutable_solver_options() { return &options_; }

 private:
  // Templated functions used for automatically differentiated cost
  // functions.
  class F1 {
   public:
    template <typename T> bool operator()(const T* const x1,
                                          const T* const x2,
                                          T* residual) const {
      // f1 = x1 + 10 * x2;
      *residual = *x1 + T(10.0) * *x2;
      return true;
    }
  };

  class F2 {
   public:
    template <typename T> bool operator()(const T* const x3,
                                          const T* const x4,
                                          T* residual) const {
      // f2 = sqrt(5) (x3 - x4)
      *residual = T(sqrt(5.0)) * (*x3 - *x4);
      return true;
    }
  };

  class F3 {
   public:
    template <typename T> bool operator()(const T* const x2,
                                          const T* const x4,
                                          T* residual) const {
      // f3 = (x2 - 2 x3)^2
      residual[0] = (x2[0] - T(2.0) * x4[0]) * (x2[0] - T(2.0) * x4[0]);
      return true;
    }
  };

  class F4 {
   public:
    template <typename T> bool operator()(const T* const x1,
                                          const T* const x4,
                                          T* residual) const {
      // f4 = sqrt(10) (x1 - x4)^2
      residual[0] = T(sqrt(10.0)) * (x1[0] - x4[0]) * (x1[0] - x4[0]);
      return true;
    }
  };

  double x_[4];
  Problem problem_;
  Solver::Options options_;
};

TEST(SystemTest, PowellsFunction) {
  vector<SolverConfig> configs;
#define CONFIGURE(linear_solver, sparse_linear_algebra_library, ordering) \
  configs.push_back(SolverConfig(linear_solver,                           \
                                 sparse_linear_algebra_library,           \
                                 ordering))

  CONFIGURE(DENSE_QR,               SUITE_SPARSE, kAutomaticOrdering);
  CONFIGURE(DENSE_NORMAL_CHOLESKY,  SUITE_SPARSE, kAutomaticOrdering);
  CONFIGURE(DENSE_SCHUR,            SUITE_SPARSE, kAutomaticOrdering);

#ifndef CERES_NO_SUITESPARSE
  CONFIGURE(SPARSE_NORMAL_CHOLESKY, SUITE_SPARSE, kAutomaticOrdering);
#endif  // CERES_NO_SUITESPARSE

#ifndef CERES_NO_CXSPARSE
  CONFIGURE(SPARSE_NORMAL_CHOLESKY, CX_SPARSE,    kAutomaticOrdering);
#endif  // CERES_NO_CXSPARSE

  CONFIGURE(ITERATIVE_SCHUR,        SUITE_SPARSE, kAutomaticOrdering);

#undef CONFIGURE

  const double kMaxAbsoluteDifference = 1e-8;
  RunSolversAndCheckTheyMatch<PowellsFunction>(configs, kMaxAbsoluteDifference);
}

// This class implements the SystemTestProblem interface and provides
// access to a bundle adjustment problem. It is based on
// examples/bundle_adjustment_example.cc. Currently a small 16 camera
// problem is hard coded in the constructor. Going forward we may
// extend this to a larger number of problems.
class BundleAdjustmentProblem {
 public:
  BundleAdjustmentProblem() {
    const string input_file = TestFileAbsolutePath("problem-16-22106-pre.txt");
    ReadData(input_file);
    BuildProblem();
  }

  ~BundleAdjustmentProblem() {
    delete []point_index_;
    delete []camera_index_;
    delete []observations_;
    delete []parameters_;
  }

  Problem* mutable_problem() { return &problem_; }
  Solver::Options* mutable_solver_options() { return &options_; }

  int num_cameras()            const { return num_cameras_;        }
  int num_points()             const { return num_points_;         }
  int num_observations()       const { return num_observations_;   }
  const int* point_index()     const { return point_index_;  }
  const int* camera_index()    const { return camera_index_; }
  const double* observations() const { return observations_; }
  double* mutable_cameras() { return parameters_; }
  double* mutable_points() { return parameters_  + 9 * num_cameras_; }

 private:
  void ReadData(const string& filename) {
    FILE * fptr = fopen(filename.c_str(), "r");

    if (!fptr) {
      LOG(FATAL) << "File Error: unable to open file " << filename;
    };

    // This will die horribly on invalid files. Them's the breaks.
    FscanfOrDie(fptr, "%d", &num_cameras_);
    FscanfOrDie(fptr, "%d", &num_points_);
    FscanfOrDie(fptr, "%d", &num_observations_);

    VLOG(1) << "Header: " << num_cameras_
            << " " << num_points_
            << " " << num_observations_;

    point_index_ = new int[num_observations_];
    camera_index_ = new int[num_observations_];
    observations_ = new double[2 * num_observations_];

    num_parameters_ = 9 * num_cameras_ + 3 * num_points_;
    parameters_ = new double[num_parameters_];

    for (int i = 0; i < num_observations_; ++i) {
      FscanfOrDie(fptr, "%d", camera_index_ + i);
      FscanfOrDie(fptr, "%d", point_index_ + i);
      for (int j = 0; j < 2; ++j) {
        FscanfOrDie(fptr, "%lf", observations_ + 2*i + j);
      }
    }

    for (int i = 0; i < num_parameters_; ++i) {
      FscanfOrDie(fptr, "%lf", parameters_ + i);
    }
  }

  void BuildProblem() {
    double* points = mutable_points();
    double* cameras = mutable_cameras();

    for (int i = 0; i < num_observations(); ++i) {
      // Each Residual block takes a point and a camera as input and
      // outputs a 2 dimensional residual.
      CostFunction* cost_function =
          new AutoDiffCostFunction<BundlerResidual, 2, 9, 3>(
              new BundlerResidual(observations_[2*i + 0],
                                  observations_[2*i + 1]));

      // Each observation correponds to a pair of a camera and a point
      // which are identified by camera_index()[i] and
      // point_index()[i] respectively.
      double* camera = cameras + 9 * camera_index_[i];
      double* point = points + 3 * point_index()[i];
      problem_.AddResidualBlock(cost_function, NULL, camera, point);
    }

    options_.linear_solver_ordering = new ParameterBlockOrdering;

    // The points come before the cameras.
    for (int i = 0; i < num_points_; ++i) {
      options_.linear_solver_ordering->AddElementToGroup(points + 3 * i, 0);
    }

    for (int i = 0; i < num_cameras_; ++i) {
      options_.linear_solver_ordering->AddElementToGroup(cameras + 9 * i, 1);
    }

    options_.max_num_iterations = 25;
    options_.function_tolerance = 1e-10;
    options_.gradient_tolerance = 1e-10;
    options_.parameter_tolerance = 1e-10;
  }

  template<typename T>
  void FscanfOrDie(FILE *fptr, const char *format, T *value) {
    int num_scanned = fscanf(fptr, format, value);
    if (num_scanned != 1) {
      LOG(FATAL) << "Invalid UW data file.";
    }
  }

  // Templated pinhole camera model.  The camera is parameterized
  // using 9 parameters. 3 for rotation, 3 for translation, 1 for
  // focal length and 2 for radial distortion. The principal point is
  // not modeled (i.e. it is assumed be located at the image center).
  struct BundlerResidual {
    // (u, v): the position of the observation with respect to the image
    // center point.
    BundlerResidual(double u, double v): u(u), v(v) {}

    template <typename T>
    bool operator()(const T* const camera,
                    const T* const point,
                    T* residuals) const {
      T p[3];
      AngleAxisRotatePoint(camera, point, p);

      // Add the translation vector
      p[0] += camera[3];
      p[1] += camera[4];
      p[2] += camera[5];

      const T& focal = camera[6];
      const T& l1 = camera[7];
      const T& l2 = camera[8];

      // Compute the center of distortion.  The sign change comes from
      // the camera model that Noah Snavely's Bundler assumes, whereby
      // the camera coordinate system has a negative z axis.
      T xp = - focal * p[0] / p[2];
      T yp = - focal * p[1] / p[2];

      // Apply second and fourth order radial distortion.
      T r2 = xp*xp + yp*yp;
      T distortion = T(1.0) + r2  * (l1 + l2  * r2);

      residuals[0] = distortion * xp - T(u);
      residuals[1] = distortion * yp - T(v);

      return true;
    }

    double u;
    double v;
  };


  Problem problem_;
  Solver::Options options_;

  int num_cameras_;
  int num_points_;
  int num_observations_;
  int num_parameters_;

  int* point_index_;
  int* camera_index_;
  double* observations_;
  // The parameter vector is laid out as follows
  // [camera_1, ..., camera_n, point_1, ..., point_m]
  double* parameters_;
};

TEST(SystemTest, BundleAdjustmentProblem) {
  vector<SolverConfig> configs;

#define CONFIGURE(linear_solver, sparse_linear_algebra_library, ordering, preconditioner) \
  configs.push_back(SolverConfig(linear_solver,                         \
                                 sparse_linear_algebra_library,         \
                                 ordering,                              \
                                 preconditioner))

#ifndef CERES_NO_SUITESPARSE
  CONFIGURE(SPARSE_NORMAL_CHOLESKY, SUITE_SPARSE, kAutomaticOrdering, IDENTITY);
  CONFIGURE(SPARSE_NORMAL_CHOLESKY, SUITE_SPARSE, kUserOrdering,      IDENTITY);

  CONFIGURE(SPARSE_SCHUR,           SUITE_SPARSE, kAutomaticOrdering, IDENTITY);
  CONFIGURE(SPARSE_SCHUR,           SUITE_SPARSE, kUserOrdering,      IDENTITY);
#endif  // CERES_NO_SUITESPARSE

#ifndef CERES_NO_CXSPARSE
  CONFIGURE(SPARSE_SCHUR,           CX_SPARSE,    kAutomaticOrdering, IDENTITY);
  CONFIGURE(SPARSE_SCHUR,           CX_SPARSE,    kUserOrdering,      IDENTITY);
#endif  // CERES_NO_CXSPARSE

  CONFIGURE(DENSE_SCHUR,            SUITE_SPARSE, kAutomaticOrdering, IDENTITY);
  CONFIGURE(DENSE_SCHUR,            SUITE_SPARSE, kUserOrdering,      IDENTITY);

  CONFIGURE(CGNR,                   SUITE_SPARSE, kAutomaticOrdering, JACOBI);
  CONFIGURE(ITERATIVE_SCHUR,        SUITE_SPARSE, kUserOrdering,      JACOBI);
  CONFIGURE(ITERATIVE_SCHUR,        SUITE_SPARSE, kUserOrdering,      SCHUR_JACOBI);

#ifndef CERES_NO_SUITESPARSE

  CONFIGURE(ITERATIVE_SCHUR,        SUITE_SPARSE, kUserOrdering,      CLUSTER_JACOBI);
  CONFIGURE(ITERATIVE_SCHUR,        SUITE_SPARSE, kUserOrdering,      CLUSTER_TRIDIAGONAL);
#endif  // CERES_NO_SUITESPARSE

  CONFIGURE(ITERATIVE_SCHUR,        SUITE_SPARSE, kAutomaticOrdering, JACOBI);
  CONFIGURE(ITERATIVE_SCHUR,        SUITE_SPARSE, kAutomaticOrdering, SCHUR_JACOBI);

#ifndef CERES_NO_SUITESPARSE

  CONFIGURE(ITERATIVE_SCHUR,        SUITE_SPARSE, kAutomaticOrdering, CLUSTER_JACOBI);
  CONFIGURE(ITERATIVE_SCHUR,        SUITE_SPARSE, kAutomaticOrdering, CLUSTER_TRIDIAGONAL);
#endif  // CERES_NO_SUITESPARSE

#undef CONFIGURE

  // Single threaded evaluators and linear solvers.
  const double kMaxAbsoluteDifference = 1e-4;
  RunSolversAndCheckTheyMatch<BundleAdjustmentProblem>(configs,
                                                       kMaxAbsoluteDifference);

#ifdef CERES_USE_OPENMP
  // Multithreaded evaluators and linear solvers.
  for (int i = 0; i < configs.size(); ++i) {
    configs[i].num_threads = 2;
  }
  RunSolversAndCheckTheyMatch<BundleAdjustmentProblem>(configs,
                                                       kMaxAbsoluteDifference);
#endif  // CERES_USE_OPENMP
}

}  // namespace internal
}  // namespace ceres
