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
// This file contains tests for the GradientChecker class.

#include "ceres/gradient_checker.h"

#include <cmath>
#include <cstdlib>
#include <vector>

#include "ceres/cost_function.h"
#include "ceres/random.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

// We pick a (non-quadratic) function whose derivative are easy:
//
//    f = exp(- a' x).
//   df = - f a.
//
// where 'a' is a vector of the same size as 'x'. In the block
// version, they are both block vectors, of course.
class GoodTestTerm : public CostFunction {
 public:
  GoodTestTerm(int arity, int const *dim) : arity_(arity) {
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
          }
        }
      }
    }

    return true;
  }

 private:
  int arity_;
  vector<vector<double> > a_;  // our vectors.
};

class BadTestTerm : public CostFunction {
 public:
  BadTestTerm(int arity, int const *dim) : arity_(arity) {
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
            jacobians[j][u] = - f * a_[j][u] + 0.001;
          }
        }
      }
    }

    return true;
  }

 private:
  int arity_;
  vector<vector<double> > a_;  // our vectors.
};

TEST(GradientChecker, SmokeTest) {
  srand(5);

  // Test with 3 blocks of size 2, 3 and 4.
  int const arity = 3;
  int const dim[arity] = { 2, 3, 4 };

  // Make a random set of blocks.
  FixedArray<double*> parameters(arity);
  for (int j = 0; j < arity; ++j) {
    parameters[j] = new double[dim[j]];
    for (int u = 0; u < dim[j]; ++u) {
      parameters[j][u] = 2.0 * RandDouble() - 1.0;
    }
  }

  // Make a term and probe it.
  GoodTestTerm good_term(arity, dim);
  typedef GradientChecker<GoodTestTerm, 1, 2, 3, 4> GoodTermGradientChecker;
  EXPECT_TRUE(GoodTermGradientChecker::Probe(
      parameters.get(), 1e-6, &good_term, NULL));

  BadTestTerm bad_term(arity, dim);
  typedef GradientChecker<BadTestTerm, 1, 2, 3, 4> BadTermGradientChecker;
  EXPECT_FALSE(BadTermGradientChecker::Probe(
      parameters.get(), 1e-6, &bad_term, NULL));

  for (int j = 0; j < arity; j++) {
    delete[] parameters[j];
  }
}

}  // namespace internal
}  // namespace ceres
