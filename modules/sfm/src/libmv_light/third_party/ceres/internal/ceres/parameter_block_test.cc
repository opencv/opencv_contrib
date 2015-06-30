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

#include "ceres/parameter_block.h"

#include "gtest/gtest.h"
#include "ceres/internal/eigen.h"

namespace ceres {
namespace internal {

TEST(ParameterBlock, SetLocalParameterization) {
  double x[3] = { 1.0, 2.0, 3.0 };
  ParameterBlock parameter_block(x, 3, -1);

  // The indices to set constant within the parameter block (used later).
  vector<int> indices;
  indices.push_back(1);

  // Can't set the parameterization if the sizes don't match.
  SubsetParameterization subset_wrong_size(4, indices);
  EXPECT_DEATH_IF_SUPPORTED(
      parameter_block.SetParameterization(&subset_wrong_size), "global");

  // Can't set parameterization to NULL from NULL.
  EXPECT_DEATH_IF_SUPPORTED
      (parameter_block.SetParameterization(NULL), "NULL");

  // Now set the parameterization.
  SubsetParameterization subset(3, indices);
  parameter_block.SetParameterization(&subset);

  // Re-setting the parameterization to the same value is supported.
  parameter_block.SetParameterization(&subset);

  // Can't set parameterization to NULL from another parameterization.
  EXPECT_DEATH_IF_SUPPORTED(parameter_block.SetParameterization(NULL), "NULL");

  // Can't set the parameterization more than once.
  SubsetParameterization subset_different(3, indices);
  EXPECT_DEATH_IF_SUPPORTED
      (parameter_block.SetParameterization(&subset_different), "re-set");

  // Ensure the local parameterization jacobian result is correctly computed.
  ConstMatrixRef local_parameterization_jacobian(
      parameter_block.LocalParameterizationJacobian(),
      3,
      2);
  ASSERT_EQ(1.0, local_parameterization_jacobian(0, 0));
  ASSERT_EQ(0.0, local_parameterization_jacobian(0, 1));
  ASSERT_EQ(0.0, local_parameterization_jacobian(1, 0));
  ASSERT_EQ(0.0, local_parameterization_jacobian(1, 1));
  ASSERT_EQ(0.0, local_parameterization_jacobian(2, 0));
  ASSERT_EQ(1.0, local_parameterization_jacobian(2, 1));

  // Check that updating works as expected.
  double x_plus_delta[3];
  double delta[2] = { 0.5, 0.3 };
  parameter_block.Plus(x, delta, x_plus_delta);
  ASSERT_EQ(1.5, x_plus_delta[0]);
  ASSERT_EQ(2.0, x_plus_delta[1]);
  ASSERT_EQ(3.3, x_plus_delta[2]);
}

struct TestParameterization : public LocalParameterization {
 public:
  virtual ~TestParameterization() {}
  virtual bool Plus(const double* x,
                    const double* delta,
                    double* x_plus_delta) const {
    LOG(FATAL) << "Shouldn't get called.";
    return true;
  }
  virtual bool ComputeJacobian(const double* x,
                               double* jacobian) const {
    jacobian[0] = *x * 2;
    return true;
  }

  virtual int GlobalSize() const { return 1; }
  virtual int LocalSize() const { return 1; }
};

TEST(ParameterBlock, SetStateUpdatesLocalParameterizationJacobian) {
  TestParameterization test_parameterization;
  double x[1] = { 1.0 };
  ParameterBlock parameter_block(x, 1, -1, &test_parameterization);

  EXPECT_EQ(2.0, *parameter_block.LocalParameterizationJacobian());

  x[0] = 5.5;
  parameter_block.SetState(x);
  EXPECT_EQ(11.0, *parameter_block.LocalParameterizationJacobian());
}

TEST(ParameterBlock, PlusWithNoLocalParameterization) {
  double x[2] = { 1.0, 2.0 };
  ParameterBlock parameter_block(x, 2, -1);

  double delta[2] = { 0.2, 0.3 };
  double x_plus_delta[2];
  parameter_block.Plus(x, delta, x_plus_delta);
  EXPECT_EQ(1.2, x_plus_delta[0]);
  EXPECT_EQ(2.3, x_plus_delta[1]);
}

// Stops computing the jacobian after the first time.
class BadLocalParameterization : public LocalParameterization {
 public:
  BadLocalParameterization()
      : calls_(0) {
  }

  virtual ~BadLocalParameterization() {}
  virtual bool Plus(const double* x,
                    const double* delta,
                    double* x_plus_delta) const {
    *x_plus_delta = *x + *delta;
    return true;
  }

  virtual bool ComputeJacobian(const double* x, double* jacobian) const {
    if (calls_ == 0) {
      jacobian[0] = 0;
    }
    ++calls_;
    return true;
  }

  virtual int GlobalSize() const { return 1;}
  virtual int LocalSize()  const { return 1;}

 private:
  mutable int calls_;
};

TEST(ParameterBlock, DetectBadLocalParameterization) {
  double x = 1;
  BadLocalParameterization bad_parameterization;
  ParameterBlock parameter_block(&x, 1, -1, &bad_parameterization);
  double y = 2;
  EXPECT_FALSE(parameter_block.SetState(&y));
}

}  // namespace internal
}  // namespace ceres
