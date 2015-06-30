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

#include "ceres/loss_function.h"

#include <cstddef>

#include "glog/logging.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {
namespace {

// Helper function for testing a LossFunction callback.
//
// Compares the values of rho'(s) and rho''(s) computed by the
// callback with estimates obtained by symmetric finite differencing
// of rho(s).
void AssertLossFunctionIsValid(const LossFunction& loss, double s) {
  CHECK_GT(s, 0);

  // Evaluate rho(s), rho'(s) and rho''(s).
  double rho[3];
  loss.Evaluate(s, rho);

  // Use symmetric finite differencing to estimate rho'(s) and
  // rho''(s).
  const double kH = 1e-4;
  // Values at s + kH.
  double fwd[3];
  // Values at s - kH.
  double bwd[3];
  loss.Evaluate(s + kH, fwd);
  loss.Evaluate(s - kH, bwd);

  // First derivative.
  const double fd_1 = (fwd[0] - bwd[0]) / (2 * kH);
  ASSERT_NEAR(fd_1, rho[1], 1e-6);

  // Second derivative.
  const double fd_2 = (fwd[0] - 2*rho[0] + bwd[0]) / (kH * kH);
  ASSERT_NEAR(fd_2, rho[2], 1e-6);
}
}  // namespace

// Try two values of the scaling a = 0.7 and 1.3
// (where scaling makes sense) and of the squared norm
// s = 0.357 and 1.792
//
// Note that for the Huber loss the test exercises both code paths
//  (i.e. both small and large values of s).

TEST(LossFunction, TrivialLoss) {
  AssertLossFunctionIsValid(TrivialLoss(), 0.357);
  AssertLossFunctionIsValid(TrivialLoss(), 1.792);
}

TEST(LossFunction, HuberLoss) {
  AssertLossFunctionIsValid(HuberLoss(0.7), 0.357);
  AssertLossFunctionIsValid(HuberLoss(0.7), 1.792);
  AssertLossFunctionIsValid(HuberLoss(1.3), 0.357);
  AssertLossFunctionIsValid(HuberLoss(1.3), 1.792);
}

TEST(LossFunction, SoftLOneLoss) {
  AssertLossFunctionIsValid(SoftLOneLoss(0.7), 0.357);
  AssertLossFunctionIsValid(SoftLOneLoss(0.7), 1.792);
  AssertLossFunctionIsValid(SoftLOneLoss(1.3), 0.357);
  AssertLossFunctionIsValid(SoftLOneLoss(1.3), 1.792);
}

TEST(LossFunction, CauchyLoss) {
  AssertLossFunctionIsValid(CauchyLoss(0.7), 0.357);
  AssertLossFunctionIsValid(CauchyLoss(0.7), 1.792);
  AssertLossFunctionIsValid(CauchyLoss(1.3), 0.357);
  AssertLossFunctionIsValid(CauchyLoss(1.3), 1.792);
}

TEST(LossFunction, ArctanLoss) {
  AssertLossFunctionIsValid(ArctanLoss(0.7), 0.357);
  AssertLossFunctionIsValid(ArctanLoss(0.7), 1.792);
  AssertLossFunctionIsValid(ArctanLoss(1.3), 0.357);
  AssertLossFunctionIsValid(ArctanLoss(1.3), 1.792);
}

TEST(LossFunction, TolerantLoss) {
  AssertLossFunctionIsValid(TolerantLoss(0.7, 0.4), 0.357);
  AssertLossFunctionIsValid(TolerantLoss(0.7, 0.4), 1.792);
  AssertLossFunctionIsValid(TolerantLoss(0.7, 0.4), 55.5);
  AssertLossFunctionIsValid(TolerantLoss(1.3, 0.1), 0.357);
  AssertLossFunctionIsValid(TolerantLoss(1.3, 0.1), 1.792);
  AssertLossFunctionIsValid(TolerantLoss(1.3, 0.1), 55.5);
  // Check the value at zero is actually zero.
  double rho[3];
  TolerantLoss(0.7, 0.4).Evaluate(0.0, rho);
  ASSERT_NEAR(rho[0], 0.0, 1e-6);
  // Check that loss before and after the approximation threshold are good.
  // A threshold of 36.7 is used by the implementation.
  AssertLossFunctionIsValid(TolerantLoss(20.0, 1.0), 20.0 + 36.6);
  AssertLossFunctionIsValid(TolerantLoss(20.0, 1.0), 20.0 + 36.7);
  AssertLossFunctionIsValid(TolerantLoss(20.0, 1.0), 20.0 + 36.8);
  AssertLossFunctionIsValid(TolerantLoss(20.0, 1.0), 20.0 + 1000.0);
}

TEST(LossFunction, ComposedLoss) {
  {
    HuberLoss f(0.7);
    CauchyLoss g(1.3);
    ComposedLoss c(&f, DO_NOT_TAKE_OWNERSHIP, &g, DO_NOT_TAKE_OWNERSHIP);
    AssertLossFunctionIsValid(c, 0.357);
    AssertLossFunctionIsValid(c, 1.792);
  }
  {
    CauchyLoss f(0.7);
    HuberLoss g(1.3);
    ComposedLoss c(&f, DO_NOT_TAKE_OWNERSHIP, &g, DO_NOT_TAKE_OWNERSHIP);
    AssertLossFunctionIsValid(c, 0.357);
    AssertLossFunctionIsValid(c, 1.792);
  }
}

TEST(LossFunction, ScaledLoss) {
  // Wrap a few loss functions, and a few scale factors. This can't combine
  // construction with the call to AssertLossFunctionIsValid() because Apple's
  // GCC is unable to eliminate the copy of ScaledLoss, which is not copyable.
  {
    ScaledLoss scaled_loss(NULL, 6, TAKE_OWNERSHIP);
    AssertLossFunctionIsValid(scaled_loss, 0.323);
  }
  {
    ScaledLoss scaled_loss(new TrivialLoss(), 10, TAKE_OWNERSHIP);
    AssertLossFunctionIsValid(scaled_loss, 0.357);
  }
  {
    ScaledLoss scaled_loss(new HuberLoss(0.7), 0.1, TAKE_OWNERSHIP);
    AssertLossFunctionIsValid(scaled_loss, 1.792);
  }
  {
    ScaledLoss scaled_loss(new SoftLOneLoss(1.3), 0.1, TAKE_OWNERSHIP);
    AssertLossFunctionIsValid(scaled_loss, 1.792);
  }
  {
    ScaledLoss scaled_loss(new CauchyLoss(1.3), 10, TAKE_OWNERSHIP);
    AssertLossFunctionIsValid(scaled_loss, 1.792);
  }
  {
    ScaledLoss scaled_loss(new ArctanLoss(1.3), 10, TAKE_OWNERSHIP);
    AssertLossFunctionIsValid(scaled_loss, 1.792);
  }
  {
    ScaledLoss scaled_loss(
        new TolerantLoss(1.3, 0.1), 10, TAKE_OWNERSHIP);
    AssertLossFunctionIsValid(scaled_loss, 1.792);
  }
  {
    ScaledLoss scaled_loss(
        new ComposedLoss(
            new HuberLoss(0.8), TAKE_OWNERSHIP,
            new TolerantLoss(1.3, 0.5), TAKE_OWNERSHIP), 10, TAKE_OWNERSHIP);
    AssertLossFunctionIsValid(scaled_loss, 1.792);
  }
}

TEST(LossFunction, LossFunctionWrapper) {
  // Initialization
  HuberLoss loss_function1(1.0);
  LossFunctionWrapper loss_function_wrapper(new HuberLoss(1.0),
                                            TAKE_OWNERSHIP);

  double s = 0.862;
  double rho_gold[3];
  double rho[3];
  loss_function1.Evaluate(s, rho_gold);
  loss_function_wrapper.Evaluate(s, rho);
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(rho[i], rho_gold[i], 1e-12);
  }

  // Resetting
  HuberLoss loss_function2(0.5);
  loss_function_wrapper.Reset(new HuberLoss(0.5), TAKE_OWNERSHIP);
  loss_function_wrapper.Evaluate(s, rho);
  loss_function2.Evaluate(s, rho_gold);
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(rho[i], rho_gold[i], 1e-12);
  }

  // Not taking ownership.
  HuberLoss loss_function3(0.3);
  loss_function_wrapper.Reset(&loss_function3, DO_NOT_TAKE_OWNERSHIP);
  loss_function_wrapper.Evaluate(s, rho);
  loss_function3.Evaluate(s, rho_gold);
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(rho[i], rho_gold[i], 1e-12);
  }
}

}  // namespace internal
}  // namespace ceres
