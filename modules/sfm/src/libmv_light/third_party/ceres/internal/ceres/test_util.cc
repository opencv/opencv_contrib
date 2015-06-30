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
// Utility functions useful for testing.

#include <cmath>
#include "ceres/file.h"
#include "ceres/stringprintf.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

DECLARE_string(test_srcdir);

// This macro is used to inject additional path information specific
// to the build system.

#ifndef CERES_TEST_SRCDIR_SUFFIX
#define CERES_TEST_SRCDIR_SUFFIX ""
#endif

namespace ceres {
namespace internal {

bool ExpectClose(double x, double y, double max_abs_relative_difference) {
  double absolute_difference = fabs(x - y);
  double relative_difference = absolute_difference / std::max(fabs(x), fabs(y));
  if (x == 0 || y == 0) {
    // If x or y is exactly zero, then relative difference doesn't have any
    // meaning. Take the absolute difference instead.
    relative_difference = absolute_difference;
  }
  if (relative_difference > max_abs_relative_difference) {
    VLOG(1) << StringPrintf("x=%17g y=%17g abs=%17g rel=%17g",
                            x, y, absolute_difference, relative_difference);
  }

  EXPECT_NEAR(relative_difference, 0.0, max_abs_relative_difference);
  return relative_difference <= max_abs_relative_difference;
}

void ExpectArraysCloseUptoScale(int n,
                                const double* p,
                                const double* q,
                                double tol) {
  CHECK_GT(n, 0);
  CHECK(p);
  CHECK(q);

  double p_max = 0;
  double q_max = 0;
  int p_i = 0;
  int q_i = 0;

  for (int i = 0; i < n; ++i) {
    if (std::abs(p[i]) > p_max) {
      p_max = std::abs(p[i]);
      p_i = i;
    }
    if (std::abs(q[i]) > q_max) {
      q_max = std::abs(q[i]);
      q_i = i;
    }
  }

  // If both arrays are all zeros, they are equal up to scale, but
  // for testing purposes, that's more likely to be an error than
  // a desired result.
  CHECK_NE(p_max, 0.0);
  CHECK_NE(q_max, 0.0);

  for (int i = 0; i < n; ++i) {
    double p_norm = p[i] / p[p_i];
    double q_norm = q[i] / q[q_i];

    EXPECT_NEAR(p_norm, q_norm, tol) << "i=" << i;
  }
}

void ExpectArraysClose(int n,
                       const double* p,
                       const double* q,
                       double tol) {
  CHECK_GT(n, 0);
  CHECK(p);
  CHECK(q);

  for (int i = 0; i < n; ++i) {
    EXPECT_NEAR(p[i], q[i], tol) << "i=" << i;
  }
}

string TestFileAbsolutePath(const string& filename) {
  return JoinPath(FLAGS_test_srcdir + CERES_TEST_SRCDIR_SUFFIX,
                  filename);
}


}  // namespace internal
}  // namespace ceres
