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

#include <string>
#include "ceres/internal/port.h"

#ifndef CERES_INTERNAL_TEST_UTIL_H_
#define CERES_INTERNAL_TEST_UTIL_H_

namespace ceres {
namespace internal {

// Expects that x and y have a relative difference of no more than
// max_abs_relative_difference. If either x or y is zero, then the relative
// difference is interpreted as an absolute difference.
bool ExpectClose(double x, double y, double max_abs_relative_difference);

// Expects that for all i = 1,.., n - 1
//
//   |p[i] - q[i]| / max(|p[i]|, |q[i]|) < tolerance
void ExpectArraysClose(int n,
                       const double* p,
                       const double* q,
                       double tolerance);

// Expects that for all i = 1,.., n - 1
//
//   |p[i] / max_norm_p - q[i] / max_norm_q| < tolerance
//
// where max_norm_p and max_norm_q are the max norms of the arrays p
// and q respectively.
void ExpectArraysCloseUptoScale(int n,
                                const double* p,
                                const double* q,
                                double tolerance);

// Construct a fully qualified path for the test file depending on the
// local build/testing environment.
string TestFileAbsolutePath(const string& filename);

}  // namespace internal
}  // namespace ceres

#endif  // CERES_INTERNAL_TEST_UTIL_H_
