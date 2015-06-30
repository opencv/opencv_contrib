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

#include "gtest/gtest.h"
#include "ceres/iteration_callback.h"
#include "ceres/minimizer.h"
#include "ceres/solver.h"

namespace ceres {
namespace internal {

class FakeIterationCallback : public IterationCallback {
 public:
  virtual ~FakeIterationCallback() {}
  virtual CallbackReturnType operator()(const IterationSummary& summary) {
    return SOLVER_CONTINUE;
  }
};

TEST(MinimizerTest, InitializationCopiesCallbacks) {
  FakeIterationCallback callback0;
  FakeIterationCallback callback1;

  Solver::Options solver_options;
  solver_options.callbacks.push_back(&callback0);
  solver_options.callbacks.push_back(&callback1);

  Minimizer::Options minimizer_options(solver_options);
  ASSERT_EQ(2, minimizer_options.callbacks.size());

  EXPECT_EQ(minimizer_options.callbacks[0], &callback0);
  EXPECT_EQ(minimizer_options.callbacks[1], &callback1);
}

}  // namespace internal
}  // namespace ceres
