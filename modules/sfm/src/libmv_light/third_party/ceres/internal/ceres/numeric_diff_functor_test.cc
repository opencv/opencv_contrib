// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2013 Google Inc. All rights reserved.
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

#include "ceres/numeric_diff_functor.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>
#include "ceres/autodiff_cost_function.h"
#include "ceres/internal/scoped_ptr.h"
#include "ceres/numeric_diff_test_utils.h"
#include "ceres/test_util.h"
#include "ceres/types.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

TEST(NumericDiffCostFunction, EasyCaseCentralDifferences) {
  typedef NumericDiffFunctor<EasyFunctor, CENTRAL, 3, 5, 5>
      NumericDiffEasyFunctor;

  internal::scoped_ptr<CostFunction> cost_function;
  EasyFunctor functor;

  cost_function.reset(
      new AutoDiffCostFunction<NumericDiffEasyFunctor, 3, 5, 5>(
          new NumericDiffEasyFunctor));

  functor.ExpectCostFunctionEvaluationIsNearlyCorrect(*cost_function, CENTRAL);

  cost_function.reset(
      new AutoDiffCostFunction<NumericDiffEasyFunctor, 3, 5, 5>(
          new NumericDiffEasyFunctor(new EasyFunctor)));
  functor.ExpectCostFunctionEvaluationIsNearlyCorrect(*cost_function, CENTRAL);
}

TEST(NumericDiffCostFunction, EasyCaseForwardDifferences) {
  typedef NumericDiffFunctor<EasyFunctor, FORWARD, 3, 5, 5>
      NumericDiffEasyFunctor;

  internal::scoped_ptr<CostFunction> cost_function;
  EasyFunctor functor;

  cost_function.reset(
      new AutoDiffCostFunction<NumericDiffEasyFunctor, 3, 5, 5>(
          new NumericDiffEasyFunctor));

  functor.ExpectCostFunctionEvaluationIsNearlyCorrect(*cost_function, FORWARD);

  cost_function.reset(
      new AutoDiffCostFunction<NumericDiffEasyFunctor, 3, 5, 5>(
          new NumericDiffEasyFunctor(new EasyFunctor)));
  functor.ExpectCostFunctionEvaluationIsNearlyCorrect(*cost_function, FORWARD);
}

TEST(NumericDiffCostFunction, TranscendentalCaseCentralDifferences) {
  typedef NumericDiffFunctor<TranscendentalFunctor, CENTRAL, 2, 5, 5>
      NumericDiffTranscendentalFunctor;

  internal::scoped_ptr<CostFunction> cost_function;
  TranscendentalFunctor functor;

  cost_function.reset(
      new AutoDiffCostFunction<NumericDiffTranscendentalFunctor, 2, 5, 5>(
          new NumericDiffTranscendentalFunctor));

  functor.ExpectCostFunctionEvaluationIsNearlyCorrect(*cost_function, CENTRAL);

  cost_function.reset(
      new AutoDiffCostFunction<NumericDiffTranscendentalFunctor, 2, 5, 5>(
          new NumericDiffTranscendentalFunctor(new TranscendentalFunctor)));
  functor.ExpectCostFunctionEvaluationIsNearlyCorrect(*cost_function, CENTRAL);
}

TEST(NumericDiffCostFunction, TranscendentalCaseForwardDifferences) {
  typedef NumericDiffFunctor<TranscendentalFunctor, FORWARD, 2, 5, 5>
      NumericDiffTranscendentalFunctor;

  internal::scoped_ptr<CostFunction> cost_function;
  TranscendentalFunctor functor;

  cost_function.reset(
      new AutoDiffCostFunction<NumericDiffTranscendentalFunctor, 2, 5, 5>(
          new NumericDiffTranscendentalFunctor));

  functor.ExpectCostFunctionEvaluationIsNearlyCorrect(*cost_function, FORWARD);

  cost_function.reset(
      new AutoDiffCostFunction<NumericDiffTranscendentalFunctor, 2, 5, 5>(
          new NumericDiffTranscendentalFunctor(new TranscendentalFunctor)));
  functor.ExpectCostFunctionEvaluationIsNearlyCorrect(*cost_function, FORWARD);
}

}  // namespace internal
}  // namespace ceres
