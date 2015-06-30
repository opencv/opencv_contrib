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

#include "ceres/numeric_diff_cost_function.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>
#include "ceres/internal/macros.h"
#include "ceres/internal/scoped_ptr.h"
#include "ceres/numeric_diff_test_utils.h"
#include "ceres/test_util.h"
#include "ceres/types.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

TEST(NumericDiffCostFunction, EasyCaseFunctorCentralDifferences) {
  internal::scoped_ptr<CostFunction> cost_function;
  cost_function.reset(
      new NumericDiffCostFunction<EasyFunctor,
                                  CENTRAL,
                                  3,  /* number of residuals */
                                  5,  /* size of x1 */
                                  5   /* size of x2 */>(
          new EasyFunctor));
  EasyFunctor functor;
  functor.ExpectCostFunctionEvaluationIsNearlyCorrect(*cost_function, CENTRAL);
}

TEST(NumericDiffCostFunction, EasyCaseFunctorForwardDifferences) {
  internal::scoped_ptr<CostFunction> cost_function;
  cost_function.reset(
      new NumericDiffCostFunction<EasyFunctor,
                                  FORWARD,
                                  3,  /* number of residuals */
                                  5,  /* size of x1 */
                                  5   /* size of x2 */>(
          new EasyFunctor));
  EasyFunctor functor;
  functor.ExpectCostFunctionEvaluationIsNearlyCorrect(*cost_function, FORWARD);
}

TEST(NumericDiffCostFunction, EasyCaseCostFunctionCentralDifferences) {
  internal::scoped_ptr<CostFunction> cost_function;
  cost_function.reset(
      new NumericDiffCostFunction<EasyCostFunction,
                                  CENTRAL,
                                  3,  /* number of residuals */
                                  5,  /* size of x1 */
                                  5   /* size of x2 */>(
          new EasyCostFunction, TAKE_OWNERSHIP));
  EasyFunctor functor;
  functor.ExpectCostFunctionEvaluationIsNearlyCorrect(*cost_function, CENTRAL);
}

TEST(NumericDiffCostFunction, EasyCaseCostFunctionForwardDifferences) {
  internal::scoped_ptr<CostFunction> cost_function;
  cost_function.reset(
      new NumericDiffCostFunction<EasyCostFunction,
                                  FORWARD,
                                  3,  /* number of residuals */
                                  5,  /* size of x1 */
                                  5   /* size of x2 */>(
          new EasyCostFunction, TAKE_OWNERSHIP));
  EasyFunctor functor;
  functor.ExpectCostFunctionEvaluationIsNearlyCorrect(*cost_function, FORWARD);
}

TEST(NumericDiffCostFunction, TranscendentalCaseFunctorCentralDifferences) {
  internal::scoped_ptr<CostFunction> cost_function;
  cost_function.reset(
      new NumericDiffCostFunction<TranscendentalFunctor,
                                  CENTRAL,
                                  2,  /* number of residuals */
                                  5,  /* size of x1 */
                                  5   /* size of x2 */>(
          new TranscendentalFunctor));
  TranscendentalFunctor functor;
  functor.ExpectCostFunctionEvaluationIsNearlyCorrect(*cost_function, CENTRAL);
}

TEST(NumericDiffCostFunction, TranscendentalCaseFunctorForwardDifferences) {
  internal::scoped_ptr<CostFunction> cost_function;
  cost_function.reset(
      new NumericDiffCostFunction<TranscendentalFunctor,
                                  FORWARD,
                                  2,  /* number of residuals */
                                  5,  /* size of x1 */
                                  5   /* size of x2 */>(
          new TranscendentalFunctor));
  TranscendentalFunctor functor;
  functor.ExpectCostFunctionEvaluationIsNearlyCorrect(*cost_function, FORWARD);
}

TEST(NumericDiffCostFunction, TranscendentalCaseCostFunctionCentralDifferences) {
  internal::scoped_ptr<CostFunction> cost_function;
  cost_function.reset(
      new NumericDiffCostFunction<TranscendentalCostFunction,
                                  CENTRAL,
                                  2,  /* number of residuals */
                                  5,  /* size of x1 */
                                  5   /* size of x2 */>(
          new TranscendentalCostFunction, TAKE_OWNERSHIP));
  TranscendentalFunctor functor;
  functor.ExpectCostFunctionEvaluationIsNearlyCorrect(*cost_function, CENTRAL);
}

TEST(NumericDiffCostFunction, TranscendentalCaseCostFunctionForwardDifferences) {
  internal::scoped_ptr<CostFunction> cost_function;
  cost_function.reset(
      new NumericDiffCostFunction<TranscendentalCostFunction,
                                  FORWARD,
                                  2,  /* number of residuals */
                                  5,  /* size of x1 */
                                  5   /* size of x2 */>(
          new TranscendentalCostFunction, TAKE_OWNERSHIP));
  TranscendentalFunctor functor;
  functor.ExpectCostFunctionEvaluationIsNearlyCorrect(*cost_function, FORWARD);
}

template<int num_rows, int num_cols>
class SizeTestingCostFunction : public SizedCostFunction<num_rows, num_cols> {
 public:
  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    return true;
  }
};

// As described in
// http://forum.kde.org/viewtopic.php?f=74&t=98536#p210774
// Eigen3 has restrictions on the Row/Column major storage of vectors,
// depending on their dimensions. This test ensures that the correct
// templates are instantiated for various shapes of the Jacobian
// matrix.
TEST(NumericDiffCostFunction, EigenRowMajorColMajorTest) {
  scoped_ptr<CostFunction> cost_function;
  cost_function.reset(
      new NumericDiffCostFunction<SizeTestingCostFunction<1,1>,  CENTRAL, 1, 1>(
          new SizeTestingCostFunction<1,1>, ceres::TAKE_OWNERSHIP));

  cost_function.reset(
      new NumericDiffCostFunction<SizeTestingCostFunction<2,1>,  CENTRAL, 2, 1>(
          new SizeTestingCostFunction<2,1>, ceres::TAKE_OWNERSHIP));

  cost_function.reset(
      new NumericDiffCostFunction<SizeTestingCostFunction<1,2>,  CENTRAL, 1, 2>(
          new SizeTestingCostFunction<1,2>, ceres::TAKE_OWNERSHIP));

  cost_function.reset(
      new NumericDiffCostFunction<SizeTestingCostFunction<2,2>,  CENTRAL, 2, 2>(
          new SizeTestingCostFunction<2,2>, ceres::TAKE_OWNERSHIP));
}

}  // namespace internal
}  // namespace ceres
