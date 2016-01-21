// Copyright (c) 2007, 2008 libmv authors.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

#include "libmv/multiview/fundamental_kernel.h"
#include "libmv/multiview/robust_estimation.h"
#include "libmv/multiview/robust_fundamental.h"
#include "libmv/numeric/numeric.h"

namespace libmv {

// TODO(keir): This interface is a bit ugly; consider fixing it.
double FundamentalFromCorrespondences8PointRobust(const Mat &x1,
                                                  const Mat &x2,
                                                  double max_error,
                                                  Mat3 *F,
                                                  vector<int> *inliers,
                                                  double outliers_probability) {
  // The threshold is on the sum of the squared errors in the two images.
  // Actually, Sampson's approximation of this error.
  double threshold = 2 * Square(max_error);
  double best_score = HUGE_VAL;
  typedef fundamental::kernel::NormalizedEightPointKernel Kernel;
  Kernel kernel(x1, x2);
  *F = Estimate(kernel, MLEScorer<Kernel>(threshold), inliers,
                &best_score, outliers_probability);
  if (best_score == HUGE_VAL)
    return HUGE_VAL;
  else
    return std::sqrt(best_score / 2.0);
}

double FundamentalFromCorrespondences7PointRobust(const Mat &x1,
                                                  const Mat &x2,
                                                  double max_error,
                                                  Mat3 * F,
                                                  vector<int> *inliers,
                                                  double outliers_probability) {
  // The threshold is on the sum of the squared errors in the two images.
  // Actually, Sampson's approximation of this error.
  double threshold = 2 * Square(max_error);
  double best_score = HUGE_VAL;
  typedef fundamental::kernel::NormalizedSevenPointKernel Kernel;
  Kernel kernel(x1, x2);
  *F = Estimate(kernel, MLEScorer<Kernel>(threshold), inliers,
                &best_score, outliers_probability);
  if (best_score == HUGE_VAL)
    return HUGE_VAL;
  else
    return std::sqrt(best_score / 2.0);
}

}  // namespace libmv
