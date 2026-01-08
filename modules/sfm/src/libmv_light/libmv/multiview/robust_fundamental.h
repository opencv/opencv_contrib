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

#ifndef LIBMV_MULTIVIEW_ROBUST_FUNDAMENTAL_H_
#define LIBMV_MULTIVIEW_ROBUST_FUNDAMENTAL_H_

#include "libmv/base/vector.h"
#include "libmv/numeric/numeric.h"

namespace libmv {

// Estimate robustly the fundamental matrix between two dataset of 2D point
// (image coords space). The fundamental solver relies on the 8 point solution.
// Returns the score associated to the solution F
double FundamentalFromCorrespondences8PointRobust(
    const Mat &x1,
    const Mat &x2,
    double max_error,
    Mat3 *F,
    vector<int> *inliers = NULL,
    double outliers_probability = 1e-2);

// Estimate robustly the fundamental matrix between two dataset of 2D point
// (image coords space). The fundamental solver relies on the 7 point solution.
// Returns the score associated to the solution F
double FundamentalFromCorrespondences7PointRobust(
    const Mat &x1,
    const Mat &x2,
    double max_error,
    Mat3 * F,
    vector<int> *inliers = NULL,
    double outliers_probability = 1e-2);

} // namespace libmv

#endif  // LIBMV_MULTIVIEW_ROBUST_FUNDAMENTAL_H_
