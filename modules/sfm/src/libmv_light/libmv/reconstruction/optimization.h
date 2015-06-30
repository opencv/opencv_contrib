// Copyright (c) 2010 libmv authors.
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

#ifndef LIBMV_RECONSTRUCTION_OPTIMIZATION_H_
#define LIBMV_RECONSTRUCTION_OPTIMIZATION_H_

#include "libmv/reconstruction/reconstruction.h"

namespace libmv {

double EstimateRootMeanSquareError(const Matches &matches,
                                   Reconstruction *reconstruction);

// This method performs an Euclidean Bundle Adjustment
// and returns the root mean square error.
double MetricBundleAdjust(const Matches &matches,
                          Reconstruction *reconstruction);

// Remove the matches associated to the points structures seen in the image
// image_id and have a root mean square error bigger than rmse_threshold
// NOTE It is at least barely started
uint RemoveOutliers(CameraID image_id,
                    Matches *matches,
                    Reconstruction *reconstruction,
                    double rmse_threshold = 2.0);


}  // namespace libmv

#endif  // LIBMV_RECONSTRUCTION_OPTIMIZATION_H_
