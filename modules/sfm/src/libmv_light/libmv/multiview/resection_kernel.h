// Copyright (c) 2009 libmv authors.
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

#ifndef LIBMV_MULTIVIEW_RESECTION_KERNEL_H
#define LIBMV_MULTIVIEW_RESECTION_KERNEL_H

#include "libmv/base/vector.h"
#include "libmv/logging/logging.h"
#include "libmv/multiview/resection.h"
#include "libmv/multiview/projection.h"
#include "libmv/numeric/numeric.h"

namespace libmv {
namespace resection {
namespace kernel {

class Kernel {
 public:
  typedef Mat34 Model;
  enum { MINIMUM_SAMPLES = 6 };

  Kernel(const Mat2X &x, const Mat4X &X) : x_(x), X_(X) {
    CHECK(x.cols() == X.cols());
  }
  void Fit(const vector<int> &samples, vector<Model> *models) const {
    Mat2X x = ExtractColumns(x_, samples);
    Mat4X X = ExtractColumns(X_, samples);
    Mat34 P;
    Resection(x, X, &P);
    models->push_back(P);
  }
  double Error(int sample, const Model &model) const {
    Mat4X X = X_.col(sample);
    Mat2X error = Project(model, X) - x_.col(sample);
    return error.col(0).squaredNorm();
  }
  int NumSamples() const {
    return x_.cols();
  }
 private:
  const Mat2X &x_;
  const Mat4X &X_;
};

}  // namespace kernel
}  // namespace resection
}  // namespace libmv

#endif  // LIBMV_MULTIVIEW_RESECTION_KERNEL_H
