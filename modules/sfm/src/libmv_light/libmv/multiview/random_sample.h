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

#ifndef LIBMV_MULTIVIEW_RANDOM_SAMPLE_H_
#define LIBMV_MULTIVIEW_RANDOM_SAMPLE_H_

#include "libmv/base/vector.h"
#include "libmv/logging/logging.h"

namespace libmv {

/*!
    Pick a random subset of the integers [0, total), in random order. Note that
    this can behave badly if num_samples is close to total; runtime could be
    unlimited!

    This uses a quadratic rejection strategy and should only be used for small
    num_samples.

    \param num_samples   The number of samples to produce.
    \param total_samples The number of samples available.
    \param samples       num_samples of numbers in [0, total_samples) is placed
                         here on return.
*/
static void UniformSample(int num_samples,
                          int total_samples,
                          vector<int> *samples) {
  samples->resize(0);
  while (samples->size() < num_samples) {
    int sample = rand() % total_samples;
    bool found = false;
    for (int j = 0; j < samples->size(); ++j) {
      found = (*samples)[j] == sample;
      if (found) {
        break;
      }
    }
    if (!found) {
      samples->push_back(sample);
    }
  }
}

}  // namespace libmv

#endif  // LIBMV_MULTIVIEW_RANDOM_SAMPLE_H_
