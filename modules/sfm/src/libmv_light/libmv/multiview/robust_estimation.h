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

#ifndef LIBMV_MULTIVIEW_ROBUST_ESTIMATION_H_
#define LIBMV_MULTIVIEW_ROBUST_ESTIMATION_H_

#include <set>

#include "libmv/base/vector.h"
#include "libmv/logging/logging.h"
#include "libmv/multiview/random_sample.h"
#include "libmv/numeric/numeric.h"

namespace libmv {

template<typename Kernel>
class MLEScorer {
 public:
  MLEScorer(double threshold) : threshold_(threshold) {}
  double Score(const Kernel &kernel,
               const typename Kernel::Model &model,
               const vector<int> &samples,
               vector<int> *inliers) const {
    double cost = 0.0;
    for (int j = 0; j < samples.size(); ++j) {
      double error = kernel.Error(samples[j], model);
      if (error < threshold_) {
        cost += error;
        inliers->push_back(samples[j]);
      } else {
        cost += threshold_;
      }
    }
    return cost;
  }
 private:
  double threshold_;
};

static uint IterationsRequired(int min_samples,
                        double outliers_probability,
                        double inlier_ratio) {
  return static_cast<uint>(
      log(outliers_probability) / log(1.0 - pow(inlier_ratio, min_samples)));
}

// 1. The model.
// 2. The minimum number of samples needed to fit.
// 3. A way to convert samples to a model.
// 4. A way to convert samples and a model to an error.
//
// 1. Kernel::Model
// 2. Kernel::MINIMUM_SAMPLES
// 3. Kernel::Fit(vector<int>, vector<Kernel::Model> *)
// 4. Kernel::Error(Model, int) -> error
template<typename Kernel, typename Scorer>
typename Kernel::Model Estimate(const Kernel &kernel,
                                const Scorer &scorer,
                                vector<int> *best_inliers = NULL,
                                double *best_score = NULL,
                                double outliers_probability = 1e-2) {
  CHECK(outliers_probability < 1.0);
  CHECK(outliers_probability > 0.0);
  size_t iteration = 0;
  const size_t min_samples = Kernel::MINIMUM_SAMPLES;
  const size_t total_samples = kernel.NumSamples();

  size_t max_iterations = 100;
  const size_t really_max_iterations = 1000;

  int best_num_inliers = 0;
  double best_cost = HUGE_VAL;
  double best_inlier_ratio = 0.0;
  typename Kernel::Model best_model;

  // Test if we have sufficient points to for the kernel.
  if (total_samples < min_samples)  {
    if (best_inliers) {
      best_inliers->resize(0);
    }
    return best_model;
  }

  // In this robust estimator, the scorer always works on all the data points
  // at once. So precompute the list ahead of time.
  vector<int> all_samples;
  for (int i = 0; i < total_samples; ++i) {
    all_samples.push_back(i);
  }

  vector<int> sample;
  for (iteration = 0;
       iteration < max_iterations &&
       iteration < really_max_iterations; ++iteration) {
    UniformSample(min_samples, total_samples, &sample);

    vector<typename Kernel::Model> models;
    kernel.Fit(sample, &models);
    VLOG(4) << "Fitted subset; found " << models.size() << " model(s).";

    // Compute costs for each fit.
    for (int i = 0; i < models.size(); ++i) {
      vector<int> inliers;
      double cost = scorer.Score(kernel, models[i], all_samples, &inliers);
      VLOG(5) << "Fit cost: " << cost
              << ", number of inliers: " << inliers.size();

      if (cost < best_cost) {
        best_cost = cost;
        best_inlier_ratio = inliers.size() / double(total_samples);
        best_num_inliers = inliers.size();
        best_model = models[i];
        if (best_inliers) {
          best_inliers->swap(inliers);
        }
        VLOG(4) << "New best cost: " << best_cost << " with "
                << best_num_inliers << " inlying of "
                << total_samples << " total samples.";
      }
      if (best_inlier_ratio) {
        max_iterations = IterationsRequired(min_samples,
                                            outliers_probability,
                                            best_inlier_ratio);
      }

      VLOG(5) << "Max iterations needed given best inlier ratio: "
        << max_iterations << "; best inlier ratio: " << best_inlier_ratio;
    }
  }
  if (best_score)
    *best_score = best_cost;
  return best_model;
}

} // namespace libmv

#endif  // LIBMV_MULTIVIEW_ROBUST_ESTIMATION_H_
