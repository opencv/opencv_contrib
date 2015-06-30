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
// Author: strandmark@google.com (Petter Strandmark)
//
// Denoising using Fields of Experts and the Ceres minimizer.
//
// Note that for good denoising results the weighting between the data term
// and the Fields of Experts term needs to be adjusted. This is discussed
// in [1]. This program assumes Gaussian noise. The noise model can be changed
// by substituing another function for QuadraticCostFunction.
//
// [1] S. Roth and M.J. Black. "Fields of Experts." International Journal of
//     Computer Vision, 82(2):205--229, 2009.

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include <sstream>
#include <string>

#include "ceres/ceres.h"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "fields_of_experts.h"
#include "pgm_image.h"

DEFINE_string(input, "", "File to which the output image should be written");
DEFINE_string(foe_file, "", "FoE file to use");
DEFINE_string(output, "", "File to which the output image should be written");
DEFINE_double(sigma, 20.0, "Standard deviation of noise");
DEFINE_bool(verbose, false, "Prints information about the solver progress.");
DEFINE_bool(line_search, false, "Use a line search instead of trust region "
            "algorithm.");

namespace ceres {
namespace examples {

// This cost function is used to build the data term.
//
//   f_i(x) = a * (x_i - b)^2
//
class QuadraticCostFunction : public ceres::SizedCostFunction<1, 1> {
 public:
  QuadraticCostFunction(double a, double b)
    : sqrta_(std::sqrt(a)), b_(b) {}
  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    const double x = parameters[0][0];
    residuals[0] = sqrta_ * (x - b_);
    if (jacobians != NULL && jacobians[0] != NULL) {
      jacobians[0][0] = sqrta_;
    }
    return true;
  }
 private:
  double sqrta_, b_;
};

// Creates a Fields of Experts MAP inference problem.
void CreateProblem(const FieldsOfExperts& foe,
                   const PGMImage<double>& image,
                   Problem* problem,
                   PGMImage<double>* solution) {
  // Create the data term
  CHECK_GT(FLAGS_sigma, 0.0);
  const double coefficient = 1 / (2.0 * FLAGS_sigma * FLAGS_sigma);
  for (unsigned index = 0; index < image.NumPixels(); ++index) {
    ceres::CostFunction* cost_function =
        new QuadraticCostFunction(coefficient,
                                  image.PixelFromLinearIndex(index));
    problem->AddResidualBlock(cost_function,
                              NULL,
                              solution->MutablePixelFromLinearIndex(index));
  }

  // Create Ceres cost and loss functions for regularization. One is needed for
  // each filter.
  std::vector<ceres::LossFunction*> loss_function(foe.NumFilters());
  std::vector<ceres::CostFunction*> cost_function(foe.NumFilters());
  for (int alpha_index = 0; alpha_index < foe.NumFilters(); ++alpha_index) {
    loss_function[alpha_index] = foe.NewLossFunction(alpha_index);
    cost_function[alpha_index] = foe.NewCostFunction(alpha_index);
  }

  // Add FoE regularization for each patch in the image.
  for (int x = 0; x < image.width() - (foe.Size() - 1); ++x) {
    for (int y = 0; y < image.height() - (foe.Size() - 1); ++y) {
      // Build a vector with the pixel indices of this patch.
      std::vector<double*> pixels;
      const std::vector<int>& x_delta_indices = foe.GetXDeltaIndices();
      const std::vector<int>& y_delta_indices = foe.GetYDeltaIndices();
      for (int i = 0; i < foe.NumVariables(); ++i) {
        double* pixel = solution->MutablePixel(x + x_delta_indices[i],
                                               y + y_delta_indices[i]);
        pixels.push_back(pixel);
      }
      // For this patch with coordinates (x, y), we will add foe.NumFilters()
      // terms to the objective function.
      for (int alpha_index = 0; alpha_index < foe.NumFilters(); ++alpha_index) {
        problem->AddResidualBlock(cost_function[alpha_index],
                                  loss_function[alpha_index],
                                  pixels);
      }
    }
  }
}

// Solves the FoE problem using Ceres and post-processes it to make sure the
// solution stays within [0, 255].
void SolveProblem(Problem* problem, PGMImage<double>* solution) {
  // These parameters may be experimented with. For example, ceres::DOGLEG tends
  // to be faster for 2x2 filters, but gives solutions with slightly higher
  // objective function value.
  ceres::Solver::Options options;
  options.max_num_iterations = 100;
  if (FLAGS_verbose) {
    options.minimizer_progress_to_stdout = true;
  }

  if (FLAGS_line_search) {
    options.minimizer_type = ceres::LINE_SEARCH;
  }

  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.function_tolerance = 1e-3;  // Enough for denoising.

  ceres::Solver::Summary summary;
  ceres::Solve(options, problem, &summary);
  if (FLAGS_verbose) {
    std::cout << summary.FullReport() << "\n";
  }

  // Make the solution stay in [0, 255].
  for (int x = 0; x < solution->width(); ++x) {
    for (int y = 0; y < solution->height(); ++y) {
      *solution->MutablePixel(x, y) =
          std::min(255.0, std::max(0.0, solution->Pixel(x, y)));
    }
  }
}
}  // namespace examples
}  // namespace ceres

int main(int argc, char** argv) {
  using namespace ceres::examples;
  std::string
      usage("This program denoises an image using Ceres.  Sample usage:\n");
  usage += argv[0];
  usage += " --input=<noisy image PGM file> --foe_file=<FoE file name>";
  google::SetUsageMessage(usage);
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  if (FLAGS_input.empty()) {
    std::cerr << "Please provide an image file name.\n";
    return 1;
  }

  if (FLAGS_foe_file.empty()) {
    std::cerr << "Please provide a Fields of Experts file name.\n";
    return 1;
  }

  // Load the Fields of Experts filters from file.
  FieldsOfExperts foe;
  if (!foe.LoadFromFile(FLAGS_foe_file)) {
    std::cerr << "Loading \"" << FLAGS_foe_file << "\" failed.\n";
    return 2;
  }

  // Read the images
  PGMImage<double> image(FLAGS_input);
  if (image.width() == 0) {
    std::cerr << "Reading \"" << FLAGS_input << "\" failed.\n";
    return 3;
  }
  PGMImage<double> solution(image.width(), image.height());
  solution.Set(0.0);

  ceres::Problem problem;
  CreateProblem(foe, image, &problem, &solution);

  SolveProblem(&problem, &solution);

  if (!FLAGS_output.empty()) {
    CHECK(solution.WriteToFile(FLAGS_output))
        << "Writing \"" << FLAGS_output << "\" failed.";
  }

  return 0;
}
