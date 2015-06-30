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
// Class for loading the data required for descibing a Fields of Experts (FoE)
// model. The Fields of Experts regularization consists of terms of the type
//
//   alpha * log(1 + (1/2)*sum(F .* X)^2),
//
// where F is a d-by-d image patch and alpha is a constant. This is implemented
// by a FieldsOfExpertsSum object which represents the dot product between the
// image patches and a FieldsOfExpertsLoss which implements the log(1 + (1/2)s)
// part.
//
// [1] S. Roth and M.J. Black. "Fields of Experts." International Journal of
//     Computer Vision, 82(2):205--229, 2009.

#ifndef CERES_EXAMPLES_FIELDS_OF_EXPERTS_H_
#define CERES_EXAMPLES_FIELDS_OF_EXPERTS_H_

#include <iostream>
#include <vector>

#include "ceres/loss_function.h"
#include "ceres/cost_function.h"
#include "ceres/sized_cost_function.h"

#include "pgm_image.h"

namespace ceres {
namespace examples {

// One sum in the FoE regularizer. This is a dot product between a filter and an
// image patch. It simply calculates the dot product between the filter
// coefficients given in the constructor and the scalar parameters passed to it.
class FieldsOfExpertsCost : public ceres::CostFunction {
 public:
  explicit FieldsOfExpertsCost(const std::vector<double>& filter);
  // The number of scalar parameters passed to Evaluate must equal the number of
  // filter coefficients passed to the constructor.
  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const;

 private:
  const std::vector<double>& filter_;
};

// The loss function used to build the correct regularization. See above.
//
//   f(x) = alpha_i * log(1 + (1/2)s)
//
class FieldsOfExpertsLoss : public ceres::LossFunction {
 public:
  explicit FieldsOfExpertsLoss(double alpha) : alpha_(alpha) { }
  virtual void Evaluate(double, double*) const;

 private:
  const double alpha_;
};

// This class loads a set of filters and coefficients from file. Then the users
// obtains the correct loss and cost functions through NewCostFunction and
// NewLossFunction.
class FieldsOfExperts {
 public:
  // Creates an empty object with size() == 0.
  FieldsOfExperts();
  // Attempts to load filters from a file. If unsuccessful it returns false and
  // sets size() == 0.
  bool LoadFromFile(const std::string& filename);

  // Side length of a square filter in this FoE. They are all of the same size.
  int Size() const {
    return size_;
  }

  // Total number of pixels the filter covers.
  int NumVariables() const {
    return size_ * size_;
  }

  // Number of filters used by the FoE.
  int NumFilters() const {
    return num_filters_;
  }

  // Creates a new cost function. The caller is responsible for deallocating the
  // memory. alpha_index specifies which filter is used in the cost function.
  ceres::CostFunction* NewCostFunction(int alpha_index) const;
  // Creates a new loss function. The caller is responsible for deallocating the
  // memory. alpha_index specifies which filter this loss function is for.
  ceres::LossFunction* NewLossFunction(int alpha_index) const;

  // Gets the delta pixel indices for all pixels in a patch.
  const std::vector<int>& GetXDeltaIndices() const {
    return x_delta_indices_;
  }
  const std::vector<int>& GetYDeltaIndices() const {
    return y_delta_indices_;
  }

 private:
  // The side length of a square filter.
  int size_;
  // The number of different filters used.
  int num_filters_;
  // Pixel offsets for all variables.
  std::vector<int> x_delta_indices_, y_delta_indices_;
  // The coefficients in front of each term.
  std::vector<double> alpha_;
  // The filters used for the dot product with image patches.
  std::vector<std::vector<double> > filters_;
};

}  // namespace examples
}  // namespace ceres

#endif  // CERES_EXAMPLES_FIELDS_OF_EXPERTS_H_
