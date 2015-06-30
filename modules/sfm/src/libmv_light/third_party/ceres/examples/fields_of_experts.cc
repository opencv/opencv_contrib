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
// model.

#include "fields_of_experts.h"

#include <fstream>
#include <cmath>

#include "pgm_image.h"

namespace ceres {
namespace examples {

FieldsOfExpertsCost::FieldsOfExpertsCost(const std::vector<double>& filter)
    : filter_(filter) {
  set_num_residuals(1);
  for (int i = 0; i < filter_.size(); ++i) {
    mutable_parameter_block_sizes()->push_back(1);
  }
}

// This is a dot product between a the scalar parameters and a vector of filter
// coefficients.
bool FieldsOfExpertsCost::Evaluate(double const* const* parameters,
                                   double* residuals,
                                   double** jacobians) const {
  int num_variables = filter_.size();
  residuals[0] = 0;
  for (int i = 0; i < num_variables; ++i) {
    residuals[0] += filter_[i] * parameters[i][0];
  }

  if (jacobians != NULL) {
    for (int i = 0; i < num_variables; ++i) {
      if (jacobians[i] != NULL) {
        jacobians[i][0] = filter_[i];
      }
    }
  }

  return true;
}

// This loss function builds the FoE terms and is equal to
//
//   f(x) = alpha_i * log(1 + (1/2)s)
//
void FieldsOfExpertsLoss::Evaluate(double sq_norm, double rho[3]) const {
  const double c = 0.5;
  const double sum = 1.0 + sq_norm * c;
  const double inv = 1.0 / sum;
  // 'sum' and 'inv' are always positive, assuming that 's' is.
  rho[0] = alpha_ *  log(sum);
  rho[1] = alpha_ * c * inv;
  rho[2] = - alpha_ * c * c * inv * inv;
}

FieldsOfExperts::FieldsOfExperts()
    :  size_(0), num_filters_(0) {
}

bool FieldsOfExperts::LoadFromFile(const std::string& filename) {
  std::ifstream foe_file(filename.c_str());
  foe_file >> size_;
  foe_file >> num_filters_;
  if (size_ < 0 || num_filters_ < 0) {
    return false;
  }
  const int num_variables = NumVariables();

  x_delta_indices_.resize(num_variables);
  for (int i = 0; i < num_variables; ++i) {
    foe_file >> x_delta_indices_[i];
  }

  y_delta_indices_.resize(NumVariables());
  for (int i = 0; i < num_variables; ++i) {
    foe_file >> y_delta_indices_[i];
  }

  alpha_.resize(num_filters_);
  for (int i = 0; i < num_filters_; ++i) {
    foe_file >> alpha_[i];
  }

  filters_.resize(num_filters_);
  for (int i = 0; i < num_filters_; ++i) {
    filters_[i].resize(num_variables);
    for (int j = 0; j < num_variables; ++j) {
      foe_file >> filters_[i][j];
    }
  }

  // If any read failed, return failure.
  if (!foe_file) {
    size_ = 0;
    return false;
  }

  // There cannot be anything else in the file. Try reading another number and
  // return failure if that succeeded.
  double temp;
  foe_file >> temp;
  if (foe_file) {
    size_ = 0;
    return false;
  }

  return true;
}

ceres::CostFunction* FieldsOfExperts::NewCostFunction(int alpha_index) const {
  return new FieldsOfExpertsCost(filters_[alpha_index]);
}

ceres::LossFunction* FieldsOfExperts::NewLossFunction(int alpha_index) const {
  return new FieldsOfExpertsLoss(alpha_[alpha_index]);
}


}  // namespace examples
}  // namespace ceres
