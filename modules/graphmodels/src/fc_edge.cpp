#include "fc_edge.hpp"
#include <iostream>

namespace cv
{
namespace graphmodels
{

FCEdge::FCEdge(const config::Edge& edge_config) :
  EdgeWithWeight(edge_config){}

void FCEdge::AllocateMemoryBprop() {
  int input_size = image_size_ * image_size_ * num_input_channels_;
  grad_weights_.AllocateGPUMemory(num_output_channels_, input_size, GetName() + "_grad_weight");
  weight_optimizer_->AllocateMemory(num_output_channels_, input_size);

  if (!has_no_bias_) {
    grad_bias_.AllocateGPUMemory(1, num_output_channels_, GetName() + "_grad_bias");
    bias_optimizer_->AllocateMemory(1, num_output_channels_);
  }
}

void FCEdge::AllocateMemoryFprop() {
  int input_size = image_size_ * image_size_ * num_input_channels_;
  weights_.AllocateGPUMemory(num_output_channels_, input_size, GetName() + "_weight");
  if (!has_no_bias_) {
    bias_.AllocateGPUMemory(1, num_output_channels_, GetName() + "_bias");
  }
}

void FCEdge::AllocateMemory(bool fprop_only) {
  if (is_tied_) return;
  Edge::AllocateMemory(fprop_only);
  cout << name_ << " ";
  printf("Fully connected : %d-%d-%d (%d) : %d\n", image_size_, image_size_,
         num_input_channels_, image_size_ * image_size_ * num_input_channels_,
         num_output_channels_);

  AllocateMemoryFprop();
  if (!fprop_only) AllocateMemoryBprop();
}

void FCEdge::ComputeUp(Matrix& input, Matrix& output, bool overwrite) {
  Matrix& w = is_tied_? tied_edge_->GetWeight() : weights_;
  int scale_targets = overwrite ? 0 : 1;
  Matrix::Dot(input, w, output, scale_targets, 1, false, true);

  if (!has_no_bias_) {
    Matrix& b = is_tied_? tied_edge_->GetBias() : bias_;
    output.AddRowVec(b);
  }
}

void FCEdge::ComputeDown(Matrix& deriv_output, Matrix& input,
                         Matrix& output, Matrix& deriv_input, bool overwrite) {
  Matrix& w = is_tied_? tied_edge_->GetWeight() : weights_;
  int scale_targets = overwrite ? 0 : 1;
  Matrix::Dot(deriv_output, w, deriv_input, scale_targets, 1);
}

void FCEdge::ComputeOuter(Matrix& input, Matrix& deriv_output) {
  Matrix& dw = is_tied_ ? tied_edge_->GetGradWeight() : grad_weights_;
  int scale_targets = GetNumGradsReceived() > 0 ? 1 : 0;
  const int batch_size = input.GetRows();

  Matrix::Dot(deriv_output, input, dw, scale_targets, scale_gradients_ / batch_size, true, false);

  if (!has_no_bias_) {
    Matrix& db = is_tied_ ? tied_edge_->GetGradBias() : grad_bias_;
    deriv_output.SumRows(db, scale_targets, scale_gradients_ / batch_size);
  }
  IncrementNumGradsReceived();
}

}
}
