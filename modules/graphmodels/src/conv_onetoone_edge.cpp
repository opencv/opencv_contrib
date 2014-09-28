#include "conv_onetoone_edge.hpp"
#include <iostream>

namespace cv
{
namespace graphmodels
{

ConvOneToOneEdge::ConvOneToOneEdge(const config::Edge& edge_config) :
  EdgeWithWeight(edge_config) {}

void ConvOneToOneEdge::SetImageSize(int image_size) {
  Edge::SetImageSize(image_size);
  num_modules_ = image_size;
}

void ConvOneToOneEdge::AllocateMemory(bool fprop_only) {
  if (is_tied_) return;
  Edge::AllocateMemory(fprop_only);

  cout << name_ << " ";
  printf("One to one convolution: %d - %d ", num_input_channels_, num_output_channels_);
  printf("Layer: %d-%d-%d (%d) ", image_size_, image_size_, num_input_channels_,
         image_size_ * image_size_ * num_input_channels_);
 
  AllocateMemoryFprop();
  if (!fprop_only) AllocateMemoryBprop();

  cout << " Allocated weight " << weights_.GetRows() << " " << weights_.GetCols()
       << " One to One Convolutional" << endl;
}


void ConvOneToOneEdge::AllocateMemoryBprop() {
  if (!is_tied_) {
    grad_weights_.AllocateGPUMemory(num_output_channels_, num_input_channels_);
    weight_optimizer_->AllocateMemory(num_output_channels_, num_input_channels_);
  }

  if (!has_no_bias_ && !is_tied_) {
    grad_bias_.AllocateGPUMemory(1, num_output_channels_);
    bias_optimizer_->AllocateMemory(1, num_output_channels_);
  }
}

void ConvOneToOneEdge::AllocateMemoryFprop() {
  weights_.AllocateGPUMemory(num_output_channels_, num_input_channels_);
  if (!has_no_bias_) {
    bias_.AllocateGPUMemory(1, num_output_channels_);
  }
}

void ConvOneToOneEdge::ComputeUp(Matrix& input, Matrix& output, bool overwrite) {
  int batch_size = input.GetRows();
  input.Reshape(-1, num_input_channels_);
  output.Reshape(-1, num_output_channels_);

  Matrix& w = is_tied_? tied_edge_->GetWeight() : weights_;
  int scale_targets = overwrite ? 0 : 1;
  Matrix::Dot(input, w, output, scale_targets, 1, false, true);

  if (!has_no_bias_) {
    Matrix& b = is_tied_? tied_edge_->GetBias() : bias_;
    output.AddRowVec(b);
  }

  input.Reshape(batch_size, -1);
  output.Reshape(batch_size, -1);
}

void ConvOneToOneEdge::ComputeDown(Matrix& deriv_output, Matrix& input,
                           Matrix& output, Matrix& deriv_input, bool overwrite) {
  int batch_size = input.GetRows();
  deriv_output.Reshape(-1, num_output_channels_);
  deriv_input.Reshape(-1, num_input_channels_);
  Matrix& w = is_tied_? tied_edge_->GetWeight() : weights_;
  int scale_targets = overwrite ? 0 : 1;
  Matrix::Dot(deriv_output, w, deriv_input, scale_targets, 1);
  deriv_output.Reshape(batch_size, -1);
  deriv_input.Reshape(batch_size, -1);
}

void ConvOneToOneEdge::ComputeOuter(Matrix& input, Matrix& deriv_output) {
  int batch_size = input.GetRows();
  input.Reshape(-1, num_input_channels_);
  deriv_output.Reshape(-1, num_output_channels_);
  
  Matrix& dw = is_tied_ ? tied_edge_->GetGradWeight() : grad_weights_;
  int scale_targets = GetNumGradsReceived() > 0 ? 1 : 0;
  Matrix::Dot(deriv_output, input, dw, scale_targets, scale_gradients_ / batch_size, true, false);

  if (!has_no_bias_) {
    Matrix& db = is_tied_ ? tied_edge_->GetGradBias() : grad_bias_;
    deriv_output.SumRows(db, scale_targets, scale_gradients_ / batch_size);
  }
  input.Reshape(batch_size, -1);
  deriv_output.Reshape(batch_size, -1);
  IncrementNumGradsReceived();
}

}
}
