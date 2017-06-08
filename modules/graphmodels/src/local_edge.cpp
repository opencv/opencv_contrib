#include "local_edge.hpp"
#include <iostream>

namespace cv
{
namespace graphmodels
{

LocalEdge::LocalEdge(const config::Edge& edge_config) :
  EdgeWithWeight(edge_config),
  kernel_size_(edge_config.kernel_size()),
  stride_(edge_config.stride()),
  padding_(edge_config.padding()) {}

void LocalEdge::SetTiedTo(Edge* e) {
  EdgeWithWeight::SetTiedTo(e);
  LocalEdge* ee = dynamic_cast<LocalEdge*> (e);
  kernel_size_ = ee->GetKernelSize();
  stride_ = ee->GetStride();
  padding_ = ee->GetPadding();
}

void LocalEdge::DisplayWeights() {
  if (img_display_ != NULL) {
    weights_.CopyToHost();
    img_display_->DisplayWeights(weights_.GetHostData(), kernel_size_, num_output_channels_, 250, false);
  }
}

void LocalEdge::SetImageSize(int image_size) {
  Edge::SetImageSize(image_size);
  num_modules_ = (image_size + 2 * padding_ - kernel_size_) / stride_ + 1;
}

void LocalEdge::FOV(int* size, int* sep, int* pad1, int* pad2) const {
  *size = kernel_size_ + stride_ * ((*size) - 1);
  *sep = (*sep) * stride_;
  *pad1 = (*pad1) * stride_ + padding_;
  int k = (image_size_ + 2*padding_ - kernel_size_) / stride_;
  int effective_right_pad = k * stride_ - (image_size_ + padding_ - kernel_size_);
  *pad2 = (*pad2) * stride_ + effective_right_pad;
}

void LocalEdge::AllocateMemoryFprop() {
  int input_size = kernel_size_ * kernel_size_ * num_input_channels_
                   * num_modules_ * num_modules_;
  int bias_locs = num_modules_ * num_modules_;
  weights_.AllocateGPUMemory(num_output_channels_, input_size);
  bias_.AllocateGPUMemory(1, num_output_channels_ * bias_locs);

}

void LocalEdge::AllocateMemoryBprop() {
  int input_size = kernel_size_ * kernel_size_ * num_input_channels_
                   * num_modules_ * num_modules_;
  int bias_locs = num_modules_ * num_modules_;
  // Matrix for storing the current gradient.
  grad_weights_.AllocateGPUMemory(num_output_channels_, input_size);

  weight_optimizer_->AllocateMemory(num_output_channels_, input_size);

  grad_bias_.AllocateGPUMemory(1, num_output_channels_ * bias_locs);
  bias_optimizer_->AllocateMemory(1, num_output_channels_ * bias_locs);
}

void LocalEdge::AllocateMemory(bool fprop_only) {
  if (is_tied_) return;
  Edge::AllocateMemory(fprop_only);
  
  num_modules_ = (image_size_ + 2 * padding_ - kernel_size_) / stride_ + 1;

  cout << name_ << " ";
  printf("Kernel: %d-%d-%d to %d ", kernel_size_, kernel_size_,
         num_input_channels_, num_output_channels_);
  printf("Layer: %d-%d-%d (%d) ", image_size_, image_size_, num_input_channels_,
         image_size_ * image_size_ * num_input_channels_);
  
  AllocateMemoryFprop();
  if (!fprop_only) AllocateMemoryBprop();

  cout << " Allocated weight " << weights_.GetRows() << " " << weights_.GetCols()
       << " Locally Connected" << endl;
  if (num_input_channels_ == 3) {
    int num_filters = num_output_channels_;
    int num_filters_w = int(sqrt(num_filters));
    int num_filters_h = num_filters / num_filters_w +  (((num_filters % num_filters_w) > 0) ? 1 : 0);
    int width = 250;
    int height = (width * num_filters_h) / num_filters_w;
    img_display_ = new ImageDisplayer(width, height, 3, false, "weights");
  }
}

void LocalEdge::ComputeUp(Matrix& input, Matrix& output, bool overwrite) {
  Matrix& w = is_tied_? tied_edge_->GetWeight() : weights_;
  int scale_targets = overwrite ? 0 : 1;
  Matrix::LocalUp(input, w, output, image_size_, num_modules_, num_modules_,
                  padding_, stride_, num_input_channels_, scale_targets);
  if (!has_no_bias_) {
    Matrix& b = is_tied_? tied_edge_->GetBias() : bias_;
    output.AddRowVec(b);
  }
}

void LocalEdge::ComputeDown(Matrix& deriv_output, Matrix& input,
                           Matrix& output, Matrix& deriv_input, bool overwrite) {
  Matrix& w = is_tied_? tied_edge_->GetWeight() : weights_;
  int scale_targets = overwrite ? 0 : 1;
  Matrix::LocalDown(deriv_output, w, deriv_input, image_size_, image_size_,
                    num_modules_, padding_, stride_, num_input_channels_,
                    scale_targets);
}

void LocalEdge::ComputeOuter(Matrix& input, Matrix& deriv_output) {
  Matrix& dw = is_tied_? tied_edge_->GetGradWeight() : grad_weights_;
  int batch_size = input.GetRows();
  int scale_targets = GetNumGradsReceived() > 0 ? 1 : 0;

  Matrix::LocalOutp(input, deriv_output, dw, image_size_, num_modules_,
                    num_modules_, kernel_size_, padding_, stride_,
                    num_input_channels_, scale_targets,
                    scale_gradients_ / batch_size);

  if (!has_no_bias_) {
    Matrix& db = is_tied_ ? tied_edge_->GetGradBias() : grad_bias_;
    deriv_output.SumRows(db, scale_targets, scale_gradients_ / batch_size);
  }

  IncrementNumGradsReceived();
}

}
}
