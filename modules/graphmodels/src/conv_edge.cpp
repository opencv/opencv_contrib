#include "conv_edge.hpp"
#include <iostream>

namespace cv
{
namespace graphmodels
{

ConvEdge::ConvEdge(const config::Edge& edge_config) :
  EdgeWithWeight(edge_config),
  kernel_size_(edge_config.kernel_size()),
  stride_(edge_config.stride()),
  padding_(edge_config.padding()),
  partial_sum_(edge_config.partial_sum()),
  shared_bias_(edge_config.shared_bias()) {}

void ConvEdge::SetTiedTo(Edge* e) {
  EdgeWithWeight::SetTiedTo(e);
  ConvEdge* ee = dynamic_cast<ConvEdge*>(e);
  kernel_size_ = ee->GetKernelSize();
  stride_ = ee->GetStride();
  padding_ = ee->GetPadding();
  if (partial_sum_ == 0) {
    partial_sum_ = ee->GetPartialSum();
  }
  shared_bias_ = ee->GetSharedBias();
}

void ConvEdge::SetImageSize(int image_size) {
  Edge::SetImageSize(image_size);
  num_modules_ = (image_size + 2 * padding_ - kernel_size_) / stride_ + 1;
}

void ConvEdge::FOV(int* size, int* sep, int* pad1, int* pad2) const {
  *size = kernel_size_ + stride_ * ((*size) - 1);
  *sep = (*sep) * stride_;
  *pad1 = (*pad1) * stride_ + padding_;
  int k = (image_size_ + 2*padding_ - kernel_size_) / stride_;
  int effective_right_pad = k * stride_ - (image_size_ + padding_ - kernel_size_);
  *pad2 = (*pad2) * stride_ + effective_right_pad;
}

void ConvEdge::DisplayWeights() {
  if (img_display_ != NULL && display_) {
    weights_.CopyToHost();
    img_display_->DisplayWeights(weights_.GetHostData(), kernel_size_, num_output_channels_, 250, false);
  }
}

void ConvEdge::AllocateMemory(bool fprop_only) {
  fprop_only |= block_backprop_;
  Edge::AllocateMemory(fprop_only);
  if (is_tied_) {
    if (!fprop_only) AllocateMemoryBprop();  // For partial sums.
    return;
  }

  cout << name_ << " ";
  printf("Kernel: %d-%d-%d to %d ", kernel_size_, kernel_size_,
         num_input_channels_, num_output_channels_);
  printf("Layer: %d-%d-%d (%d) ", image_size_, image_size_, num_input_channels_,
         image_size_ * image_size_ * num_input_channels_);
 
  AllocateMemoryFprop();
  if (!fprop_only) AllocateMemoryBprop();

  cout << " Allocated weight " << weights_.GetRows() << " " << weights_.GetCols()
       << " Convolutional" << endl;

  if (num_input_channels_ == 3) {
    int num_filters = num_output_channels_;
    int num_filters_w = int(sqrt(num_filters));
    int num_filters_h = num_filters / num_filters_w + (((num_filters % num_filters_w) > 0) ? 1 : 0);
    int width = 250;
    int height = (width * num_filters_h) / num_filters_w;
    img_display_ = new ImageDisplayer(width, height, 3, false, "weights");
  }

}


void ConvEdge::AllocateMemoryBprop() {
  int input_size = kernel_size_ * kernel_size_ * num_input_channels_;
  int num_locs = num_modules_ * num_modules_;
  int bias_locs = shared_bias_ ? 1 : num_locs;
  // Matrix for storing the current gradient.

  if (!is_tied_) {
    grad_weights_.AllocateGPUMemory(num_output_channels_, input_size);
    weight_optimizer_->AllocateMemory(num_output_channels_, input_size);
  }

  if (partial_sum_ > 0) {
    int partial_sums = DIVUP(num_modules_, partial_sum_) * DIVUP(num_modules_, partial_sum_);
    Matrix::RegisterTempMemory(num_output_channels_ * input_size * partial_sums,
                               "partial sums " + GetName());
    Matrix::RegisterOnes(partial_sums);
  }
 
  if (!has_no_bias_ && !is_tied_) {
    grad_bias_.AllocateGPUMemory(1, num_output_channels_ * bias_locs);
    bias_optimizer_->AllocateMemory(1, num_output_channels_ * bias_locs);
    if (shared_bias_) {
      Matrix::RegisterTempMemory(num_output_channels_ * num_locs, "shared bias");
    }
  }
}

void ConvEdge::AllocateMemoryFprop() {
  int input_size = kernel_size_ * kernel_size_ * num_input_channels_;
  int bias_locs = shared_bias_ ? 1: (num_modules_ * num_modules_);
  
  // Weights for this convolution.
  weights_.AllocateGPUMemory(num_output_channels_, input_size);
  if (!has_no_bias_) {
    bias_.AllocateGPUMemory(1, num_output_channels_ * bias_locs);
  }
}

void ConvEdge::ComputeUp(Matrix& input, Matrix& output, bool overwrite) {
  Matrix& w = is_tied_? tied_edge_->GetWeight() : weights_;
  float scale_targets = overwrite ? 0 : 1;
  Matrix::ConvUp(input, w, output, image_size_, num_modules_, num_modules_,
                 padding_, stride_, num_input_channels_, scale_targets);
  if (!has_no_bias_) {
    Matrix& b = is_tied_? tied_edge_->GetBias() : bias_;
    if (shared_bias_) {
      output.Reshape(-1, num_output_channels_);
      output.AddRowVec(b);
      output.Reshape(-1, num_output_channels_ * num_modules_ * num_modules_);
    } else {
      output.AddRowVec(b);
    }
  }
}

void ConvEdge::ComputeDown(Matrix& deriv_output, Matrix& input,
                           Matrix& output, Matrix& deriv_input, bool overwrite) {
  
  Matrix& w = is_tied_? tied_edge_->GetWeight() : weights_;
  float scale_targets = overwrite ? 0 : 1;
  Matrix::ConvDown(deriv_output, w, deriv_input, image_size_, image_size_,
                   num_modules_, padding_, stride_, num_input_channels_,
                   scale_targets);
}

void ConvEdge::ComputeOuter(Matrix& input, Matrix& deriv_output) {
  Matrix& dw = is_tied_ ? tied_edge_->GetGradWeight() : grad_weights_;
  const int batch_size = input.GetRows();

  int scale_targets = GetNumGradsReceived() > 0 ? 1 : 0;

  if (partial_sum_ > 0) {
    Matrix dw_temp;

    int filter_input_size = num_input_channels_ * kernel_size_ * kernel_size_;
    int partial_sums = DIVUP(num_modules_, partial_sum_) * DIVUP(num_modules_, partial_sum_);
    Matrix::GetTemp(num_output_channels_, filter_input_size * partial_sums, dw_temp);
    
    Matrix::ConvOutp(input, deriv_output, dw_temp, image_size_, num_modules_,
                     num_modules_, kernel_size_, padding_, stride_,
                     num_input_channels_, partial_sum_, 0, 1);

    dw_temp.Reshape(num_output_channels_ * filter_input_size, partial_sums);
    dw.Reshape(-1, 1);
    dw_temp.SumCols(dw, scale_targets, scale_gradients_ / batch_size);
    dw.Reshape(num_output_channels_, filter_input_size);
  } else {
    Matrix::ConvOutp(input, deriv_output, dw, image_size_, num_modules_,
                     num_modules_, kernel_size_, padding_, stride_,
                     num_input_channels_, num_modules_, scale_targets,
                     scale_gradients_ / batch_size);
  }

  if (!has_no_bias_) {
    Matrix& db = is_tied_ ? tied_edge_->GetGradBias() : grad_bias_;
    if (shared_bias_) {
      // 2 step addition is SIGNFICANTLY faster (Why ?)
      Matrix db_temp;
      Matrix::GetTemp(1, deriv_output.GetCols(), db_temp);
      deriv_output.SumRows(db_temp, 0, 1);
      db_temp.Reshape(-1, num_output_channels_);
      db_temp.SumRows(db, scale_targets, scale_gradients_ / batch_size);
    } else {
      deriv_output.SumRows(db, scale_targets, scale_gradients_ / batch_size);
    }
  }
  IncrementNumGradsReceived();
}

}
}
