#include "opencv2/graphmodels/layer.hpp"
#include <iostream>
#include <sstream>
#include <set>
using namespace std;

namespace cv
{
namespace graphmodels
{

Layer* Layer::ChooseLayerClass(const config::Layer& config) {
  Layer* l = NULL;
  switch (config.activation()) {
    case config::Layer::LINEAR :
      l = new LinearLayer(config);
      break;
    case config::Layer::LOGISTIC :
      l = new LogisticLayer(config);
      break;
    case config::Layer::RECTIFIED_LINEAR :
      l = new ReLULayer(config);
      break;
    case config::Layer::SOFTMAX :
      l = new SoftmaxLayer(config);
      break;
    case config::Layer::SOFTMAX_DIST :
      l = new SoftmaxDistLayer(config);
      break;
    case config::Layer::HINGE_LINEAR :
      l = new HingeLinearLayer(config);
      break;
    case config::Layer::HINGE_QUADRATIC :
      l = new HingeQuadraticLayer(config);
      break;
    default:
      cerr << "Undefined layer type." << endl;
      exit(1);
  }
  return l;
}

Layer::Layer(const config::Layer& config) :
  has_incoming_from_same_gpu_(false),
  has_outgoing_to_same_gpu_(false),
  has_incoming_from_other_gpus_(false),
  has_outgoing_to_other_gpus_(false),
  name_(config.name()),
  num_channels_(config.num_channels()),
  is_input_(true),
  is_output_(true),
  dropprob_(config.dropprob()),
  display_(config.display()),
  dropout_scale_up_at_train_time_(true),
  gaussian_dropout_(config.gaussian_dropout()),
  max_act_gaussian_dropout_(config.max_act_gaussian_dropout()),
  scale_targets_(0),
  image_size_(0),
  img_display_(NULL),
  gpu_id_(config.gpu_id()){}

Layer:: ~Layer() {
  if (img_display_ != NULL) delete img_display_;
}

void Layer::AddIncoming(Edge* e) {
  is_input_ = false;
  incoming_edge_.push_back(e);
  int edge_gpu_id = e->GetGPUId();
  if (edge_gpu_id != gpu_id_) {
    other_incoming_gpu_ids_.insert(edge_gpu_id);
    has_incoming_from_other_gpus_ = true;
  } else {
    has_incoming_from_same_gpu_ = true;
  }
}

void Layer::AddOutgoing(Edge* e) {
  is_output_ = false;
  outgoing_edge_.push_back(e);
  int edge_gpu_id = e->GetGPUId();
  if (edge_gpu_id != gpu_id_) {
    other_outgoing_gpu_ids_.insert(edge_gpu_id);
    has_outgoing_to_other_gpus_ = true;
  } else {
    has_outgoing_to_same_gpu_ = true;
  }
}

void Layer::AllocateMemoryOnOtherGPUs() {
  set<int> other_gpu_ids = other_incoming_gpu_ids_;
  other_gpu_ids.insert(other_outgoing_gpu_ids_.begin(),
                       other_outgoing_gpu_ids_.end());

  for (int gpu_id : other_gpu_ids) {
    Matrix::SetDevice(gpu_id);
    other_states_[gpu_id].AllocateGPUMemory(state_.GetRows(), state_.GetCols(), GetName() + " other state");
    other_derivs_[gpu_id].AllocateGPUMemory(deriv_.GetRows(), deriv_.GetCols(), GetName() + " other deriv");
    state_copied_[gpu_id] = false;
    deriv_copied_[gpu_id] = false;
  }
}

Matrix& Layer::GetOtherState(int gpu_id) {
  map<int, Matrix>::iterator it;
  it = other_states_.find(gpu_id);
  if (it == other_states_.end()) {
    cerr << "Other state not found on gpu " << gpu_id << endl;
    exit(1);
  }
  return it->second;
}
Matrix& Layer::GetOtherDeriv(int gpu_id) {
  map<int, Matrix>::iterator it;
  it = other_derivs_.find(gpu_id);
  if (it == other_derivs_.end()) {
    cerr << "Other deriv not found on gpu " << gpu_id << endl;
    exit(1);
  }
  return it->second;
}

/** Add up the state from all GPUs.*/
void Layer::AccumulateState() {
  bool overwrite = !has_incoming_from_same_gpu_;
  for (int gpu_id : other_incoming_gpu_ids_) {
    Matrix& other = GetOtherState(gpu_id);
    other.WaitTillReady();  // dst->SetReady after ComputeUp.
    if (overwrite) {
      state_.Set(other);
    } else {
      state_.Add(other);
    }
    overwrite = false;
  }
}

void Layer::AccumulateDeriv() {
  bool overwrite = !has_outgoing_to_same_gpu_;
  for (int gpu_id : other_outgoing_gpu_ids_) {
    Matrix& other = GetOtherDeriv(gpu_id);
    other.WaitTillReady();  // setready after computedown.
    if (overwrite) {
      deriv_.Set(other);
    } else {
      deriv_.Add(other);
    }
    overwrite = false;
  }
}

void Layer::BroadcastState() {
  if (has_outgoing_to_other_gpus_) {
    for (int gpu_id: other_outgoing_gpu_ids_) {
      CopyStateToGPU(gpu_id);
    }
  }
}

void Layer::ResetStateCopies() {
  for (int gpu_id: other_incoming_gpu_ids_) state_copied_[gpu_id] = false;
  for (int gpu_id: other_outgoing_gpu_ids_) state_copied_[gpu_id] = false;
}

void Layer::ResetDerivCopies() {
  for (int gpu_id: other_incoming_gpu_ids_) deriv_copied_[gpu_id] = false;
  for (int gpu_id: other_outgoing_gpu_ids_) deriv_copied_[gpu_id] = false;
}

void Layer::CopyStateToGPU(int dest_gpu) {
  if (!state_copied_[dest_gpu]) {
    Matrix::SetDevice(dest_gpu);
    state_.WaitTillReady();  // wait for l->GetState().SetReady() after ApplyActivation.
    //GetOtherState(gpu_id).CopyP2PAsync(state_);
    GetOtherState(dest_gpu).Set(state_);
    state_copied_[dest_gpu] = true;
  }
}

void Layer::BroadcastDeriv() {
  if (has_incoming_from_other_gpus_) {
    for (int gpu_id: other_incoming_gpu_ids_) {
      CopyDerivToGPU(gpu_id);
    }
  }
}

void Layer::CopyDerivToGPU(int dest_gpu) {
  if (!deriv_copied_[dest_gpu]) {
    Matrix::SetDevice(dest_gpu);
    deriv_.WaitTillReady();  // wait for l->GetDeriv().SetReady() after ApplyDerivativeofActivation.
    //GetOtherDeriv(dest_gpu).CopyP2PAsync(deriv_);
    GetOtherDeriv(dest_gpu).Set(deriv_);
    deriv_copied_[dest_gpu] = true;
  }
}

void Layer::SetSize(int image_size) {
  image_size_ = image_size;
  if (display_) {
    if (num_channels_ == 3) {
      img_display_ = new ImageDisplayer(image_size_, image_size_, num_channels_, false, name_);
    } else {
      img_display_ = new ImageDisplayer(image_size_, image_size_, num_channels_, true, name_);
    }
  }
}

void Layer::AllocateMemory(int batch_size) {
  const int num_pixels = image_size_ * image_size_;
  Matrix::SetDevice(gpu_id_);
  state_.AllocateGPUMemory(batch_size, num_pixels * num_channels_, GetName() + " state");
  deriv_.AllocateGPUMemory(batch_size, num_pixels * num_channels_, GetName() + " deriv");
  if (gaussian_dropout_) {
    rand_gaussian_.AllocateGPUMemory(batch_size, num_pixels * num_channels_);
  }
  AllocateMemoryOnOtherGPUs();
  Matrix::SetDevice(gpu_id_);
}

void Layer::ApplyDropoutAtTrainTime() {
  if (dropprob_ > 0) {
    if (gaussian_dropout_) {
      rand_gaussian_.FillWithRandn();
      rand_gaussian_.Mult(dropprob_);
      rand_gaussian_.Add(1);
      state_.Mult(rand_gaussian_);
      if (max_act_gaussian_dropout_ > 0) {
        // Clip the activations so that |act| <= max_act_gaussian_dropout_
        state_.UpperBoundMod(max_act_gaussian_dropout_);
      }
    } else {  //  Standard binary dropout.
      float scale = dropout_scale_up_at_train_time_ ?
                    (1.0 / (1 - dropprob_)) : 1.0;
      state_.Dropout(dropprob_, 0, scale);
    }
  }
}

void Layer::ApplyDerivativeofDropout() {
  if (dropprob_ > 0) {
    if (gaussian_dropout_) {
      deriv_.Mult(rand_gaussian_);
      // The real state must be used for backproping through the non linearity.
      // The gradient for the layer above has already been computed.
      // Undo dropout.
      state_.Divide(rand_gaussian_);
    } else if (dropout_scale_up_at_train_time_) {
      deriv_.Mult(1. / (1 - dropprob_));
    }
  }
}

void Layer::ApplyDropoutAtTestTime() {
  if (dropprob_ > 0) {
    // Scale down.
    if (!dropout_scale_up_at_train_time_ && !gaussian_dropout_) {
      state_.Mult(1 - dropprob_);
    }
  }
}

float Layer::GetLoss2() {
  return GetLoss();
}

void Layer::Display() {
  Display(0);
}

void Layer::Display(int image_id) {
  if (img_display_ != NULL && display_) {
    state_.CopyToHost();
    img_display_->DisplayImage(state_.GetHostData(), state_.GetRows(), image_id);
    //copy_to_host(&deriv_);
    //img_display->DisplayImage(deriv_.data_host, deriv_.size[0], image_id);
  }
}

void Layer::ApplyDropout(bool train) {
  if (train) {
    ApplyDropoutAtTrainTime();
  } else {
    ApplyDropoutAtTestTime();
  }
}

LinearLayer::LinearLayer(const config::Layer& config) : Layer(config)
{}

void LinearLayer::ApplyActivation(bool train) {
  // Linear layer, do nothing.
  ApplyDropout(train);
}

void LinearLayer::ApplyDerivativeOfActivation() {
  ApplyDerivativeofDropout();
}

void LinearLayer::ComputeDeriv() {
  state_.Subtract(data_, deriv_);
}

float LinearLayer::GetLoss() {
  Matrix temp;
  Matrix::GetTemp(data_.GetRows(), data_.GetCols(), temp);
  state_.Subtract(data_, temp);
  float norm = temp.EuclidNorm();
  float res = 0.5 * norm * norm;
  return res;
}

void LinearLayer::AllocateMemory(int batch_size) {
  Layer::AllocateMemory(batch_size);
  const int num_pixels = image_size_ * image_size_;
  if (is_output_) data_.AllocateGPUMemory(batch_size, num_pixels * num_channels_);
  //Matrix::RegisterTempMemory(batch_size * num_channels_ * num_pixels); why did I have this?
}

ReLULayer::ReLULayer(const config::Layer& config) :
  LinearLayer(config), rectify_after_gaussian_dropout_(false)
{}

void ReLULayer::ApplyActivation(bool train) {
  state_.LowerBound(0);
  ApplyDropout(train);
  if (gaussian_dropout_ && rectify_after_gaussian_dropout_) {
    state_.LowerBound(0);
  }
}

void ReLULayer::ApplyDerivativeOfActivation() {
  ApplyDerivativeofDropout();
  deriv_.ApplyDerivativeOfReLU(state_);
}

void SoftmaxLayer::AllocateMemory(int batch_size) {
  Layer::AllocateMemory(batch_size);
  if (is_output_) data_.AllocateGPUMemory(batch_size, 1);
  Matrix::RegisterTempMemory(batch_size);
}

void SoftmaxLayer::ApplyActivation(bool train) {
  state_.ApplySoftmax();
  ApplyDropout(train);
}

void SoftmaxLayer::ApplyDerivativeOfActivation() {
  cerr << "Back prop through softmax is not implemented." << endl;
  exit(1);
}

void SoftmaxLayer::ComputeDeriv() {
  Matrix::SoftmaxCEDeriv(state_, data_, deriv_);
}

float SoftmaxLayer::GetLoss() {
  Matrix temp;
  Matrix::GetTemp(data_.GetRows(), 1, temp);
  Matrix::SoftmaxCorrect(state_, data_, temp);
  float res = temp.Sum();
  return res;
}

float SoftmaxLayer::GetLoss2() {
  Matrix temp;
  Matrix::GetTemp(data_.GetRows(), 1, temp);
  Matrix::SoftmaxCE(state_, data_, temp);
  float res = temp.Sum();
  return res;
}

void SoftmaxDistLayer::AllocateMemory(int batch_size) {
  Layer::AllocateMemory(batch_size);
  const int numdims = state_.GetCols();
  Matrix::RegisterTempMemory(batch_size * numdims);  // For computing CE.
  if (is_output_) data_.AllocateGPUMemory(batch_size, numdims);
}

void SoftmaxDistLayer::ComputeDeriv() {
  state_.Subtract(data_, deriv_);
}

float SoftmaxDistLayer::GetLoss() {
  Matrix temp;
  Matrix::GetTemp(data_.GetRows(), data_.GetCols(), temp);
  Matrix::SoftmaxDistCE(state_, data_, temp);
  return temp.Sum();
}

void LogisticLayer::AllocateMemory(int batch_size) {
  Layer::AllocateMemory(batch_size);
  Matrix::RegisterTempMemory(batch_size);
  if (is_output_) data_.AllocateGPUMemory(batch_size, num_channels_);
}

void LogisticLayer::ApplyActivation(bool train) {
  state_.ApplyLogistic();
  ApplyDropout(train);
}

void LogisticLayer::ApplyDerivativeOfActivation() {
  ApplyDerivativeofDropout();
  deriv_.ApplyDerivativeOfLogistic(state_);
}

void LogisticLayer::ComputeDeriv() {
  Matrix::LogisticCEDeriv(state_, data_, deriv_);
}

float LogisticLayer::GetLoss() {
  Matrix temp;
  Matrix::GetTemp(data_.GetRows(), 1, temp);
  Matrix::LogisticCorrect(state_, data_, temp);
  return temp.Sum();
}

HingeQuadraticLayer::HingeQuadraticLayer(const config::Layer& config) :
  SoftmaxLayer(config), margin_(config.hinge_margin()) {}

void HingeQuadraticLayer::ApplyActivation(bool train) {
  ApplyDropout(train);
}

void HingeQuadraticLayer::ApplyDerivativeOfActivation() {
  ApplyDerivativeofDropout();
}

void HingeQuadraticLayer::ComputeDeriv() {
  Matrix::HingeLossDeriv(state_, data_, deriv_, true, margin_);
}

HingeLinearLayer::HingeLinearLayer(const config::Layer& config) :
  HingeQuadraticLayer(config) {}

void HingeLinearLayer::ComputeDeriv() {
  Matrix::HingeLossDeriv(state_, data_, deriv_, false, margin_);
}

}
}
