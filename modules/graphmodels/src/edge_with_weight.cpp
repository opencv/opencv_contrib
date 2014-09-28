#include "edge_with_weight.hpp"
#include <sstream>
#include <iostream>

namespace cv
{
namespace graphmodels
{

EdgeWithWeight::EdgeWithWeight(const config::Edge& edge_config) :
  Edge(edge_config),
  weight_optimizer_(Optimizer::ChooseOptimizer(edge_config.weight_optimizer())),
  bias_optimizer_(Optimizer::ChooseOptimizer(edge_config.bias_optimizer())),
  initialization_(edge_config.initialization()),
  polyak_queue_size_(edge_config.polyak_queue_size()),
  polyak_index_(0),
  polyak_queue_full_(false),
  init_wt_(edge_config.init_wt()),
  init_bias_(edge_config.init_bias()),
  has_no_bias_(edge_config.has_no_bias()),
  num_grads_received_(0),
  num_shares_(1),
  scale_gradients_(edge_config.scale_gradients()),
  pretrained_model_(edge_config.pretrained_model()),
  pretrained_edge_name_(edge_config.has_pretrained_edge_name() ?
                        edge_config.pretrained_edge_name() : name_) {
  polyak_weights_.resize(polyak_queue_size_);
  if (!has_no_bias_) {
    polyak_bias_.resize(polyak_queue_size_);
  }
}

EdgeWithWeight::~EdgeWithWeight() {
  delete weight_optimizer_;
  delete bias_optimizer_;
}

void EdgeWithWeight::SaveParameters(hid_t file) {
  if (is_tied_) return;
  stringstream ss;
  ss << source_node_ << ":" << dest_node_ << ":" << "weight";
  weights_.WriteHDF5(file, ss.str());
  weight_optimizer_->SaveParameters(file, ss.str());
  if (!has_no_bias_) {
    ss.str("");
    ss << source_node_ << ":" << dest_node_ << ":" << "bias";
    bias_.WriteHDF5(file, ss.str());
    bias_optimizer_->SaveParameters(file, ss.str());
  }
}

void EdgeWithWeight::LoadParameters(hid_t file, const string& edge_name) {
  if (is_tied_) return;
  Matrix::SetDevice(gpu_id_);
  stringstream ss;
  ss << edge_name << ":" << "weight";
  cout << "Loading " << ss.str() << endl;
  weights_.ReadHDF5(file, ss.str());
  if (weight_optimizer_->IsAllocated()) {
    weight_optimizer_->LoadParameters(file, ss.str());
  }
  if (!has_no_bias_) {
    ss.str("");
    ss << edge_name << ":" << "bias";
    cout << "Loading " << ss.str() << endl;
    bias_.ReadHDF5(file, ss.str());
    if (bias_optimizer_->IsAllocated()) {
      bias_optimizer_->LoadParameters(file, ss.str());
    }
  }
}

void EdgeWithWeight::LoadParameters(hid_t file) {
  stringstream ss;
  ss << source_node_ << ":" << dest_node_;
  LoadParameters(file, ss.str());
}

void EdgeWithWeight::DisplayWeights() {
  if (img_display_ != NULL && display_) {
    weights_.CopyToHost();
    int kernel_size = (int)sqrt(num_input_channels_);
    img_display_->DisplayWeights(weights_.GetHostData(), kernel_size, num_output_channels_, 250, false);
  }
}

void EdgeWithWeight::DisplayWeightStats() {
  /*
  FILE* pipe = gnuplotpipe_;
  if (pipe == NULL) return;
  fprintf(pipe, "set term wx\n");         // set the terminal
  fprintf(pipe, "plot '-' with lines\n"); // plot type
  for(int i = 0; i < 10; i++)             // loop over the data [0,...,9]
    fprintf(pipe, "%d\n", i);           // data terminated with \n
  fprintf(pipe, "%s\n", "e");             // termination character
  fflush(pipe);                           // flush the pipe
  if (img_display_ != NULL) {
    weights_.CopyToHost();
    img_display_->DisplayWeightStats(weights_.GetHostData(), weights_.GetNumEls());
  }
  */
}

void EdgeWithWeight::ReduceLearningRate(float factor) {
  weight_optimizer_->ReduceLearningRate(factor);
  bias_optimizer_->ReduceLearningRate(factor);
}

void EdgeWithWeight::UpdateWeights() {
  if (is_tied_) {
    tied_edge_->UpdateWeights();
    return;
  }
  if (num_grads_received_ < num_shares_) return;
  num_grads_received_ = 0;
  Matrix::SetDevice(gpu_id_);
  weight_optimizer_->Optimize(grad_weights_, weights_);
  bias_optimizer_->Optimize(grad_bias_, bias_);
}

void EdgeWithWeight::Initialize() {
  Edge::Initialize();
  if (is_tied_) return;
  if (initialization_ == config::Edge::DENSE_GAUSSIAN_SQRT_FAN_IN ||
      initialization_ == config::Edge::DENSE_GAUSSIAN) {
    weights_.FillWithRandn();
    float init_wt = init_wt_;
    if (initialization_ == config::Edge::DENSE_GAUSSIAN_SQRT_FAN_IN) {
      init_wt /= sqrt(weights_.GetCols());
    }
    weights_.Mult(init_wt);
    //cout << "Initialized weight: Dense Gaussian. Initial scale " << GetRMSWeight() << endl;
  } else if (initialization_ == config::Edge::DENSE_UNIFORM_SQRT_FAN_IN ||
             initialization_ == config::Edge::DENSE_UNIFORM) {
    weights_.FillWithRand();
    weights_.Add(-0.5);
    float init_wt = 2 * init_wt_;
    if (initialization_ == config::Edge::DENSE_UNIFORM_SQRT_FAN_IN) {
      init_wt /= sqrt(weights_.GetCols() / 3.0f);
    }
    weights_.Mult(init_wt);
    //cout << "Initialized weight: Dense Uniform. Initial scale " << GetRMSWeight() << endl;
  } else if (initialization_ == config::Edge::CONSTANT) {
    weights_.Set(init_wt_);
  } else if (initialization_ == config::Edge::PRETRAINED) {
    hid_t file = H5Fopen(pretrained_model_.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    LoadParameters(file, pretrained_edge_name_);
    H5Fclose(file);
  } else {
    cerr << "Unknown weight initialization type." << endl;
    exit(1);
  }
  if (initialization_ != config::Edge::PRETRAINED && !has_no_bias_) {
    bias_.Set(init_bias_);
  }
}

float EdgeWithWeight::GetRMSWeight() {
  Matrix temp;
  const int num_hid = weights_.GetRows();
  Matrix::GetTemp(num_hid, 1, temp);
  weights_.SqSumAxis(temp, 1, 1, 0);
  temp.Sqrt();
  float res = temp.Sum() / num_hid;
  return res;
}

bool EdgeWithWeight::HasNoParameters() const {
  return false;
}

int EdgeWithWeight::GetNumModules() const {
  return 1;
}

void EdgeWithWeight::InsertPolyak() {
  if (polyak_queue_size_ == 0) return;
  
  Matrix& w = polyak_weights_[polyak_index_];
  if (w.GetNumEls() == 0) {
    w.AllocateMainMemory(weights_.GetRows(), weights_.GetCols());
  }
  weights_.CopyToHost();
  w.CopyFromMainMemory(weights_);

  if (!has_no_bias_) {
    Matrix& b = polyak_bias_[polyak_index_];
    if (b.GetNumEls() == 0) {
      b.AllocateMainMemory(bias_.GetRows(), bias_.GetCols());
    }
    bias_.CopyToHost();
    b.CopyFromMainMemory(bias_);
  }
  polyak_index_++;
  if (polyak_index_ == polyak_queue_size_) {
    polyak_index_ = 0;
    polyak_queue_full_ = true;
  }
}

void EdgeWithWeight::BackupCurrent() {
  if (weights_backup_.GetNumEls() == 0) {
    weights_backup_.AllocateMainMemory(weights_.GetRows(), weights_.GetCols());
  }
  weights_.CopyToHost();
  weights_backup_.CopyFromMainMemory(weights_);

  if (!has_no_bias_) {
    if (bias_backup_.GetNumEls() == 0) {
      bias_backup_.AllocateMainMemory(bias_.GetRows(), bias_.GetCols());
    }
    bias_.CopyToHost();
    bias_backup_.CopyFromMainMemory(bias_);
  }
}

void EdgeWithWeight::LoadCurrentOnGPU() {
  weights_.CopyFromMainMemory(weights_backup_);
  weights_.CopyToDevice();
  if (!has_no_bias_) {
    bias_.CopyFromMainMemory(bias_backup_);
    bias_.CopyToDevice();
  }
}

void EdgeWithWeight::LoadPolyakOnGPU() {
  if (polyak_queue_size_ == 0) return;
  int max_ind = polyak_queue_full_ ? polyak_queue_size_: polyak_index_;

  float *weight_avg = weights_.GetHostData(), *w;
  for (int j = 0; j < weights_.GetNumEls(); j++) weight_avg[j] = 0;
  for (int i = 0; i < max_ind; i++) {
    w = polyak_weights_[i].GetHostData();
    for (int j = 0; j < weights_.GetNumEls(); j++) weight_avg[j] += w[j];
  }
  for (int j = 0; j < weights_.GetNumEls(); j++) weight_avg[j] /= max_ind;
  weights_.CopyToDevice();

  if (!has_no_bias_) {
    float *bias_avg = bias_.GetHostData(), *b;
    for (int j = 0; j < bias_.GetNumEls(); j++) bias_avg[j] = 0;
    for (int i = 0; i < max_ind; i++) {
      b = polyak_bias_[i].GetHostData();
      for (int j = 0; j < bias_.GetNumEls(); j++) bias_avg[j] += b[j];
    }
    for (int j = 0; j < bias_.GetNumEls(); j++) bias_avg[j] /= max_ind;
    bias_.CopyToDevice();
  }
}

void EdgeWithWeight::SetTiedTo(Edge* e) {
  tied_edge_ = dynamic_cast<EdgeWithWeight*>(e);
  if (tied_edge_ == NULL) {
    cerr << "Error: Edge " << GetName() << " cannot be tied to edge "
         << e->GetName() << " which is not of the same type." << endl;
    exit(1);
  }
  num_shares_++;
}

int EdgeWithWeight::GetNumGradsReceived() {
  return is_tied_ ? tied_edge_->GetNumGradsReceived() : num_grads_received_;
}

void EdgeWithWeight::IncrementNumGradsReceived() {
  if (is_tied_) {
    tied_edge_->IncrementNumGradsReceived();
  } else {
    num_grads_received_++;
  }
}

}
}
