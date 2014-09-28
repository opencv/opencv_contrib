#include "opencv2/graphmodels/multigpu_convnet.hpp"

namespace cv
{
namespace graphmodels
{

MultiGPUConvNet::MultiGPUConvNet(const string& model_file):
  ConvNet(model_file) {
  broadcast_ = true;
  // The data transfer can be done in broadcast mode
  // or fetch-as-required mode.
}

void MultiGPUConvNet::Fprop(bool train) {
  int dst_layer_gpu_id, edge_gpu_id, src_layer_gpu_id;
  Layer *src_layer;
  Matrix *dst, *src;

  const int num_gpus = Matrix::GetNumBoards();
  vector<bool> overwrite(num_gpus);
  for(Layer* l : layers_) {
    for (int i = 0; i < num_gpus; i++) overwrite[i] = true;
    l->ResetStateCopies();
    dst_layer_gpu_id = l->GetGPUId();
    for (Edge* e : l->incoming_edge_) {
      src_layer = e->GetSource();
      edge_gpu_id = e->GetGPUId();
      src_layer_gpu_id = src_layer->GetGPUId();
      Matrix::SetDevice(edge_gpu_id);

      if (edge_gpu_id != dst_layer_gpu_id) {
        dst = &(l->GetOtherState(edge_gpu_id));
      } else {
        dst = &(l->GetState());
      }
      if (edge_gpu_id != src_layer_gpu_id) {
        if (!broadcast_) {
          src_layer->CopyStateToGPU(edge_gpu_id);
        }
        src = &(src_layer->GetOtherState(edge_gpu_id));
      } else {
        src = &(src_layer->GetState());
      }

      e->ComputeUp(*src, *dst, overwrite[edge_gpu_id]);
      dst->SetReady();  // l->AccumulateState will wait for this.
      overwrite[edge_gpu_id] = false;
    }
    
    Matrix::SetDevice(l->GetGPUId());
    if (l->IsInput()) {
      l->ApplyDropout(train);
    } else {
      l->AccumulateState();
      l->ApplyActivation(train);
    }
    l->GetState().SetReady();  // l->BroadcastState will wait for this.
    if (broadcast_) {
      l->BroadcastState();
    }
  }
}

void MultiGPUConvNet::ComputeDeriv() {
  for (Layer* l: output_layers_) {
    Matrix::SetDevice(l->GetGPUId());
    l->ComputeDeriv();
  }
}

void MultiGPUConvNet::GetLoss(vector<float>& error) {
  error.clear();
  for (Layer* l: output_layers_) {
    Matrix::SetDevice(l->GetGPUId());
    error.push_back(l->GetLoss());
  }
}

void MultiGPUConvNet::Bprop(bool update_weights) {
  Layer *l, *dst_layer;
  const int num_gpus = Matrix::GetNumBoards();
  vector<bool> overwrite(num_gpus);
  int src_layer_gpu_id, edge_gpu_id, dst_layer_gpu_id;
  Matrix *src_deriv, *dst_deriv, *src_state, *dst_state;
  for (int i = layers_.size() - 1; i >= 0; i--) {
    for (int i = 0; i < num_gpus; i++) overwrite[i] = true;
    l = layers_[i];
    l->ResetDerivCopies();
    src_layer_gpu_id = l->GetGPUId();
    for (Edge* e : l->outgoing_edge_) {
      dst_layer = e->GetDest();
      edge_gpu_id = e->GetGPUId();
      dst_layer_gpu_id = dst_layer->GetGPUId();

      Matrix::SetDevice(edge_gpu_id);
      if (edge_gpu_id != src_layer_gpu_id) {
        src_deriv = &(l->GetOtherDeriv(edge_gpu_id));
        src_state = &(l->GetOtherState(edge_gpu_id));
      } else {
        src_deriv = &(l->GetDeriv());
        src_state = &(l->GetState());
      }
      if (edge_gpu_id != dst_layer_gpu_id) {
        if (!broadcast_) {
          dst_layer->CopyDerivToGPU(edge_gpu_id);
          dst_layer->CopyStateToGPU(edge_gpu_id);
        }
        dst_deriv = &(dst_layer->GetOtherDeriv(edge_gpu_id));
        dst_state = &(dst_layer->GetOtherState(edge_gpu_id));
      } else {
        dst_deriv = &(dst_layer->GetDeriv());
        dst_state = &(dst_layer->GetState());
      }
      e->ComputeOuter(*src_state, *dst_deriv);
      if (!l->IsInput() && !e->IsBackPropBlocked()) {
        e->ComputeDown(*dst_deriv, *src_state, *dst_state, *src_deriv,
                       overwrite[edge_gpu_id]);
        src_deriv->SetReady();  // l->AccumulateDeriv will wait for this.
      }
      if (update_weights) e->UpdateWeights();
      overwrite[edge_gpu_id] = false;
    }
    if (!l->IsInput()) {
      Matrix::SetDevice(l->GetGPUId());
      if (!l->IsOutput()) {
        l->AccumulateDeriv();
        l->ApplyDerivativeOfActivation();
      }
      l->GetDeriv().SetReady(); // l->BroadcastDeriv will wait for this.
      if (broadcast_) {
        l->BroadcastDeriv();
      }
    }
  }
}

}
}
