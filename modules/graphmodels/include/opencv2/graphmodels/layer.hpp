#ifndef LAYER_H_
#define LAYER_H_
#include "edge.hpp"
#include <set>

#include "opencv2/core.hpp"

namespace cv
{
namespace graphmodels
{

/** The base class for all layers.
 * Each layer has a state_ and deriv_.
 */ 
class CV_EXPORTS Layer {
 public:
  /** Instantiate a layer from config. */ 
  Layer(const config::Layer& config);
  virtual ~Layer();

  /** Allocate memory for storing the state and derivative at this layer.
   * @param batch_size The mini-batch size.
   */ 
  virtual void AllocateMemory(int batch_size);

  /** Apply the activation function.
   * Derived classes must implement this. This method applies the activation
   * function to the state_ and overwrites it.
   * @param train If true, use dropout.
   */ 
  virtual void ApplyActivation(bool train) = 0;

  /** Apply the derivative of the activation.
   * Derived classes must implement this. Computes the derivative w.r.t the
   * inputs to this layer from the derivative w.r.t the outputs of this layer.
   * Applies the derivative of the activation function to deriv_ and overwrites
   * it.
   */ 
  virtual void ApplyDerivativeOfActivation() = 0;

  /** Compute derivative of loss function.
   * This is applicable only if this layer is an output layer.
   */ 
  virtual void ComputeDeriv() = 0;

  /** Compute the value of the loss function that is displayed during training.
   * This is applicable only if this layer is an output layer.
   */ 
  virtual float GetLoss() = 0;

  /** Compute the value of the actual loss function.
   * This is applicable only if this layer is an output layer.
   */ 
  virtual float GetLoss2();

  /** Apply dropout to this layer.
   * @param train If train is true, drop units stochastically,
   * else use all the units.
   */ 
  void ApplyDropout(bool train);

  /** Apply derivative of dropout.
   * This method scales the derivative to compensate for dropout.
   */ 
  void ApplyDerivativeofDropout();

  // Methods for preventing race conditions when using multiple GPUs.
  void AccessStateBegin();
  void AccessStateEnd();
  void AccessDerivBegin();
  void AccessDerivEnd();

  /** Returns the incoming edge by index. */
  Edge* GetIncomingEdge(int index) { return incoming_edge_[index]; }  // TODO:add check for size.

  /** Returns a reference to the state of the layer.*/
  Matrix& GetState() { return state_;}

  /** Returns a reference to the deriv at this layer.*/
  Matrix& GetDeriv() { return deriv_;}
  
  /** Returns a reference to the data at this layer.*/
  Matrix& GetData() { return data_;}

  void Display();
  void Display(int image_id);

  /** Add an incoming edge to this layer.*/
  void AddIncoming(Edge* e);

  /** Add an outgoing edge from this layer.*/
  void AddOutgoing(Edge* e);

  const string& GetName() const { return name_; }
  int GetNumChannels() const { return num_channels_; }
  int GetSize() const { return image_size_; }
  bool IsInput() const { return is_input_; }
  bool IsOutput() const { return is_output_; }

  void SetSize(int image_size);
  int GetGPUId() const { return gpu_id_; }
  void AllocateMemoryOnOtherGPUs();
  Matrix& GetOtherState(int gpu_id);
  Matrix& GetOtherDeriv(int gpu_id);

  void AccumulateState();
  void AccumulateDeriv();
  void BroadcastState();
  void BroadcastDeriv();
  void CopyStateToGPU(int dest_gpu);
  void CopyDerivToGPU(int dest_gpu);
  void ResetStateCopies();
  void ResetDerivCopies();

  static Layer* ChooseLayerClass(const config::Layer& layer_config);

  vector<Edge*> incoming_edge_, outgoing_edge_;
  bool has_incoming_from_same_gpu_, has_outgoing_to_same_gpu_;
  bool has_incoming_from_other_gpus_, has_outgoing_to_other_gpus_;

 protected:
  void ApplyDropoutAtTrainTime();
  void ApplyDropoutAtTestTime();

  const string name_;
  const int num_channels_;
  bool is_input_, is_output_;
  const float dropprob_;
  const bool display_, dropout_scale_up_at_train_time_, gaussian_dropout_;

  // Maximum activation after applying gaussian dropout.
  // This is needed to prevent blow ups due to sampling large values.
  const float max_act_gaussian_dropout_;

  int scale_targets_, image_size_;

  Matrix state_;  /** State (activation) of the layer. */
  Matrix deriv_;  /** Deriv of the loss function w.r.t. the state. */
  Matrix data_;   /** Data (targets) associated with this layer. */
  Matrix rand_gaussian_;  /** Need to store random variates when doing gaussian dropout. */
  map<int, Matrix> other_states_; /** Copies of this layer's state on other gpus.*/
  map<int, Matrix> other_derivs_; /** Copies of this layer's deriv on other gpus.*/
  map<int, bool> state_copied_;
  map<int, bool> deriv_copied_;
  ImageDisplayer *img_display_;
  const int gpu_id_;
  set<int> other_incoming_gpu_ids_, other_outgoing_gpu_ids_;
};

/** Implements a layer with a linear activation function.*/
class CV_EXPORTS LinearLayer : public Layer {
 public:
  LinearLayer(const config::Layer& config);
  virtual void AllocateMemory(int batch_size);
  virtual void ApplyActivation(bool train);
  virtual void ApplyDerivativeOfActivation();
  virtual void ComputeDeriv();
  virtual float GetLoss();
};

/** Implements a layer with a rectified linear activation function.*/
class CV_EXPORTS ReLULayer : public LinearLayer {
 public:
  ReLULayer(const config::Layer& config);
  virtual void ApplyActivation(bool train);
  virtual void ApplyDerivativeOfActivation();
 protected:
  const bool rectify_after_gaussian_dropout_;
};

/** Implements a layer with a softmax activation function.
 * This must be an output layer. The target must be one of K choices.
 */
class CV_EXPORTS SoftmaxLayer : public Layer {
 public:
  SoftmaxLayer(const config::Layer& config) : Layer(config) {};
  virtual void AllocateMemory(int batch_size);
  virtual void ApplyActivation(bool train);
  virtual void ApplyDerivativeOfActivation();
  virtual void ComputeDeriv();
  virtual float GetLoss();
  virtual float GetLoss2();
};

/** Implements a layer with a softmax activation function.
 * This must be an output layer.
 * The target must be a distribution over K choices.
 */
class CV_EXPORTS SoftmaxDistLayer : public SoftmaxLayer {
 public:
  SoftmaxDistLayer(const config::Layer& config) : SoftmaxLayer(config) {};
  virtual void AllocateMemory(int batch_size);
  virtual void ComputeDeriv();
  virtual float GetLoss();

 private:
  Matrix cross_entropy_;
};

/** Implements a layer with a logistic activation function.
 */ 
class CV_EXPORTS LogisticLayer : public Layer {
 public:
  LogisticLayer(const config::Layer& config) : Layer(config) {};
  virtual void AllocateMemory(int batch_size);
  virtual void ApplyActivation(bool train);
  virtual void ApplyDerivativeOfActivation();
  virtual void ComputeDeriv();
  virtual float GetLoss();
};

/** Implements a layer with a linear activation and hinge loss.*/
class CV_EXPORTS HingeQuadraticLayer : public SoftmaxLayer {
 public:
  HingeQuadraticLayer(const config::Layer& config);
  virtual void ApplyActivation(bool train);
  virtual void ApplyDerivativeOfActivation();
  virtual void ComputeDeriv();
 protected:
  const float margin_;
};

/** Implements a layer with a linear activation and hinge loss.*/
class CV_EXPORTS HingeLinearLayer : public HingeQuadraticLayer {
 public:
  HingeLinearLayer(const config::Layer& config);
  virtual void ComputeDeriv();
};

}
}

#endif
