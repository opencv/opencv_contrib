#ifndef EDGE_H_
#define EDGE_H_
#include "util.hpp"
#include "matrix.hpp"
#include <iostream>

#include "opencv2/core.hpp"

namespace cv
{
namespace graphmodels
{

class Layer;

/** This class is intended to be used as a base class for implementing edges.
 * This is an abstract class - ComputeUp and ComputeDown methods must be
 * implemented by derived classes.
 */
class CV_EXPORTS Edge {
 public:
  /** Instatntiate an Edge from the config.*/
  Edge(const config::Edge& edge_config);
  virtual ~Edge();
  
  /** Allocate memory for the model.
   * @param fprop_only If true, does not allocate memory needed for optimization.
   */ 
  virtual void AllocateMemory(bool fprop_only);
  
  /** Initialize the weights and biases.*/
  virtual void Initialize();

  /** Write the weights and biases in an hdf5 file.
   * @param file The file handle. The file has been opened for writing. Do not close it.
   */ 
  virtual void SaveParameters(hid_t file);
  
  /** Load the weights and biases from an hdf5 file.
   * @param file The file handle. The file has been opened for reading. Do not close it.
   */ 
  virtual void LoadParameters(hid_t file);

  virtual void InsertPolyak();
  virtual void BackupCurrent();
  virtual void LoadCurrentOnGPU();
  virtual void LoadPolyakOnGPU();

  /** Returns the root mean square weight value.*/
  virtual float GetRMSWeight();

  /** Reduce the learning rate by factor.*/
  virtual void ReduceLearningRate(float factor);

  /** Returns whether the edge has any parameters.*/
  virtual bool HasNoParameters() const;

  /** Returns the number of modules.
   * This is relevant for convolution-like edges.
   */ 
  virtual int GetNumModules() const;

  /** Displays the weights.
   * Supportsinput layer weights only.
   */
  virtual void DisplayWeights();

  /** Displays the statistics of the weights.*/
  virtual void DisplayWeightStats();

  /** Sets the edge to be tied to another edge.*/
  virtual void SetTiedTo(Edge* e);

  /** Computes the output layer state given the input.
   * Applies the weights and adds bias.
   */
  virtual void ComputeUp(Matrix& input, Matrix& output, bool overwrite) = 0;
  
  /** Computes the derivative w.r.t the inputs of this edge given the derivative
   * w.r.t the outputs of this edge.
   * @param deriv_output Derivative w.r.t outputs of this edge.(In)
   * @param input The input to this edge.(In)
   * @param output The output of this edge.(In)
   * @param deriv_input Derivative w.r.t inputs of this edge.(Out)
   */
  virtual void ComputeDown(Matrix& deriv_output, Matrix& input,
                           Matrix& output, Matrix& deriv_input,
                           bool overwrite) = 0;

  /** Computes the gradient for the weights and biases.
   * @param input The input to this edge.
   * @param deriv_output The derivative w.r.t the output of this edge.
   */ 
  virtual void ComputeOuter(Matrix& input, Matrix& deriv_output);
  
  /** Update the weights.*/
  virtual void UpdateWeights();

  //virtual bool RequiresMemoryForDeriv() const;
  
  /** Set the spatial size of the input to this edge.*/
  virtual void SetImageSize(int image_size);

  /** Returns the size of the input field of view corresponding to an output
   * of 'size'.*/
  virtual void FOV(int* size, int* sep, int* pad1, int* pad2) const;

  /** Returns whether back prop is blocked through this edge.*/
  bool IsBackPropBlocked() const { return block_backprop_; }

  void SetSource(Layer* source);
  void SetDest(Layer* dest);
  Layer* GetSource();
  Layer* GetDest();
  const string& GetSourceName();
  const string& GetDestName();
  const string& GetName();

  /** Set the number of input channels.*/
  void SetInputChannels(int a);
  /** Set the number of output channels.*/
  void SetOutputChannels(int a);

  void SetMark();
  bool HasMark();
  string GetTiedEdgeName();
  bool IsTied();
  int GetGPUId() const { return gpu_id_; }

  /** Selects the appropriate derived class for the edge config.*/
  static Edge* ChooseEdgeClass(const config::Edge& edge_config);
  
 protected:
  Layer *source_;  /** The source layer for this edge.*/
  Layer *dest_;  /** The destination layer for this edge.*/
  const string source_node_, dest_node_, name_, tied_edge_name_;
  Edge* tied_edge_;  /* The edge to which this edge is tied.*/
  int num_input_channels_, num_output_channels_, image_size_, num_modules_;
  bool mark_;  /** A marker. Used for topological sorting.*/
  const bool block_backprop_, is_tied_;
  ImageDisplayer *img_display_;
  const int gpu_id_;  /** The GPU on which this edge should do its computation.*/
  const bool display_;
};

}
}

#endif
