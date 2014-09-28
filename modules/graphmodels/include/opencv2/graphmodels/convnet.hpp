#ifndef CONVNET_H_
#define CONVNET_H_
#include "layer.hpp"
#include "hdf5.h"
#include "datahandler.hpp"
#include "datawriter.hpp"
#include <vector>
#include <string>

#include "opencv2/core.hpp"

using namespace std;

namespace cv
{
namespace graphmodels
{

/**
 * A Convolutional Net Model.
 * This class provides the interface for training and using conv nets.
 */
class CV_EXPORTS ConvNet {
 public:
  /**
  * Instantiate a model using the config in model_file.
  */ 
  ConvNet(const string& model_file);
  virtual ~ConvNet();
  virtual void SetupDataset(const string& train_data_config_file);
  virtual void SetupDataset(const string& train_data_config_file, const string& val_data_config_file);

  /** Start training.*/
  virtual void Train();

  /** Validate the model on the specfied dataset and return the error.
   * @param dataset The dataset.
   * @param error A vector of errors (one element for each output layer).
   */ 
  void Validate(DataHandler* dataset, vector<float>& error);
  
  /** Validate the model on the validation dataset and return the error.
   * @param error A vector of errors (one element for each output layer).
   */ 
  void Validate(vector<float>& error);

  /** Write the model to disk.*/
  void Save();

  /** Write the model to disk in the file specified. */
  void Save(const string& output_file);

  /** Load the model.*/
  void Load();

  /** Load the model from the file specified.*/
  void Load(const string& input_file);

  /** Display the state of the model.
   * Shows the layers and edges for which display is enabled.
   */ 
  void Display();
  
  /** Write the state of the layers to disk.
   * Runs the model on the dataset specified in config and writes
   * the requested layer states out to disk in a hdf5 file.
   * @param config Feature extractor configuartion.
   */ 
  void ExtractFeatures(const config::FeatureExtractorConfig& config);
  void ExtractFeatures(const string& config_file);

  /** Allocate memory for the model.
   * @param fprop_only If true, does not allocate memory needed for optimization.
   */ 
  void AllocateMemory(bool fprop_only);

  void SetBatchsize(const int batch_size);
  
  Layer* GetLayerByName(const string& name);

  /** Forward propagate through the network.
   * @param train If true, this forward prop is being done during training,
   * otherwise during test/validation. Used for determining whether to use drop
   * units stochastcially or use all of them.
   */ 
  virtual void Fprop(bool train);

 protected:
  /** Creates layers and edges.*/ 
  void BuildNet();

  /** Release all memory held by the model.*/
  void DestroyNet();

  /** Add a sub-network into this network.*/
  void AddSubnet(config::Model& model, const config::Subnet& subnet);

  /** Allocate layer memory for using mini-batches of batch_size_.*/
  void AllocateLayerMemory();

  /** Allocate memory for edges.
   * @param fprop_only If true, does not allocate memory needed for optimization.
   */ 
  void AllocateEdgeMemory(bool fprop_only);
  
  string GetCheckpointFilename();
  void TimestampModel();

  /** Sets up fields of view as seen by each location at the output layer.*/
  void FieldsOfView();
  
  /** Topologically sort layers.*/
  void Sort();

  /** Forward propagate one layer.
   * Passes up input through the edge and updates the state of the output.
   * @param input the input layer.
   * @param output the output layer.
   * @param edge the edge connecting the input to the output.
   * @param overwrite If true, overwrite the state present in output, else add to it.
   */ 
  void Fprop(Layer& input, Layer& output, Edge& edge, bool overwrite);
  
  /** Back propagate through one layer.
   * Passes down the gradients from the output layer to the input layer.
   * Also updates the weights on the edge (if update_weights is true).
   * @param output the output layer (gradients w.r.t this have been computed).
   * @param input the input layer (gradients w.r.t this will be computed here).
   * @param edge the edge connecting the input to the output.
   * @param overwrite If true, overwrite the deriv present in input, else add
   * to it.
   * @param update_weights If true, the weights will be updated.
   */ 
  virtual void Bprop(Layer& output, Layer& input, Edge& edge, bool overwrite, bool update_weights);


  /** Backpropagate through the network and update weights.*/ 
  virtual void Bprop(bool update_weights);
  
  /** Computes the derivative of the loss function.*/ 
  virtual void ComputeDeriv();

  /** Computes the loss function (to be displayed).*/ 
  virtual void GetLoss(vector<float>& error);

  /** Takes one optimization step.*/ 
  virtual void TrainOneBatch(vector<float>& error);
  virtual void DisplayLayers();
  void DisplayEdges();
  void InsertPolyak();
  void LoadPolyakWeights();
  void LoadCurrentWeights();
  void WriteLog(int current_iter, float time, float training_error);
  void WriteLog(int current_iter, float time, const vector<float>& training_error);
  void WriteValLog(int current_iter, const vector<float>& error);
  
  /** Decides if learning rate should be reduced.*/
  bool CheckReduceLearningRate(const vector<float>& val_error);

  /** Multiply learning rate by factor.*/
  void ReduceLearningRate(const float factor);

  void SetupLocalizationDisplay();
  void DisplayLocalization();

  config::Model model_;  /** The model protobuf config.*/
  vector<Layer*> layers_;  /** The layers in the network.*/
  vector<Layer*> data_layers_;  /** Layers which have data associated with them.*/
  vector<Layer*> input_layers_;  /** Input layers.*/
  vector<Layer*> output_layers_;  /** Output layers.*/
  vector<Edge*> edges_;  /** The edges in the network.*/
  int max_iter_, batch_size_, current_iter_, lr_reduce_counter_;
  DataHandler *train_dataset_, *val_dataset_;
  string checkpoint_dir_, output_file_, model_name_;
  ImageDisplayer displayer_;
  string model_filename_, timestamp_, log_file_, val_log_file_;

  // Field of view.
  int fov_size_, fov_stride_, fov_pad1_, fov_pad2_;
  int num_fov_x_, num_fov_y_;
  bool localizer_;

  // a+=b;
  static void AddVectors(vector<float>& a, vector<float>& b);
  ImageDisplayer* localization_display_;
};

}
}

#endif
