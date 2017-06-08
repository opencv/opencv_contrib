#include "opencv2/graphmodels/convnet.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <algorithm>
#include <sstream>
#include <stack>
#include <queue>
#include <chrono>
#include <csignal>

namespace cv
{
namespace graphmodels
{

using namespace std;
ConvNet::ConvNet(const string& model_file):
  max_iter_(0), batch_size_(0), current_iter_(0),
  lr_reduce_counter_(0),
  train_dataset_(NULL), val_dataset_(NULL),
  model_filename_(model_file) {
  ReadModel(model_file, model_);
  for (const config::Subnet& subnet : model_.subnet()) {
    AddSubnet(model_, subnet);
  }
  Matrix::InitRandom(model_.seed());
  srand(model_.seed());
  localizer_ = model_.localizer();
  model_name_ = model_.name();
  checkpoint_dir_ = model_.checkpoint_dir();

  int num_tstampts = model_.timestamp_size();
  if (num_tstampts > 0) {
    timestamp_ = model_.timestamp(num_tstampts - 1);
  }
  BuildNet();  // Build a net using the connectivity specified in model_.
}

ConvNet::~ConvNet() {
  DestroyNet();
  if (val_dataset_ != NULL) delete val_dataset_;
  if (train_dataset_ != NULL) delete train_dataset_;
}

void ConvNet::DestroyNet() {
  for (Layer* l: layers_) delete l;
  for (Edge* e: edges_) delete e;
}

void ConvNet::AddSubnet(config::Model& model, const config::Subnet& subnet) {
  config::Model submodel;
  ReadModel(subnet.model_file(), submodel);

  // Recursively add subnets.
  for (const config::Subnet& s : submodel.subnet()) {
    AddSubnet(submodel, s);
  }

  const string& name = subnet.name();
  int gpu_offset = subnet.gpu_id_offset();
  int num_channels_multiplier = subnet.num_channels_multiplier();
  cout << "Adding subnet " << name << endl;
  map<string, string> merge_layers;
  set<string> remove_layers;
  for (const config::Subnet_MergeLayer& ml : subnet.merge_layer()) {
    merge_layers[ml.subnet_layer()] = ml.net_layer();
  }
  for (const string& l : subnet.remove_layer()) {
    remove_layers.insert(l);
  }
  for (const config::Layer& layer : submodel.layer()) {
    if (merge_layers.find(layer.name()) == merge_layers.end() &&
        remove_layers.find(layer.name()) == remove_layers.end()) {
      config::Layer* l = model.add_layer();
      l->MergeFrom(layer);
      l->set_name(name + "_" + l->name());
      l->set_gpu_id(gpu_offset + l->gpu_id());
      l->set_num_channels(num_channels_multiplier * l->num_channels());
    }
  }
  int soa = subnet.start_optimization_after(); 
  for (const config::Edge& edge : submodel.edge()) {
    if (remove_layers.find(edge.source()) != remove_layers.end() ||
        remove_layers.find(edge.dest()) != remove_layers.end()) continue;
    config::Edge* e = model.add_edge();
    e->MergeFrom(edge);
    e->set_gpu_id(gpu_offset + e->gpu_id());
    if (!subnet.parameters_file().empty()) {
      e->set_initialization(config::Edge::PRETRAINED);
      e->set_pretrained_model(subnet.parameters_file());
      e->set_pretrained_edge_name(edge.source() + ":" + edge.dest());
    }
    map<string, string>::iterator it1 = merge_layers.find(edge.source());
    map<string, string>::iterator it2 = merge_layers.find(edge.dest());
    e->set_source((it1 == merge_layers.end()) ? (name + "_" + e->source()):
                                                it1->second);
    e->set_dest((it2 == merge_layers.end()) ? (name + "_" + e->dest()):
                                               it2->second);
    e->set_block_backprop(edge.block_backprop() | subnet.block_backprop());
    e->mutable_weight_optimizer()->set_start_optimization_after(soa);
    e->mutable_bias_optimizer()->set_start_optimization_after(soa);
  }
}

void ConvNet::BuildNet() {
  layers_.resize(model_.layer_size());
  edges_.resize(model_.edge_size());

  // Setup edges.
  for (int i = 0; i < edges_.size(); i++) {
    edges_[i] = Edge::ChooseEdgeClass(model_.edge(i));
  }

  // Communicate information about tied edges.
  map<string, Edge*> edge_name_map;
  for (Edge* e: edges_) {
    edge_name_map[e->GetName()] = e;
  }
  for (Edge* e: edges_) {
    if (e->IsTied()) {
      e->SetTiedTo(edge_name_map[e->GetTiedEdgeName()]);  // TODO: Check if not found.
    }
  }

  // Setup layers.
  for (int i = 0; i < layers_.size(); i++) {
    layers_[i] = Layer::ChooseLayerClass(model_.layer(i));
    for (Edge* e : edges_) {
      if (layers_[i]->GetName().compare(e->GetSourceName()) == 0) {
        layers_[i]->AddOutgoing(e);
        e->SetSource(layers_[i]);
        e->SetInputChannels(layers_[i]->GetNumChannels());
      }
      if (layers_[i]->GetName().compare(e->GetDestName()) == 0) {
        layers_[i]->AddIncoming(e);
        e->SetDest(layers_[i]);
        e->SetOutputChannels(layers_[i]->GetNumChannels());
      }
    }
  }

  // Topologically sort layers.
  Sort();
  // layers_ now contains the layers in an fprop-safe order.

  for (Layer* l : layers_) {
    if (l->incoming_edge_.size() == 0) input_layers_.push_back(l);
    if (l->outgoing_edge_.size() == 0) output_layers_.push_back(l);
    if (l->incoming_edge_.size() == 0 || l->outgoing_edge_.size() == 0) {
      data_layers_.push_back(l);
    }
  }

  int image_size;
  for (Layer* l : layers_) {
    image_size = l->IsInput() ? model_.patch_size() :
                                l->incoming_edge_[0]->GetNumModules();
    l->SetSize(image_size);
    for (Edge* e: l->outgoing_edge_) {
      e->SetImageSize(image_size);
    }
  }

  if (localizer_) {
    FieldsOfView();
  }
}

void ConvNet::FieldsOfView() {
  Layer* l = output_layers_[0];
  fov_size_ = 1, fov_stride_ = 1, fov_pad1_ = 0, fov_pad2_ = 0;
  while(!l->IsInput()) {
    Edge* e = l->incoming_edge_[0];
    e->FOV(&fov_size_, &fov_stride_, &fov_pad1_, &fov_pad2_);
    l = e->GetSource();
  }
  float image_size = (float)model_.patch_size();

  cout << "FOV: " << fov_size_ << " " << fov_stride_ << " " << fov_pad1_ << " " << fov_pad2_ << endl;
  cout << "Image size " << image_size << endl;
  
  num_fov_x_ = output_layers_[0]->GetSize();
  num_fov_y_ = output_layers_[0]->GetSize();
}

void ConvNet::AllocateLayerMemory() {
  for (Layer* l : layers_) {
    l->AllocateMemory(batch_size_);
  }
}

void ConvNet::AllocateEdgeMemory(bool fprop_only) {
  for (Edge* e : edges_) e->AllocateMemory(fprop_only);

  if (timestamp_.empty()) {
    // Initialize randomly.
    for (Edge* e: edges_) e->Initialize();
  } else {
    // Initialize from a saved model.
    Load();
  }
}

void ConvNet::Sort() {
  Layer *m, *n;
  vector<Layer*> L;
  //stack<Layer*> S;  // Depth-first sort.
  queue<Layer*> S;  // Breadth-first sort.
  // Breadth-first usually works well for multi-gpu multi-column nets where
  // each column has roughly the same amount of work.
  
  for (Layer* l : layers_) if (l->IsInput()) S.push(l);
  if (S.empty()) {
    cerr << "Error: No layer is set to be input!" << endl;
    exit(1);
  }
  bool x;

  while (!S.empty()) {
    //n = S.top();
    n = S.front();
    S.pop();
    L.push_back(n);
    for (Edge* e : n->outgoing_edge_) {
      e->SetMark();
      m = e->GetDest();
      x = true;
      for (Edge* f : m->incoming_edge_) x &= f->HasMark();
      if (x) S.push(m);
    }
  }
  x = true;
  for (Edge* f : edges_) x &= f->HasMark();
  if (!x) {
    cerr << "Error : Network has loop(s)!" << endl;
    exit(1);
  }

  // Re-order layers in the instance variable.
  for (int i = 0; i < layers_.size(); i++) layers_[i] = L[i];
}

void ConvNet::Fprop(Layer& input, Layer& output, Edge& edge, bool overwrite) {
  edge.ComputeUp(input.GetState(), output.GetState(), overwrite);
}

void ConvNet::Bprop(Layer& output, Layer& input, Edge& edge, bool overwrite,
                    bool update_weights) {
  if (edge.IsBackPropBlocked()) return;
  edge.ComputeOuter(input.GetState(), output.GetDeriv());
  if (!input.IsInput()) {
    edge.ComputeDown(output.GetDeriv(), input.GetState(), output.GetState(),
                     input.GetDeriv(), overwrite);
  }
  if (update_weights) edge.UpdateWeights();
}

void ConvNet::Fprop(bool train) {
  bool overwrite;
  for(Layer* l : layers_) {
    overwrite = true;
    for (Edge* e : l->incoming_edge_) {
      Fprop(*(e->GetSource()), *l, *e, overwrite);
      overwrite = false;
    }
    if (l->IsInput()) {
      l->ApplyDropout(train);
    } else {
      l->ApplyActivation(train);
    }
  }
}

void ConvNet::Bprop(bool update_weights) {
  Layer *l;
  bool overwrite;
  for (int i = layers_.size() - 1; i >= 0; i--) {
    overwrite = true;
    l = layers_[i];
    for (Edge* e : l->outgoing_edge_) {
      Bprop(*(e->GetDest()), *l, *e, overwrite, update_weights);
      overwrite = false;
    }
    if (!l->IsInput() && l->outgoing_edge_.size() > 0) {
      l->ApplyDerivativeOfActivation();
    }
  }
}

void ConvNet::ComputeDeriv() {
  for (Layer* l: output_layers_) l->ComputeDeriv();
}

void ConvNet::GetLoss(vector<float>& error) {
  error.clear();
  for (Layer* l: output_layers_) {
    error.push_back(l->GetLoss());
  }
}

void ConvNet::TrainOneBatch(vector<float>& error) {
  train_dataset_->GetBatch(data_layers_);
  Fprop(true);
  ComputeDeriv();
  GetLoss(error);
  Bprop(true);
}

void ConvNet::SetupDataset(const string& train_data_config_file) {
  SetupDataset(train_data_config_file, "");
}

void ConvNet::SetBatchsize(const int batch_size) {
  batch_size_ = batch_size;
}

void ConvNet::SetupDataset(const string& train_data_config_file,
                           const string& val_data_config_file) {

  config::DatasetConfig train_data_config;
  ReadDataConfig(train_data_config_file, train_data_config);
  train_dataset_ = new DataHandler(train_data_config);
  if (localizer_) {
    train_dataset_->SetFOV(fov_size_, fov_stride_, fov_pad1_, fov_pad2_,
                           model_.patch_size(), num_fov_x_, num_fov_y_);
  }
  SetBatchsize(train_dataset_->GetBatchSize());
  int dataset_size = train_dataset_->GetDataSetSize();
  train_dataset_->AllocateMemory();
  cout << "Training data set size " << dataset_size << endl;
  if (!val_data_config_file.empty()) {
    config::DatasetConfig val_data_config;
    ReadDataConfig(val_data_config_file, val_data_config);
    val_dataset_ = new DataHandler(val_data_config);
    if (localizer_) {
      val_dataset_->SetFOV(fov_size_, fov_stride_, fov_pad1_, fov_pad2_,
                           model_.patch_size(), num_fov_x_, num_fov_y_);
    }
    dataset_size = val_dataset_->GetDataSetSize();
    val_dataset_->AllocateMemory();
    cout << "Validation data set size " << dataset_size << endl;
  }
}

void ConvNet::AllocateMemory(bool fprop_only) {
  AllocateLayerMemory();
  AllocateEdgeMemory(fprop_only);
}

void ConvNet::Validate(DataHandler* dataset, vector<float>& total_error) {
  if (dataset == NULL) return;
  vector<float> error;
  dataset->Seek(0);
  int dataset_size = dataset->GetDataSetSize(),
      batch_size = dataset->GetBatchSize(),
      num_batches = dataset_size / batch_size;
  for (int k = 0; k < num_batches; k++) {

    dataset->GetBatch(data_layers_);
    Fprop(false);
    GetLoss(error);
    if (total_error.size() != error.size()) total_error.resize(error.size());
    for (int i = 0; i < error.size(); i++) {
      total_error[i] = (total_error[i] * k) / (k+1) + error[i] / (batch_size * (k+1));
    }
  }
  dataset->Sync();
}

Layer* ConvNet::GetLayerByName(const string& name) {
  for (Layer* l:layers_) {
    if (l->GetName().compare(name) == 0) return l;
  }
  cerr << "Error: No layer called " << name << endl;
  exit(1);
  return NULL;
}

void ConvNet::ExtractFeatures(const string& config_file) {
  config::FeatureExtractorConfig config;
  ReadFeatureExtractorConfig(config_file, config);
  ExtractFeatures(config);
}

void ConvNet::ExtractFeatures(const config::FeatureExtractorConfig& config) {
  DataHandler dataset = DataHandler(config.input());
  int dataset_size = dataset.GetDataSetSize(),
      batch_size = dataset.GetBatchSize(),
      num_batches = dataset_size / batch_size,
      left_overs = dataset_size % batch_size,
      multiplicity = dataset.GetMultiplicity();
  SetBatchsize(batch_size);
  dataset.AllocateMemory();
  AllocateMemory(true);

  const int display_after = model_.display_after();
  const bool display = model_.display();
  if (display && localizer_) {
    SetupLocalizationDisplay();
  }

 
  cout << "Extracting features for dataset of size " << dataset_size
       << " # batches " << num_batches
       << " # left overs " << left_overs << endl;
  if (left_overs > 0) num_batches++;
  cout << "Writing to " << config.output_file() << endl;
 
  DataWriter* data_writer = new DataWriter(config);
  data_writer->SetDataSetSize(dataset_size * multiplicity);
  vector<Layer*> layers;
  for (const config::FeatureStreamConfig& feature : config.feature()) {
    Layer* l = GetLayerByName(feature.layer());
    int numdims = l->GetState().GetCols();
    data_writer->SetNumDims(l->GetName(), numdims);
    layers.push_back(l);
  }
  int numcases;
  for (int k = 0; k < num_batches; k++) {
    cout << "\rBatch " << (k+1);
    cout.flush();
    numcases = (left_overs > 0 && k == num_batches - 1) ? left_overs : batch_size;
    for (int m = 0; m < multiplicity; m++) {
      dataset.GetBatch(data_layers_);
      if (display && k % display_after == 0) {
        DisplayLayers();
        if (localizer_) DisplayLocalization();
      }
      Fprop(false);
      data_writer->Write(layers, numcases);
    }
  }
  cout << endl;
  delete data_writer;
  dataset.Sync();
}

void ConvNet::Save() {
  Save(GetCheckpointFilename());
}

void ConvNet::Save(const string& output_file) {
  cout << "Saving model to " << output_file << endl;
  // Save to temp file.
  string output_file_temp = output_file + "temp";
  hid_t file = H5Fcreate(output_file_temp.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  for (Edge* e : edges_) e->SaveParameters(file);
  WriteHDF5IntAttr(file, "__lr_reduce_counter__", &lr_reduce_counter_);
  WriteHDF5IntAttr(file, "__current_iter__", &current_iter_);
  H5Fclose(file);
  cout << " .. Done" << endl;
  // Move to original file.
  int result = rename(output_file_temp.c_str(), output_file.c_str());
  if (result != 0) {
    cerr << "Error renaming file." << endl;
  }
}

void ConvNet::InsertPolyak() {
  for (Edge* e : edges_) e->InsertPolyak();
}

void ConvNet::LoadPolyakWeights() {
  for (Edge* e : edges_) {
    e->BackupCurrent();
    e->LoadPolyakOnGPU();
  }
}

void ConvNet::LoadCurrentWeights() {
  for (Edge* e : edges_) e->LoadCurrentOnGPU();
}

string ConvNet::GetCheckpointFilename() {
  string filename = checkpoint_dir_ + "/" + model_name_ + "_" + timestamp_ + ".h5";
  return filename;
}

void ConvNet::Load() {
  Load(GetCheckpointFilename());
}

void ConvNet::Load(const string& input_file) {
  cout << "Loading model from " << input_file << endl;
  hid_t file = H5Fopen(input_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  for (Edge* e : edges_) e->LoadParameters(file);
  ReadHDF5IntAttr(file, "__lr_reduce_counter__", &lr_reduce_counter_);
  for (int i = 0; i < lr_reduce_counter_; i++) {
    ReduceLearningRate(model_.reduce_lr_factor());
  }
  ReadHDF5IntAttr(file, "__current_iter__", &current_iter_);
  H5Fclose(file);
}

void ConvNet::DisplayLayers() {
  for (int i = 0; i < layers_.size(); i++){
    layers_[i]->Display(0);
  }
}

void ConvNet::DisplayEdges() {
  for (int i = 0; i < edges_.size(); i++){
    edges_[i]->DisplayWeights();
  }
}

void ConvNet::WriteLog(int current_iter, float time, float training_error) {
  vector<float> temp(1);
  temp[0] = training_error;
  WriteLog(current_iter, time, temp);
}

void ConvNet::WriteLog(int current_iter, float time, const vector<float>& training_error) {
  ofstream f(log_file_, ofstream::out | ofstream::app);
  f << current_iter << " " << time;
  for (const float& val: training_error) f << " " << val;
  f << endl;
  f.close();
}

void ConvNet::WriteValLog(int current_iter, const vector<float>& error) {
  ofstream f(val_log_file_, ofstream::out | ofstream::app);
  f << current_iter;
  for (const float& val: error) f << " " << val;
  f << endl;
  f.close();
}

// Look at the history of val_error to see if we should reduce the learning rate now.
bool ConvNet::CheckReduceLearningRate(const vector<float>& val_error) {
  const int len = val_error.size();
  /*
  const int threshold = 1;
  int argmax = 0;
  for (int i = 0; i < len; i++) {
    if (val_error[argmax] < val_error[i]) argmax = i;
  }
  cout << "Argmax : " << argmax << " len " << len << endl;
  return (len - argmax > threshold);
  */
  const int num_steps = model_.reduce_lr_num_steps();
  /*
  bool r = len >= num_steps;
  for (int i = 0; i < num_steps - 1 && r; i++) {
    float v1 = val_error[len - i - 1], v2 = val_error[len - i - 2];
    r &= smaller_is_better_ ? (v1 >= v2) : (v1 <= v2);
  }
  */
  if (len < num_steps) return false;
  int i = len - num_steps;
  float mean1 = 0, mean2 = 0;
  for (int j = 0; j < num_steps/2; j++) {
    mean1 = (mean1 * j) / (j+1) + val_error[i++] / (j+1);
  }
  for (int j = 0; j < num_steps - num_steps/2; j++) {
    mean2 = (mean2 * j) / (j+1) + val_error[i++] / (j+1);
  }
  float diff = model_.smaller_is_better() ? mean1 - mean2 : mean2 - mean1;
  return diff < model_.reduce_lr_threshold();
}

void ConvNet::ReduceLearningRate(const float factor) {
  for (Edge* e : edges_) {
    e->ReduceLearningRate(factor);
  }
}

void ConvNet::Validate(vector<float>& error) {
  Validate(val_dataset_, error);
}

void ConvNet::TimestampModel() {
  timestamp_ = GetTimeStamp();
  string fname = checkpoint_dir_ + "/" + model_name_ + "_" + timestamp_;
  TimestampModelFile(model_filename_, fname + ".pbtxt", timestamp_);
  log_file_ = fname + "_train.log";
  val_log_file_ = fname + "_valid.log";
}

void ConvNet::AddVectors(vector<float>& a, vector<float>& b) {
  if (a.size() == 0) a.resize(b.size());
  if (a.size() != b.size()) {
    cerr << "Cannot add vectors of different sizes." << endl;
    exit(1);
  }
  for (int i = 0; i < a.size(); i++) a[i] += b[i];
}

void ConvNet::SetupLocalizationDisplay() {
  int image_size = model_.patch_size();
  localization_display_ = new ImageDisplayer(image_size, image_size, 3, false,
                                          "localization");
  localization_display_->SetFOV(fov_size_, fov_stride_, fov_pad1_, fov_pad2_,
                                image_size, num_fov_x_, num_fov_y_);
}

void ConvNet::DisplayLocalization() {
  Layer *input_layer = input_layers_[0],
        *output_layer = output_layers_[0];
  Matrix& input = input_layer->GetState();
  Matrix& output = output_layer->GetState();
  Matrix& ground_truth = output_layer->GetData();

  input.CopyToHost();
  output.CopyToHost();
  ground_truth.CopyToHost();

  float *data = input.GetHostData(),
        *gt = ground_truth.GetHostData(),
        *preds = output.GetHostData();
  
  localization_display_->DisplayLocalization(data, preds, gt, input.GetRows());
}

void ConvNet::Train() {

  // Check if train data is available.
  if (train_dataset_ == NULL) {
    cerr << "Error: Train dataset is NULL." << endl;
    exit(1);
  }

  // Before starting the training, mark this model with a timestamp.
  TimestampModel();

  const int display_after = model_.display_after(),
            print_after = model_.print_after(),
            validate_after = model_.validate_after(),
            save_after = model_.save_after(),
            polyak_after = model_.polyak_after(),
            start_polyak_queue = validate_after - polyak_after * model_.polyak_queue_size();

  const bool display = model_.display(), print_weights = model_.print_weights();

  if (display && localizer_) {
    SetupLocalizationDisplay();
  }
  const float learning_rate_reduce_factor = model_.reduce_lr_factor();

  // Time keeping.
  chrono::time_point<chrono::system_clock> start_t, end_t;
  chrono::duration<double> time_diff;
  start_t = chrono::system_clock::now();

  vector<float> train_error, this_train_error;
  vector<float> val_error, this_val_error;
  int dont_reduce_lr = 0;
  const int lr_max_reduce = model_.reduce_lr_max();
  bool newline;

  for(int i = current_iter_; i < model_.max_iter(); i++) {
    current_iter_++;
    cout << "\rStep " << current_iter_;
    cout.flush();

    TrainOneBatch(this_train_error);
    AddVectors(train_error, this_train_error);

    if (i % display_after == 0 && display) {
      DisplayLayers();
      DisplayEdges();
      if (localizer_) DisplayLocalization();
    }
    newline = false;
    if ((i+1) % print_after == 0) {
      end_t = chrono::system_clock::now();
      time_diff = end_t - start_t;
      printf(" Time %f s Train Acc :", time_diff.count());
      for (float& err : train_error) err /= print_after * batch_size_;
      for (const float& err : train_error) printf(" %.5f", err);
      WriteLog(current_iter_, time_diff.count(), train_error);
      if (print_weights) {
        printf(" Weight length: " );
        for (Edge* e : edges_) {
          if (e->HasNoParameters() || e->IsTied()) continue;
          printf(" %.3f", e->GetRMSWeight());
        }
      }
      train_error.clear();
      start_t = end_t;
      newline = true;
    }

    if (polyak_after > 0 && (i+1) % polyak_after == 0 && ((i+1) % validate_after) >= start_polyak_queue) {
      InsertPolyak();
    }

    if (val_dataset_ != NULL && validate_after > 0 && (i+1) % validate_after == 0) {
      if (polyak_after > 0) LoadPolyakWeights();
      train_dataset_->Sync();
      Validate(this_val_error);
      if (polyak_after > 0) LoadCurrentWeights();

      val_error.push_back(this_val_error[0]);
      cout << " Val Acc :";
      for (const float& val: this_val_error) printf(" %.5f", val);
      WriteValLog(current_iter_, this_val_error);

      // Should we reduce the learning rate ?
      if (learning_rate_reduce_factor < 1.0) {
        bool reduce_learning_rate = CheckReduceLearningRate(val_error);
        if (reduce_learning_rate && lr_reduce_counter_ < lr_max_reduce
            && dont_reduce_lr-- < 0) {
          dont_reduce_lr = model_.reduce_lr_num_steps();
          cout << "Learning rate reduced " << ++lr_reduce_counter_ << " time(s).";
          ReduceLearningRate(learning_rate_reduce_factor);
        }
      }
      newline = true;
    }
    if (newline) cout << endl;
    if ((i+1) % save_after == 0) {
      train_dataset_->Sync();
      Save();
    }
  }
  if (model_.max_iter() % save_after != 0) {
    train_dataset_->Sync();
    Save();
  }
  cout << "End of training." << endl;
  if (display && localizer_) {
    delete localization_display_;
  }
}

}
}
