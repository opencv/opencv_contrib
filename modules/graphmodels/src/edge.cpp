#include "opencv2/graphmodels/edge.hpp"
#include <iostream>
#include "edge_with_weight.hpp"
#include "fc_edge.hpp"
#include "conv_edge.hpp"
#include "maxpool_edge.hpp"
#include "local_edge.hpp"
#include "upsample_edge.hpp"
#include "downsample_edge.hpp"
#include "response_norm_edge.hpp"
#include "rgb_to_yuv_edge.hpp"
#include "conv_onetoone_edge.hpp"

using namespace std;

namespace cv
{
namespace graphmodels
{

Edge* Edge::ChooseEdgeClass(const config::Edge& edge_config) {
  Edge* e = NULL;
  switch (edge_config.edge_type()) {
    case config::Edge::FC :
      e = new FCEdge(edge_config);
      break;
    case config::Edge::CONVOLUTIONAL :
      e = new ConvEdge(edge_config);
      break;
    case config::Edge::LOCAL :
      e = new LocalEdge(edge_config);
      break;
    case config::Edge::MAXPOOL :
      e = new MaxPoolEdge(edge_config);
      break;
    case config::Edge::RESPONSE_NORM :
      e = new ResponseNormEdge(edge_config);
      break;
    case config::Edge::UPSAMPLE :
      e = new UpSampleEdge(edge_config);
      break;
    case config::Edge::DOWNSAMPLE :
      e = new DownSampleEdge(edge_config);
      break;
    case config::Edge::RGBTOYUV :
      e = new RGBToYUVEdge(edge_config);
      break;
    case config::Edge::CONV_ONETOONE :
      e = new ConvOneToOneEdge(edge_config);
      break;
    default:
      cerr << "Error: Undefined edge type." << endl;
      exit(1);
  }
  return e;
}

Edge::Edge(const config::Edge& edge_config) :
  source_(NULL), dest_(NULL),
  source_node_(edge_config.source()),
  dest_node_(edge_config.dest()),
  name_(source_node_ + ":" + dest_node_),
  tied_edge_name_(edge_config.tied_to()),
  tied_edge_(NULL),
  num_input_channels_(0),
  num_output_channels_(0),
  image_size_(0),
  num_modules_(1),
  mark_(false),
  block_backprop_(edge_config.block_backprop()),
  is_tied_(!tied_edge_name_.empty()),
  img_display_(NULL),
  gpu_id_(edge_config.gpu_id()),
  display_(edge_config.display()) {}
  //if (img_display_ != NULL) img_display_->SetTitle(name_);

Edge::~Edge() {
  if (img_display_ != NULL) delete img_display_;
  // Other pointers are not owned by this class.
}

void Edge::SetTiedTo(Edge* e) {
  tied_edge_ = e;
}

void Edge::SetInputChannels(int a) {
  num_input_channels_ = a;
}

void Edge::SetOutputChannels(int a) {
  num_output_channels_ = a;
}

void Edge::SaveParameters(hid_t file) {
  // no op.
  // Parameter saving implemented in EdgeWithWeight or derived classes thereof.
}

void Edge::LoadParameters(hid_t file) {
  // no op.
  // Parameter loading implemented in EdgeWithWeight or derived classes thereof.
}

void Edge::Initialize() {
  // no op. Initialization done in derived classes.
  Matrix::SetDevice(gpu_id_);
}

void Edge::AllocateMemory(bool fprop_only) {
  Matrix::SetDevice(gpu_id_);
  Matrix::RegisterTempMemory(num_output_channels_, "Used for computing average length of incoming weight vectors.");
  // Actual memory allocation will happen in the derived class because memory
  // requirements differ for different kinds of edges.
}

void Edge::DisplayWeights() {
  // no op.
}

void Edge::DisplayWeightStats() {
  // no op.
}

void Edge::ReduceLearningRate(float factor) {
  // no op.
}

void Edge::ComputeOuter(Matrix& input, Matrix& deriv_output) {
  // no op.
}

void Edge::UpdateWeights() {
  // no op.
}

float Edge::GetRMSWeight() {
  return 0;
}

void Edge::SetSource(Layer* source) {
  source_ = source;
}

void Edge::SetDest(Layer* dest) {
  dest_ = dest;
}

Layer* Edge::GetSource() {
  return source_;
}

Layer* Edge::GetDest() {
  return dest_;
}

const string& Edge::GetSourceName() {
  return source_node_;
}

const string& Edge::GetDestName() {
  return dest_node_;
}

const string& Edge::GetName() {
  return name_;
}

void Edge::SetMark() {
  mark_ = true;
}

bool Edge::HasMark() {
  return mark_;
}

bool Edge::HasNoParameters() const {
  return true;
}

int Edge::GetNumModules() const {
  return num_modules_;
}

string Edge::GetTiedEdgeName() {
  return tied_edge_name_;
}
/*
bool Edge::RequiresMemoryForDeriv() const {
 return false;
} 
*/

bool Edge::IsTied() {
  return is_tied_;
}

void Edge::SetImageSize(int image_size) {
  image_size_ = image_size;
}

void Edge::FOV(int* size, int* sep, int* pad1, int* pad2) const {
}

void Edge::InsertPolyak() {
}

void Edge::BackupCurrent() {
}
void Edge::LoadCurrentOnGPU() {
}
void Edge::LoadPolyakOnGPU() {
}

}
}
