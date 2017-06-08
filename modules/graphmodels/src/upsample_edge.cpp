#include "upsample_edge.hpp"

namespace cv
{
namespace graphmodels
{

UpSampleEdge::UpSampleEdge(const config::Edge& edge_config) :
  Edge(edge_config),
  sample_factor_(edge_config.sample_factor()) {}

void UpSampleEdge::SetImageSize(int image_size) {
  Edge::SetImageSize(image_size);
  num_modules_ = image_size * sample_factor_;
}

void UpSampleEdge::ComputeUp(Matrix& input, Matrix& output, bool overwrite) {
  int scale_targets = overwrite ? 0 : 1;
  Matrix::ConvUpSample(input, output, sample_factor_, image_size_, scale_targets);
}

void UpSampleEdge::ComputeDown(Matrix& deriv_output, Matrix& input,
                               Matrix& output, Matrix& deriv_input,
                               bool overwrite) {
  Matrix::ConvDownSample(deriv_output, deriv_input, sample_factor_,
                         sample_factor_ * image_size_);
}

}
}
