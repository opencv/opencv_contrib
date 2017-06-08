#include "downsample_edge.hpp"

namespace cv
{
namespace graphmodels
{

DownSampleEdge::DownSampleEdge(const config::Edge& edge_config) :
  Edge(edge_config),
  sample_factor_(edge_config.sample_factor()) {}

void DownSampleEdge::SetImageSize(int image_size) {
  Edge::SetImageSize(image_size);
  num_modules_ = image_size / sample_factor_;
}

void DownSampleEdge::ComputeUp(Matrix& input, Matrix& output, bool overwrite) {
  Matrix::ConvDownSample(input, output, sample_factor_, image_size_);
}

void DownSampleEdge::ComputeDown(Matrix& deriv_output, Matrix& input,
                                 Matrix& output, Matrix& deriv_input, bool overwrite) {
  int scale_targets = overwrite ? 0 : 1;
  Matrix::ConvUpSample(deriv_output, deriv_input, sample_factor_,
                       sample_factor_ * image_size_, scale_targets);
}

}
}
