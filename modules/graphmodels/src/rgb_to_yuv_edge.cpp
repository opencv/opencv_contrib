#include "rgb_to_yuv_edge.hpp"

namespace cv
{
namespace graphmodels
{

RGBToYUVEdge::RGBToYUVEdge(const config::Edge& edge_config) :
  Edge(edge_config), image_size_(0) {}

void RGBToYUVEdge::AllocateMemory(int image_size) {
  image_size_ = image_size;
}

void RGBToYUVEdge::ComputeUp(Matrix& input, Matrix& output, bool overwrite) {
  Matrix::ConvRGBToYUV(input, output);
}

void RGBToYUVEdge::ComputeDown(Matrix& deriv_output, Matrix& input,
                               Matrix& output, Matrix& deriv_input,
                               bool overwrite) {
  cerr << "RGBtoYUV backprop Not implemented." << endl;
  exit(1);
}

}
}
