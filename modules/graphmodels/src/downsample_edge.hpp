#ifndef DOWNSAMPLE_EDGE_H_
#define DOWNSAMPLE_EDGE_H_
#include "opencv2/graphmodels/edge.hpp"

namespace cv
{
namespace graphmodels
{

/** Implements a down-sampling edge.
 * Only integer down-sampling factors are supported.
 */ 
class DownSampleEdge : public Edge {
 public:
  DownSampleEdge(const config::Edge& edge_config);
  virtual void ComputeUp(Matrix& input, Matrix& output, bool overwrite);
  virtual void ComputeDown(Matrix& deriv_output, Matrix& input,
                           Matrix& output, Matrix& deriv_input, bool overwrite);
  virtual void SetImageSize(int image_size);

 private:
  const int sample_factor_;
};

}
}

#endif
