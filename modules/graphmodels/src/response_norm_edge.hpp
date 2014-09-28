#ifndef RESPONSE_NORM_EDGE_H_
#define RESPONSE_NORM_EDGE_H_
#include "opencv2/graphmodels/edge.hpp"

namespace cv
{
namespace graphmodels
{

/** Response Normalization across filters at the same location.
 */
class ResponseNormEdge : public Edge {
 public:
  ResponseNormEdge(const config::Edge& edge_config);
  virtual void SetTiedTo(Edge* e);
  virtual void AllocateMemory(bool fprop_only);
  virtual void ComputeUp(Matrix& input, Matrix& output, bool overwrite);
  virtual void ComputeDown(Matrix& deriv_output, Matrix& input,
                           Matrix& output, Matrix& deriv_input, bool overwrite);
  virtual void SetImageSize(int image_size);

  bool Blocked() const { return blocked_; }
  float AddScale() const { return add_scale_; }
  float PowScale() const { return pow_scale_; }
  float FracOfFilters() const { return frac_of_filters_response_norm_; }

 private:
  int num_filters_response_norm_;
  bool blocked_;
  float add_scale_, pow_scale_, frac_of_filters_response_norm_;
};

}
}

#endif
