#ifndef RGB_TO_YUV_EDGE_H_
#define RGB_TO_YUV_EDGE_H_
#include "opencv2/graphmodels/edge.hpp"

namespace cv
{
namespace graphmodels
{

/** Implements an edge that maps RGB to YUV.*/
class RGBToYUVEdge : public Edge {
 public:
  RGBToYUVEdge(const config::Edge& edge_config);
  virtual void AllocateMemory(int image_size);
  virtual void ComputeUp(Matrix& input, Matrix& output, bool overwrite);
  virtual void ComputeDown(Matrix& deriv_output, Matrix& input,
                           Matrix& output, Matrix& deriv_input, bool overwrite);
  virtual int GetNumModules() const { return image_size_; }

 private:
  int image_size_;
};

}
}

#endif
