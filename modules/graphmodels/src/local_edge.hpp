#ifndef LOCAL_EDGE_H_
#define LOCAL_EDGE_H_
#include "edge_with_weight.hpp"

namespace cv
{
namespace graphmodels
{

/** Implements a locally connected edge.*/
class LocalEdge : public EdgeWithWeight {
 public:
  LocalEdge(const config::Edge& edge_config);
  virtual void AllocateMemory(bool fprop_only);
  virtual void ComputeUp(Matrix& input, Matrix& output, bool overwrite);
  virtual void ComputeDown(Matrix& deriv_output, Matrix& input,
                           Matrix& output, Matrix& deriv_input, bool overwrite);
  virtual void ComputeOuter(Matrix& input, Matrix& deriv_output);

  virtual void SetTiedTo(Edge* e);
  virtual void DisplayWeights();
  
  virtual int GetNumModules() const { return num_modules_; }
  virtual void SetImageSize(int image_size);
  virtual void FOV(int* size, int* sep, int* pad1, int* pad2) const;

  int GetKernelSize() const { return kernel_size_; }
  int GetStride() const { return stride_; }
  int GetPadding() const { return padding_; }
 
 private:
  void AllocateMemoryBprop();
  void AllocateMemoryFprop();

  int kernel_size_, stride_, padding_;
};

}
}

#endif
