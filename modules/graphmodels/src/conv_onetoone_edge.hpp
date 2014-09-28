#ifndef CONV_ONETOONE_EDGE_H_
#define CONV_ONETOONE_EDGE_H_
#include "edge_with_weight.hpp"

namespace cv
{
namespace graphmodels
{

/** An edge with one-to-one connectivity over spatial locations.
 */ 
class ConvOneToOneEdge : public EdgeWithWeight {
 public:
  ConvOneToOneEdge(const config::Edge& edge_config);
  virtual void AllocateMemory(bool fprop_only);
  virtual void ComputeUp(Matrix& input, Matrix& output, bool overwrite);
  virtual void ComputeDown(Matrix& deriv_output, Matrix& input,
                           Matrix& output, Matrix& deriv_input, bool overwrite);
  virtual void ComputeOuter(Matrix& input, Matrix& deriv_output);

  virtual int GetNumModules() const { return num_modules_; }
  virtual void SetImageSize(int image_size);
 
 private:
  void AllocateMemoryBprop();
  void AllocateMemoryFprop();
};

}
}

#endif
