#ifndef MULTIGPU_CONVNET_H_
#define MULTIGPU_CONVNET_H_
#include "convnet.hpp"

#include "opencv2/core.hpp"

namespace cv
{
namespace graphmodels
{

class CV_EXPORTS MultiGPUConvNet : public ConvNet {
 public:
  MultiGPUConvNet(const string& model_file);

 protected:
  virtual void Fprop(bool train);
  virtual void ComputeDeriv();
  virtual void GetLoss(vector<float>& error);
  virtual void Bprop(bool update_weights);

  bool broadcast_;
};

}
}

#endif
