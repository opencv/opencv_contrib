#ifndef __OPENCV_DNN_LAYERS_MVN_LAYER_HPP__
#define __OPENCV_DNN_LAYERS_MVN_LAYER_HPP__
#include "../precomp.hpp"

namespace cv
{
namespace dnn
{

class MVNLayer : public Layer
{
    double eps;
    bool acrossChannels, normalizeVariance;

public:

    MVNLayer(LayerParams &params);
    void allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs);
    void forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs);
};

}
}
#endif