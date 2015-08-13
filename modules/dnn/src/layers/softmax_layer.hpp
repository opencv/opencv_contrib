#ifndef __OPENCV_DNN_LAYERS_SOFTMAX_LAYER_HPP__
#define __OPENCV_DNN_LAYERS_SOFTMAX_LAYER_HPP__
#include "../precomp.hpp"

namespace cv
{
namespace dnn
{
    class SoftMaxLayer : public Layer
    {
        int axis_, axis;
        Blob maxAggregator;

    public:
        SoftMaxLayer(LayerParams &params);
        void allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs);
        void forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs);
    };
}
}
#endif