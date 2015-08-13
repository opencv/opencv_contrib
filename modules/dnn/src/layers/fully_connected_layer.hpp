#ifndef __OPENCV_DNN_LAYERS_FULLY_CONNECTED_LAYER_HPP__
#define __OPENCV_DNN_LAYERS_FULLY_CONNECTED_LAYER_HPP__
#include "../precomp.hpp"

namespace cv
{
namespace dnn
{
    class FullyConnectedLayer : public Layer
    {
        bool bias;
        int numOutputs;
        int axis_, axis;

        int innerSize;

        void reshape(const Blob &inp, Blob &out);

    public:
        FullyConnectedLayer(LayerParams &params);
        void allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs);
        void forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs);
    };
}
}
#endif