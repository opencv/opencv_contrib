#ifndef __OPENCV_DNN_LAYERS_BLANK_LAYER_HPP__
#define __OPENCV_DNN_LAYERS_BLANK_LAYER_HPP__
#include "../precomp.hpp"

namespace cv
{
namespace dnn
{
    class BlankLayer : public Layer
    {
    public:

        BlankLayer(LayerParams&)
        {

        }

        void allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
        {
            outputs.resize(inputs.size());
            for (size_t i = 0; i < inputs.size(); i++)
                outputs[i].shareFrom(*inputs[i]);
        }

        void forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
        {
            for (size_t i = 0; i < inputs.size(); i++)
                outputs[i] = *inputs[i];
        }
    };
}
}
#endif