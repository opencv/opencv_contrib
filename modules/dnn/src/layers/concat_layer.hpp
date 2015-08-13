#ifndef __OPENCV_DNN_LAYERS_CONCAT_LAYER_HPP__
#define __OPENCV_DNN_LAYERS_CONCAT_LAYER_HPP__
#include "../precomp.hpp"

namespace cv
{
namespace dnn
{
    class ConcatLayer : public Layer
    {
        int axis;

    public:
        ConcatLayer(LayerParams& params);
        void allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs);
        void forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs);
    };
}
}
#endif